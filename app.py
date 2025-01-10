import subprocess, uvicorn, os, argparse, glob, importlib, yaml

from collections import OrderedDict
from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
from huggingface_hub import hf_hub_download, snapshot_download
from loguru import logger
from pdf2image import convert_from_path
from typing import List, Annotated, Union

from src.ocr_modelling import OcrModelling
from src.plot_modelling import PlotModelling
from src.handler.sqlite_db_handler import SqliteDBHandler
from src.utils.api_models import ConfigModel, Prompt, MetaData


DESCRIPTION = """
"""


class App:
    def __init__(self, ip: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
        """
        Builds the App Object for the Server Backend.
        This class is essentially the API for the AI-OCR project.

        :param ip: ip to serve
        :param port: port to serve
        """
        self._ip = ip
        self._port = port
        self._debug = debug
        self._app = FastAPI(
            title="AI-OCR: Extracting data from images via GPT_4 or models from Huggingface 🤗",
            description=DESCRIPTION
        )
        self._model_db = SqliteDBHandler("config_models")
        self._unmodified_model_db = SqliteDBHandler("unmodified_config_models")

        # caching
        self._ocr_model_cache = OrderedDict()
        self._prompt_cache = OrderedDict()
        self._images = OrderedDict()

        config = self.load_yml("configs/startup_params.yaml")
        self._cache = config["cache"]
        self._prompts = config["prompts"]

        # instantiate LLM for prompt optimisation
        llm_config = config["llm_configs"]
        self._predict_params = llm_config["predict_params"]

        self._llm_model_name = llm_config["model_name"]
        if llm_config["config_dict"]["model_wrapper"] != "open_ai":
            self._download_model(llm_config["config_dict"])

        self._llm_model = self._instantiate_model(llm_config["config_dict"])

        self._configure_routes()

    @staticmethod
    def load_yml(configfile: str) -> dict:
        """
        Imports a YAML Configuration file

        :param configfile: Path to the YAML config file.
        :return: A dictionary containing the configuration data.
        """
        with open(configfile, "r") as b:
            try:
                data = yaml.safe_load(b)
            except yaml.YAMLError as err:
                logger.error(err)
        return data

    @staticmethod
    def _instantiate_model(config_dict: dict) -> object:
        """
        Instantiates a model object based on the config_dict

        :param config_dict: parameters for instantiation
        :return: instantiated model object
        """
        module_name = config_dict.get("model_wrapper")
        class_name = "".join(x.capitalize() for x in module_name.split("_"))
        module = importlib.import_module(f"src.model_wrapper.{module_name}")

        return getattr(module, class_name)(config_dict)

    @staticmethod
    def _download_model(config_dict: dict) -> None:
        """
        Downloads a model from the Huggingface Hub or a snapshot of a given repository.

        :param config_dict: dictionary containing the download configuration
        :return: None
        """
        repo_id = config_dict.pop("repo_id")
        access_token = config_dict.pop("access_token", None)
        file_name = config_dict.pop("file_name", None)
        clip_model_name = config_dict.pop("clip_model_name", None)

        subprocess.call(f"mkdir -p models/{repo_id}", shell=True)
        if file_name is None:
            snapshot_download(repo_id=repo_id, local_dir=f"models/{repo_id}", token=access_token)
            config_dict["model_path"] = f"models/{repo_id}"
        else:
            hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=f"models/{repo_id}", token=access_token)
            config_dict["model_path"] = f"models/{repo_id}/{file_name}"

        if clip_model_name is not None:
            config_dict["clip_model_path"] = f"models/{repo_id}/{clip_model_name}"
            hf_hub_download(repo_id=repo_id, filename=clip_model_name, local_dir=f"models/{repo_id}")

        logger.info(f"Finished downloading model {config_dict['model_path']}.")

    @staticmethod
    async def _save_image(image_file: UploadFile) -> str:
        """
        Saves an image file to the disk

        :param image_file: File object of the image
        :return: string of the path to the saved image
        """
        subprocess.call("mkdir -p tmp", shell=True)
        image_path = f"tmp/{image_file.filename}"
        with open(image_path, 'wb') as image:
            content = await image_file.read()
            image.write(content)

        if image_path.endswith(".pdf"):
            images = convert_from_path(image_path, 300)
            for i, image in enumerate(images):
                image.save(f"{image_path.split('.pdf')[0]}_{i}.png", "PNG")

        return image_path

    def _configure_routes(self) -> None:
        """
        Creates the route(s)

        :return: None
        """

        @self._app.get("/get_all_model_wrapper")
        async def get_all_model_wrapper() -> List[str]:
            """
            Returns all model wrappers that are currently stored in the model_wrapper directory.

            :return: list of model wrappers
            """
            model_wrapper_paths = glob.glob("src/model_wrapper/*.py")
            return list(map(lambda path: path.split("/")[-1].split(".")[0], model_wrapper_paths))

        @self._app.post("/insert_model")
        async def insert_model(model_config: Annotated[ConfigModel, Body(
            examples=[{
                "model_name": "MiniCPM-v-2_6",
                "config_dict": {
                    "model_wrapper": "llama_cpp",
                    "repo_id": "openbmb/MiniCPM-V-2_6-gguf",
                    "file_name": "ggml-model-Q4_K.gguf",
                    "clip_model_name": "mmproj-model-f16.gguf",
                    "construct_params": {
                        "n_ctx": 2048,
                        "n_gpu_layers": -1
                    },
                }
            }],
        )]
                               ) -> bool:
            """
            Adds a configuration of a model to the config db.
            :param model_config: dictionary containing the model name and the configuration
            :return: True if successfully added
            """
            all_config_names = self._unmodified_model_db.get_all_config_names()
            method = "add_config" if model_config.model_name not in all_config_names else "update_config"

            if model_config.config_dict["model_wrapper"] == "open_ai":
                openai_api_key = model_config.config_dict.pop("openai_api_key", None)
            else:
                openai_api_key = None

            getattr(self._unmodified_model_db, method)(model_config.config_dict, model_config.model_name)
            _ = model_config.config_dict.pop("_rev", None)

            try:
                if model_config.config_dict["model_wrapper"] == "open_ai":
                    if openai_api_key is None:
                        raise RuntimeError("No API key provided!")
                    model_config.config_dict["openai_api_key"] = openai_api_key
                else:
                    self._download_model(model_config.config_dict)

                getattr(self._model_db, method)(model_config.config_dict, model_config.model_name)
                logger.info(f"Finished {method} the model {model_config.model_name}.")
            except Exception as e:
                config_deleted_unmodified = self._unmodified_model_db.delete_config(model_config.model_name)
                logger.error(f"{e} - Config {'was' if config_deleted_unmodified else 'was not'} deleted again.")
                raise RuntimeError("Something went wrong during the download or saving the config file.")

            return True

        @self._app.post("/delete_models")
        async def delete_models(config_names: List[str]) -> bool:
            """
            Deletes a configuration of a model from the couchdb.
            If the config doesn't exist, an error will be raised.

            :param config_names: List of names of model configs that will be deleted \n
            :return: True if successfully deleted
            """
            for config_name in config_names:
                config = self._model_db.get_config(config_name)
                config_del = self._model_db.delete_config(config_name)
                config_del_unmodified = self._unmodified_model_db.delete_config(config_name)
                config_cache = self._ocr_model_cache.pop(config_name, None)

                if not config_del:
                    self._unmodified_model_db.add_config(config, config_name)
                    self._ocr_model_cache[config_name] = config
                    logger.error(f"Model config {config_name} could not be deleted from model_db.")
                    return False
                if not config_del_unmodified:
                    self._model_db.add_config(config, config_name)
                    self._ocr_model_cache[config_name] = config
                    logger.error(f"Model config {config_name} could not be deleted from unmodified_model_db.")
                    return False

                try:
                    if config["model_wrapper"] != "open_ai":
                        subprocess.call(f"rm {config['model_path']}", shell=True)
                        subprocess.call(f"rm {config['clip_model_path']}", shell=True)

                    logger.info(f"Successfully deleted model {config_name}.")
                except Exception as e:
                    logger.error(f"Model config {config_name} could not be deleted from disk.")
                    return False

            return True

        @self._app.get("/get_all_unmodified_models")
        async def get_all_unmodified_models() -> dict:
            """
            Returns all configured models that are currently stored in the couchdb.
            Returns the configurations in unmodified form.

            :return: Dictionary of all model configs
            """
            config = {}
            all_models = self._unmodified_model_db.get_all_config_names()
            for model_name in all_models:
                config[model_name] = self._unmodified_model_db.get_config(model_name)

            return config

        @self._app.post("/upload_images")
        async def upload_images(images: List[UploadFile]) -> bool:
            """
            Uploads images to the server and saves them in the tmp folder.

            :param images: List of images to be uploaded
            :return: True if successfully uploaded
            """
            subprocess.call("rm -r tmp", shell=True)

            for image in images:
                self._images[image.filename] = await self._save_image(image)
                logger.info(f"Image {image.filename} was saved in {self._images[image.filename]}.")

            return True

        @self._app.post("/recognize_values")
        async def recognize_values(input_json: Annotated[Prompt, Body(
            examples=[{
                "prompt": "",
                "model_name": "MiniCPM-v-2_6",
                "parameters": {
                    "temperature": 0,
                    "top_p": 0.1
                }
            }]
        )],
                                   image_name: str
                                   ) -> dict:
            """
            Recognizes values (described in prompt) from an image using an OCR model.

            :param input_json: dictionary containing the prompt, the model name and additional parameters
            :param image_name: name of image from which to recognize values
            :return: dictionary containing the recognized values
            """
            config_dict = self._model_db.get_config(input_json.model_name)
            model = self._ocr_model_cache.get(input_json.model_name, None)
            prompt = self._prompt_cache.get(input_json.prompt, None)

            if model is None:
                model = self._instantiate_model(config_dict)
                self._ocr_model_cache[input_json.model_name] = model
                logger.info(f"Saved {input_json.model_name} in cache.")
            else:
                logger.info(f"Retrieved {input_json.model_name} from cache.")

            # instantiate ocr model
            ocr_model = OcrModelling(model, self._llm_model, self._prompts)

            if prompt is None:
                prompt = ocr_model.enhance_prompt(input_json.prompt, self._predict_params)
                self._prompt_cache[input_json.prompt] = prompt
                logger.info(f"Saved prompt in cache.")
            else:
                logger.info(f"Retrieved prompt from cache.")

            ocr_dict = ocr_model.run_ocr(prompt, self._images[image_name], input_json.parameters)

            if len(self._ocr_model_cache) > self._cache["max_number_models"]:
                self._ocr_model_cache.popitem(last=False)

            if len(self._prompt_cache) > self._cache["max_number_prompts"]:
                self._prompt_cache.popitem(last=False)

            subprocess.call(f"rm {self._images[image_name]}", shell=True)
            if self._images[image_name].endswith(".pdf"):
                subprocess.call(f"rm {self._images[image_name].split('.pdf')[0]}*.png", shell=True)
            _ = self._images.pop(image_name)

            return ocr_dict

        @self._app.post("/plot_suggestions")
        async def plot_suggestions(input_json: Annotated[MetaData, Body(
            examples=[{
                "meta_data_df": {
                    "dtypes": {"col_0": "int", "col_1": "float", "col_2": "string"},
                    "describe": {
                        "col_0": {"mean": 0, "std": 0, "min": 0, "max": 0},
                        "col_1": {"mean": 0, "std": 0, "min": 0, "max": 0},
                        "col_2": {"mean": 0, "std": 0, "min": 0, "max": 0}
                    }
                },
                "parameters": {
                    "temperature": 0,
                    "top_p": 0.1
                }
            }]
        )]) -> dict[str, List[str]]:
            """
            Returns suggestions for possible plots based on the metadata of the dataframe.

            :param input_json: dictionary containing the metadata of the dataframe and additional LLM parameters
            :return: dictionary containing a list of suggestions
            """
            # instantiate plot modelling class
            plot_modelling = PlotModelling(None, self._llm_model, self._prompts)
            logger.info("Suggesting prompts.")
            return {
                "list_of_suggestions": plot_modelling.suggest_prompt(input_json.meta_data_df, input_json.parameters)
            }

        @self._app.post("/plot_code")
        async def plot_code(input_json: Annotated[MetaData, Body(
            examples=[{
                "meta_data_df": {
                    "dtypes": {"col_0": "int", "col_1": "float", "col_2": "string"},
                    "describe": {
                        "col_0": {"mean": 0, "std": 0, "min": 0, "max": 0},
                        "col_1": {"mean": 0, "std": 0, "min": 0, "max": 0},
                        "col_2": {"mean": 0, "std": 0, "min": 0, "max": 0}
                    }
                },
                "model_name": "open_ai",
                "prompt_suggestion": "plot the data",
                "parameters": {
                    "temperature": 0,
                    "top_p": 0.1
                }
            }]
        )]) -> dict[str, Union[str, bool | None]]:
            """
            Creates code from a GenAI model to plot the data based on a plot suggestion
            and the metadata of the dataframe.

            :param input_json: dictionary containing the metadata of the dataframe, the model name, the prompt suggestion
            :return: dictionary containing the code to plot the data
            """
            # instantiate model
            config_dict = self._model_db.get_config(input_json.model_name)
            model = self._ocr_model_cache.get(input_json.model_name, None)

            if model is None:
                model = self._instantiate_model(config_dict)
                self._ocr_model_cache[input_json.model_name] = model
                logger.info(f"Saved {input_json.model_name} in cache.")
            else:
                logger.info(f"Retrieved {input_json.model_name} from cache.")

            plot_modelling = PlotModelling(model, self._llm_model, self._prompts)
            if prompt_check := plot_modelling.check("prompt", input_json.prompt_suggestion, input_json.parameters):
                code = plot_modelling.plot_prompt(input_json.meta_data_df, input_json.prompt_suggestion,
                                                  input_json.parameters)

                logger.info(f"Created code: {code}")

                if code_check := plot_modelling.check("code", code, input_json.parameters):
                    return {"code": code, "prompt_check": prompt_check, "code_check": code_check}
            else:
                code_check = None

            return {"code": False, "prompt_check": prompt_check, "code_check": code_check}

    def run(self) -> None:
        """
        Run the api
        :return: None
        """
        uvicorn.run(self._app, host=self._ip, port=self._port)
        subprocess.call("rm -r tmp", shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Host AI-NER.')
    parser.add_argument('-p', '--port', type=int, default=5000, help='the TCP/Port value')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('localaddress', nargs='*', help='the local Address where the server will listen')

    args = parser.parse_args()

    os.environ["COUCHDB_USER"] = "admin"
    os.environ["COUCHDB_PASSWORD"] = "JensIsCool"
    os.environ["COUCHDB_IP"] = "127.0.0.1:5984"

    api = App(ip=args.localaddress[0], port=args.port, debug=args.debug)
    api.run()
