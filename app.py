import subprocess, uvicorn, os, argparse, glob, importlib

from collections import OrderedDict
from huggingface_hub import hf_hub_download, snapshot_download
from loguru import logger
from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
from typing import List, Annotated

from src.ocr_modelling import OcrModelling
from src.handler.sqlite_db_handler import SqliteDBHandler
from src.utils.api_models import ConfigModel, Prompt


DESCRIPTION = """
"""


class App:
    def __init__(self, ip: str = "127.0.0.1", port: int = 8000, debug: bool = False) -> None:
        """
        Builds the App Object for the Server Backend

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

        # instantiate LLM for prompt optimisation
        llm_config = {
            "model_name": "gemma-2-9b-it-gguf",
            "config_dict": {
                "model_wrapper": "llama_cpp",
                "repo_id": "bartowski/gemma-2-9b-it-GGUF",
                "file_name": "gemma-2-9b-it-Q4_K_M.gguf",
                "construct_params": {
                    "n_ctx": 2048,
                    "n_gpu_layers": -1
                }
            },
        }
        self._llm_model_name = llm_config["model_name"]
        self._download_model(llm_config["config_dict"])
        self._ocr_model_cache[self._llm_model_name] = self._instantiate_model(llm_config["config_dict"])

        self._configure_routes()

    @staticmethod
    def _instantiate_model(config_dict: dict) -> object:
        module_name = config_dict.get("model_wrapper")
        class_name = "".join(x.capitalize() for x in module_name.split("_"))
        module = importlib.import_module(f"src.model_wrapper.{module_name}")

        return getattr(module, class_name)(config_dict)

    @staticmethod
    def _download_model(config_dict: dict) -> None:
        """

        :param config_dict:
        :return:
        """
        repo_id = config_dict.pop("repo_id")
        file_name = config_dict.pop("file_name")
        clip_model_name = config_dict.pop("clip_model_name", None)

        subprocess.call(f"mkdir -p models/{repo_id}", shell=True)
        hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=f"models/{repo_id}")

        config_dict["model_path"] = f"models/{repo_id}/{file_name}"
        if clip_model_name is not None:
            config_dict["clip_model_path"] = f"models/{repo_id}/{clip_model_name}"
            hf_hub_download(repo_id=repo_id, filename=clip_model_name, local_dir=f"models/{repo_id}")

        logger.info(f"Finished downloading model {config_dict['model_path']}.")

    @staticmethod
    async def _save_image(image_file: UploadFile) -> str:
        subprocess.call("mkdir -p tmp", shell=True)
        image_path = f"tmp/{image_file.filename}"
        with open(image_path, 'wb') as image:
            content = await image_file.read()
            image.write(content)

        return image_path

    def _configure_routes(self) -> None:
        """
        Creates the route(s)

        :return: None
        """

        @self._app.get("/get_all_model_wrapper")
        async def get_all_model_wrapper() -> List[str]:
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

            :param model_config:
            :return:
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
            except Exception as e:
                self._unmodified_model_db.delete_config(model_config.model_name)
                logger.error(e)
                RuntimeError("Something went wrong during the download.")

            getattr(self._model_db, method)(model_config.config_dict, model_config.model_name)
            logger.info(f"Finished {method} the model {model_config.model_name}.")

            return True

        @self._app.post("/delete_models")
        async def delete_models(config_names: List[str]) -> bool:
            """
            Deletes a configuration of a model from the couchdb.
            If the config doesnt exist, an error will be raised.

            :param config_names: List of names of model configs that will be deleted \n
            :return: True if successfully deleted
            """
            for config_name in config_names:
                config = self._model_db.get_config(config_name)
                if config["model_wrapper"] != "open_ai":
                    subprocess.call(f"rm {config['model_path']}", shell=True)
                    subprocess.call(f"rm {config['clip_model_path']}", shell=True)
                self._model_db.delete_config(config_name)
                self._unmodified_model_db.delete_config(config_name)
                self._ocr_model_cache.pop(config_name, None)

                logger.info(f"Deleted model {config_name}.")

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

            :param images:
            :return:
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
            config_dict = self._model_db.get_config(input_json.model_name)
            model = self._ocr_model_cache.get(input_json.model_name, None)
            llm_model = self._ocr_model_cache.get(self._llm_model_name)
            prompt = self._prompt_cache.get(input_json.prompt, None)

            if model is None:
                model = self._instantiate_model(config_dict)
                self._ocr_model_cache[input_json.model_name] = model
                logger.info(f"Saved {input_json.model_name} in cache.")
            else:
                logger.info(f"Retrieved {input_json.model_name} from cache.")

            # instantiate ocr model
            ocr_model = OcrModelling(model, llm_model)

            if prompt is None:
                prompt = ocr_model.enhance_prompt(input_json.prompt)
                self._prompt_cache[input_json.prompt] = prompt
                logger.info(f"Saved prompt in cache.")
            else:
                logger.info(f"Retrieved prompt from cache.")

            ocr_dict = ocr_model.run_ocr(prompt, self._images[image_name], input_json.parameters)

            if len(self._ocr_model_cache) > 2:
                self._ocr_model_cache.popitem(last=False)

            if len(self._prompt_cache) > 10:
                self._prompt_cache.popitem(last=False)

            subprocess.call(f"rm {self._images[image_name]}", shell=True)
            _ = self._images.pop(image_name)

            return ocr_dict

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
