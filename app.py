import subprocess, uvicorn, os, argparse, glob

from collections import OrderedDict
from huggingface_hub import hf_hub_download, snapshot_download
from fastapi import FastAPI, HTTPException, Body, UploadFile
from typing import List, Annotated

from src.ocr_modelling import OcrModelling
from src.handler.couch_db_handler import CouchDBHandler
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
        self._model_db = CouchDBHandler("config_models")
        self._unmodified_model_db = CouchDBHandler("unmodified_config_models")
        self._ocr_model_cache = OrderedDict()
        self._prompt_cache = OrderedDict()

        self._configure_routes()

    @staticmethod
    def _download_model(config_dict: dict) -> None:
        """

        :param config_dict:
        :return:
        """
        repo_id = config_dict.pop("repo_id")
        file_name = config_dict.pop("file_name")
        clip_model_name = config_dict.pop("clip_model_name")

        config_dict["model_path"] = f"models/{repo_id}/{file_name}"
        config_dict["clip_model_path"] = f"models/{repo_id}/{clip_model_name}"

        subprocess.call(f"mkdir -p models/{repo_id}", shell=True)
        hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=config_dict["model_path"])
        hf_hub_download(repo_id=repo_id, filename=clip_model_name, local_dir=config_dict["model_path"])

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

        @self._app.post("/insert_model")
        async def insert_model(model_config: Annotated[ConfigModel, Body(
            examples={
                "model_name": "MiniCPM-v-2_6",
                "config_dict": {
                    "model_wrapper": "llama_cpp",
                    "repo_id": "openbmb/MiniCPM-V-2_6-gguf",
                    "file_name": "test",
                    "clip_model_name": "test",
                    "construct_params": {
                        "n_ctx": 2048,
                        "n_gpu_layers": -1
                    },
                }
            },
        )]
                               ) -> bool:
            """

            :param model_config:
            :return:
            """
            all_config_names = self._unmodified_model_db.get_all_config_names()
            method = "add_config" if model_config.model_name not in all_config_names else "update_config"

            getattr(self._unmodified_model_db, method)(model_config.config_dict, model_config.model_name)
            if model_config.config_dict["model_wrapper"] != "openai":
                self._download_model(model_config.config_dict)
            getattr(self._model_db, method)(model_config.config_dict, model_config.model_name)

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
                subprocess.call(f"rm {config['model_path']}", shell=True)
                subprocess.call(f"rm {config['clip_model_path']}", shell=True)
                self._model_db.delete_config(config_name)
                self._unmodified_model_db.delete_config(config_name)
                self._ocr_model_cache.pop(config_name, None)

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

        @self._app.post("/recognize_values")
        async def recognize_values(input_json: Prompt,
                                   image: UploadFile,
                                   ) -> dict:

            config_dict = self._model_db.get_config(input_json.model_name)
            ocr_model = self._ocr_model_cache.get(input_json.model_name, None)
            prompt = self._prompt_cache.get(input_json.prompt, None)

            if ocr_model is None:
                ocr_model = OcrModelling(config_dict)
                self._ocr_model_cache[input_json.model_name] = ocr_model

            if prompt is None:
                prompt = ocr_model.enhance_prompt(input_json.prompt)
                self._prompt_cache[input_json.prompt] = prompt

            if len(self._ocr_model_cache) > 2:
                self._ocr_model_cache.popitem(last=False)

            if len(self._prompt_cache) > 10:
                self._prompt_cache.popitem(last=False)

            if len(glob.glob("tmp/*")) > 10:
                subprocess.call("rm -r tmp")

            ocr_dict = ocr_model.run_ocr(prompt, self._save_image(image), prompt.parameters)

            if len(glob.glob("tmp/*")) > 10:
                subprocess.call("rm -r tmp")

            return ocr_dict

    def run(self) -> None:
        """
        Run the api
        :return: None
        """
        uvicorn.run(self._app, host=self._ip, port=self._port)


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
