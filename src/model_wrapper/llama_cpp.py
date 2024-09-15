import json, time

from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
from loguru import logger
from typing import Optional


class LlamaCpp:
    def __init__(self, config_dict: dict) -> None:
        """

        :param model_path:
        :param clip_model_path:
        :param construct_params:
        """
        construct_params = config_dict.get("construct_params", {})

        if config_dict.get("clip_model_path", None) is not None:
            chat_handler = MiniCPMv26ChatHandler(clip_model_path=config_dict.get("clip_model_path"))
            construct_params["chat_handler"] = chat_handler

        self._model = Llama(model_path=config_dict.get("model_path"), **construct_params)

    def predict(self, text: str, image: Optional[object] = None, parameters: Optional[dict] = None):
        """

        :param text:
        :param image:
        :param parameters:
        :return:
        """
        content_text = {
            "type": "text",
            "text": text
        }

        if image is not None:
            content_image = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }
            }
        else:
            content_image = None

        t0 = time.time()
        llm_response = self._model.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [content_text] if content_image is None else [
                    content_text,
                    content_image
                ]
            }],
            response_format={"type": "json_object"},
            **parameters
        )
        logger.info(f"Inference duration was {time.time() - t0} sec.")
        return json.loads(llm_response["choices"][0]["message"]["content"])
