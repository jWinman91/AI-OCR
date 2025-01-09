import json, time

from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
from loguru import logger
from typing import Optional


class LlamaCpp:
    def __init__(self, config_dict: dict) -> None:
        """
        Model wrapper for LlamaCpp (a wrapper to handle quantized models from huggingface).

        :param config_dict: dictionary containing the model path and other parameters
        """
        construct_params = config_dict.get("construct_params", {})

        if config_dict.get("clip_model_path", None) is not None:
            chat_handler = MiniCPMv26ChatHandler(clip_model_path=config_dict.get("clip_model_path"))
            construct_params["chat_handler"] = chat_handler

        self._model = Llama(model_path=config_dict.get("model_path"), **construct_params)

    def predict(self, text: str, images: Optional[list[object]] = None, parameters: Optional[dict] = None):
        """
        Returns a response from the LLM.

        :param text: input text for the LLM
        :param images: image for the LLM
        :param parameters: parameters for the LLM
        :return: response from the LLM
        """
        content_text = [{
            "type": "text",
            "text": text
        }]

        if images is not None:
            content_images = [{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }
            } for image in images]
        else:
            content_images = None

        t0 = time.time()
        llm_response = self._model.create_chat_completion(
            messages=[{
                "role": "user",
                "content": content_text if content_images is None else content_text + content_images
            }],
            response_format={"type": "json_object"},
            **parameters
        )
        logger.info(f"Inference duration was {time.time() - t0} sec.")
        return json.loads(llm_response["choices"][0]["message"]["content"])
