import json
import os

from openai import OpenAI
from typing import Optional


class OpenAi:
    def __init__(self, config_dict: dict):
        """
        Wrapper to load and use OpenAI API for prediction.

        :param config_dict: dictionary with model name and OpenAI API key
        """
        self._client = OpenAI(api_key=config_dict["openai_api_key"])
        self._model_name = config_dict["model_name"]

    def predict(self, text: str, images: Optional[list[object]] = None, parameters: Optional[dict] = None):
        """
        Returns a response from the LLM.

        :param text: input text for the LLM
        :param images: image from which to extract data for the LLM
        :param parameters: parameters for the LLM
        :return:
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

        llm_response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{
                "role": "user",
                "content": content_text if content_images is None else content_text + content_images
            }],
            response_format={"type": "json_object"},
            **parameters
        )
        return json.loads(llm_response.choices[0].message.content)
