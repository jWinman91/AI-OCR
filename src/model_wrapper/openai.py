import json
import os

from openai import OpenAI
from typing import Optional


class OpenAi:
    def __init__(self, config_dict: dict):
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model_name = config_dict["model_name"]

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

        llm_response = self._client.chat.completions.create(
            model=self._model_name,
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
        return json.loads(llm_response.choices[0].message.content)
