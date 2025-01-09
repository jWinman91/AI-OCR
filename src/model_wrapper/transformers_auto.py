import json, torch

from transformers import AutoModel, AutoTokenizer
from typing import Optional


class TransformersAuto:
    def __init__(self, config_dict: dict) -> None:
        """
        Model wrapper to load and use LLMs via transformers library for prediction.
        This class is based on the description found for this model: openbmb/MiniCPM-V-2_6

        :param config_dict: dictionary containing the configuration for the model
        """
        construct_params = config_dict.get("construct_params", {})
        construct_params["attn_implementation"] = "sdpa"
        construct_params["torch_dtype"] = torch.bfloat16
        construct_params["trust_remote_code"] = True

        self._model = AutoModel.from_pretrained(config_dict.get("model_path"), **construct_params)
        self._model = self._model.eval().cuda()
        self._tokenizer = AutoTokenizer.from_pretrained(config_dict.get("model_path"), trust_remote_code=True)

    def predict(self, text: str, images: list[object] = None, parameters: Optional[dict] = None) -> str:
        """
        Returns a response from the LLM.

        :param text: input text for the LLM
        :param images: image for the LLM
        :param parameters: additional parameters for the LLM
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

        messages = [{
            "role": "user",
            "content": content_images + content_text
        }]

        return json.loads(self._model.chat(
            image=None,
            msgs=messages,
            tokenizer=self._tokenizer,
            **parameters
        ).replace("```json","").encode("utf-8").decode().replace("```", ""))
