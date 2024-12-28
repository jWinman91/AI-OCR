import torch

from transformers import AutoModel, AutoTokenizer
from typing import Optional


class TransformersAuto:
    def __init__(self, config_dict: dict) -> None:
        construct_params = config_dict.get("construct_params", {})
        construct_params["attn_implementation"] = "sdpa"
        construct_params["torch_dtype"] = torch.bfloat16
        construct_params["trust_remote_code"] = True

        self._model = AutoModel.from_pretrained(config_dict.get("model_path"), **construct_params)
        print(self._model)
        #self._model = self._model.eval().cuda()
        self._tokenizer = AutoTokenizer.from_pretrained(config_dict.get("model_path"), trust_remote_code=True)

    def predict(self, text: str, image: object, parameters: Optional[dict] = None) -> str:
        messages = [{
            "role": "user",
            "content": [image, text]
        }]

        return self._model.chat(
            image=None,
            msgs=messages,
            tokenizer=self._tokenizer,
            **parameters
        ).replace("```json","").encode("utf-8").decode().replace("```", "")
