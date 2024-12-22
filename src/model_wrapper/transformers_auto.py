import torch

from transformers import AutoModel, AutoTokenizer
from typing import Optional


class TransformersAuto:
    def __init__(self, model_name: str, **kwargs) -> None:
        self._model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, **kwargs)
        self._model = self._model.eval().cuda()
        self._tokenizer = AutoTokenizer.from_pretrained(self._model, trust_remote_code=True)

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
