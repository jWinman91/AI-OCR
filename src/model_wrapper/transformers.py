from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor
)
from typing import Optional


class Transformers:
    def __init__(self, model_name: str, **kwargs) -> None:
        """
        Wrapper to load and use LLMs via vLLM for prediction.
        Args:
            model_name: name of LLM
            **kwargs: dictionary of arguments
        """
        self._model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", **kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def predict(self, text: str, image: Optional[object] = None, parameters: Optional[dict] = None) -> str:
        messages = [{
            "role": "user",
            "content": text
        }]
        if image is not None:
            prompt = self._processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(prompt, [image], return_tensors="pt").to("cuda")
        else:
            inputs = self._tokenizer(messages, return_tensors="pt").to("cuda")

        output = self._model.generate(**inputs, **parameters)
        return self._tokenizer.decode(output[0][len(inputs[0]):], skip_special_tokens=True).replace("```json",
                                                                                                    "").encode(
            "utf-8").decode().replace("```", "")
