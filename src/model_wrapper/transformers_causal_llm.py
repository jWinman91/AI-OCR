import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor
)
from typing import Optional


class TransformersCasualLLM:
    def __init__(self, config_dict: dict) -> None:
        """
        Wrapper to load and use LLMs via vLLM for prediction.
        Args:
            model_name: name of LLM
            **kwargs: dictionary of arguments
        """
        construct_params = config_dict.get("construct_params", {})
        self._model = AutoModelForCausalLM.from_pretrained(config_dict.get("model_path"), device_map="auto", **construct_params)
        self._tokenizer = AutoTokenizer.from_pretrained(config_dict.get("model_path"), use_fast=True)
        self._processor = AutoProcessor.from_pretrained(config_dict.get("model_path"), trust_remote_code=True)

    def predict(self, text: str, images: Optional[list[object]] = None, parameters: Optional[dict] = None) -> str:
        messages = [{
            "role": "user",
            "content": text
        }]
        if images is not None:
            prompt = self._processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(prompt, images, return_tensors="pt").to("cuda")
        else:
            inputs = self._tokenizer(messages, return_tensors="pt").to("cuda")

        output = self._model.generate(**inputs, **parameters)
        return json.loads(self._tokenizer.decode(output[0][len(inputs[0]):], skip_special_tokens=True).replace("```json",
                                                                                                    "").encode(
            "utf-8").decode().replace("```", ""))
