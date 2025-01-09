import base64
import datetime
import glob

from PIL import Image
from loguru import logger


class OcrModelling:
    def __init__(self, model: object, llm_model: object, prompts: dict) -> None:
        self._model = model
        self._llm_model = llm_model
        self.ENHANCE_PROMPT_PROMPT = prompts["enhance_prompt"]
        self.PROMPT_TEMPLATE = prompts["template"]

    # Function to encode the image
    @staticmethod
    def _encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def enhance_prompt(self, prompt: str, parameters: dict) -> str:
        list_of_names = self._llm_model.predict(self.ENHANCE_PROMPT_PROMPT.format(input=prompt),
                                                parameters=parameters)["namen"]
        json_ausdruck = "{" + ", ".join(f'"{name}": "zahl"' for name in list_of_names) + "}"

        logger.info(f"json of value names: {json_ausdruck}")

        return self.PROMPT_TEMPLATE.format(prompt=prompt, json_ausdruck=json_ausdruck)

    def run_ocr(self, prompt: str, image_path: str, parameters: dict) -> dict:
        if image_path.endswith(".pdf"):
            image_paths = glob.glob(image_path.split(".pdf")[0] + "*.png")
            images = [self._encode_image(image) for image in image_paths]
        else:
            images = [self._encode_image(image_path)]

        ocr_dict = self._model.predict(prompt, images=images, parameters=parameters)
        ocr_dict["image_name"] = image_path.split("/")[1]
        logger.info(ocr_dict)

        if not image_path.endswith(".pdf"):
            exif = Image.open(image_path)._getexif()
            if exif is not None and len(exif) > 36867:
                ocr_dict["creation_date"] = datetime.datetime.strptime(exif[36867], "%Y:%m:%d %H:%M:%S")

        return ocr_dict



