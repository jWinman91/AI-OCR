import base64
import datetime
import importlib
import yaml

from PIL import Image
from loguru import logger


class OcrModelling:
    ENHANCE_PROMPT_PROMPT = """
    Extrahiere alle Namen von Messgrößen aus dem folgenden Text.
    Gib das Ergebnis im json-Format zurück {{"namen": ["liste von namen"]}}.
    
    Input: "Lese den Zählerstand des abgebildeten Drehstromzählers ab."
    Output: {{"namen": ["zählerstand"]}}
    
    Input: "{input}"
    Output:
    """
    PROMPT_TEMPLATE = """
    {prompt}
    
    Gib das Ergebnis als JSON-Ausdruck in folgenden Format wider:
    {json_ausdruck}.
    Bevor du das Ergebnis ausgibst, stelle sicher, dass der Wert korrekt ist und vollständig erfasst wird.
    """

    def __init__(self, model: object, llm_model: object) -> None:
        self._model = model
        self._llm_model = llm_model

    @staticmethod
    def load_yml(configfile: str) -> dict:
        """
        Imports a YAML Configuration file
        :param configfile: Path to the YAML config file.
        :return: A dictionary containing the configuration data.
        """
        with open(configfile, "r") as b:
            try:
                data = yaml.safe_load(b)
            except yaml.YAMLError as err:
                logger.error(err)
        return data

    # Function to encode the image
    @staticmethod
    def _encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def enhance_prompt(self, prompt: str) -> str:
        parameters = {"temperature": 0}
        list_of_names = self._llm_model.predict(self.ENHANCE_PROMPT_PROMPT.format(input=prompt),
                                                parameters=parameters)["namen"]
        json_ausdruck = "{" + ", ".join(f'"{name}": "zahl"' for name in list_of_names) + "}"

        logger.info(f"json of value names: {json_ausdruck}")

        return self.PROMPT_TEMPLATE.format(prompt=prompt, json_ausdruck=json_ausdruck)

    def run_ocr(self, prompt: str, image_path: str, parameters: dict) -> dict:
        image = self._encode_image(image_path)

        ocr_dict = self._model.predict(prompt, image=image, parameters=parameters)
        ocr_dict["image_name"] = image_path.split("/")[1]
        logger.info(ocr_dict)

        exif = Image.open(image_path)._getexif()
        if exif is not None and len(exif) > 36867:
            ocr_dict["creation_date"] = datetime.datetime.strptime(exif[36867], "%Y:%m:%d %H:%M:%S")

        return ocr_dict



