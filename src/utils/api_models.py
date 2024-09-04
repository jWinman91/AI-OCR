from pydantic import BaseModel


class ConfigModel(BaseModel):
    model_name: str
    config_dict: dict


class Prompt(BaseModel):
    prompt: str
    model_name: str
    parameters: dict

