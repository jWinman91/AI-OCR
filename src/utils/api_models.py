from pydantic import BaseModel
from typing import Optional


class ConfigModel(BaseModel):
    model_name: str
    config_dict: dict


class Prompt(BaseModel):
    prompt: str
    model_name: str
    parameters: dict


class MetaData(BaseModel):
    meta_data_df: dict
    parameters: dict
    model_name: Optional[str] = None
    prompt_suggestion: Optional[str] = None

