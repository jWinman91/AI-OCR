import numpy as np

from loguru import logger
from typing import Literal


class PlotModelling:
    def __init__(self, model: object, llm_model: object, prompts: dict) -> None:
        """
        Class to generate python code for plotting a DataFrame with the help of an LLM

        :param model: object of the model for the code generation
        :param llm_model: object of the LLM model for prompt generation
        :param prompts: dictionary with prompts for the LLM model
        """
        self._model = model
        self._llm_model = llm_model
        self._suggestion_prompt = prompts["plot_suggestion"]
        self._check_prompt = prompts["check"]
        self._check_code = prompts["check_code"]
        self._prompt_template = prompts["plot_template"]

    @staticmethod
    def create_describe_text(describe_dict: dict) -> str:
        """
        Creates a text with bullet points from the describe dictionary

        :param describe_dict: dictionary with statistical description of the dataframe (created via df.describe())
        :return: string with bullet points
        """
        describe_text = ""
        for col_name, value in describe_dict.items():
            describe_text += f"- {col_name}:\n"
            for stat_property, stat_value in value.items():
                if type(stat_value) is str or (stat_value is not None and not np.isnan(stat_value)):
                    describe_text += f"\t- {stat_property}: {stat_value}\n"

        return describe_text

    def suggest_prompt(self, meta_data_df: dict, parameters: dict) -> list[str]:
        """
        Suggests prompts based on the metadata of the dataframe

        Example DataFrame:
        meta_data_df = {
            "dtypes": {"col_0": "int", "col_1": "float", "col_2": "string"},
            "describe": {
                "col_0": {"count": 10, "mean": 5.5, "std": 2.9, "min": 1, "25%": 3, "50%": 5.5, "75%": 8, "max": 10},
                "col_1": {"count": 10, "mean": 5.5, "std": 2.9, "min": 1, "25%": 3, "50%": 5.5, "75%": 8, "max": 10},
                "col_2": {"count": 10, "unique": 3, "top": "A", "freq": 4}
            }
        }

        :param meta_data_df: metadata of the dataframe in a dictionary
        :param parameters: additional parameters for the model
        :return: list of suggestions
        """
        column_names = ", ".join(list(meta_data_df["dtypes"].keys()))
        dtypes = ", ".join(list(meta_data_df["dtypes"].values()))
        statistical_description = self.create_describe_text(meta_data_df["describe"])

        prompt = self._suggestion_prompt.format(column_names=column_names, dtypes=dtypes,
                                                statistical_description=statistical_description)
        list_of_suggestions = self._llm_model.predict(prompt, parameters=parameters)["suggestions"]

        for i, suggestion in enumerate(list_of_suggestions):
            logger.info(f"Suggestion No {i}: {suggestion}")

        return list_of_suggestions

    def check(self, check_type: Literal["prompt", "code"], text: str, parameters: dict) -> bool:
        """
        Checks the prompt or code for malicious behaviour.

        :param check_type: prompt or code
        :param text: string to check
        :param parameters: parameters for the model
        :return: True/False if the check was successful
        """
        input_str = self._check_prompt.format(plot_suggestion=text) if check_type == "prompt"\
            else self._check_code.format(code=text)

        result = self._llm_model.predict(input_str, parameters=parameters)["result"]
        logger.info(f"Check result for {check_type}: {result}")

        return result

    def plot_prompt(self, meta_data_df: dict, plot_suggestion: str, parameters: dict) -> str:
        """
        Creates the code via the LLM model to plot the data.

        :param meta_data_df: metadata of the dataframe in a dictionary
        :param plot_suggestion: suggestion for the plot
        :param parameters: parameters for the model
        :return: string with the code
        """
        column_names = ", ".join(list(meta_data_df["dtypes"].keys()))
        dtypes = ", ".join(list(meta_data_df["dtypes"].values()))
        statistical_description = self.create_describe_text(meta_data_df["describe"])

        prompt = self._prompt_template.format(column_names=column_names, dtypes=dtypes,
                                              statistical_description=statistical_description,
                                              plot_suggestion=plot_suggestion)

        return self._model.predict(prompt, parameters=parameters)["code"]
