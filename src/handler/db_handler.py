from abc import ABC, abstractmethod
from typing import List

class DBHandler(ABC):
    def __init__(self, table_name: str):
        """

        :param table_name:
        """
        self._table_name = table_name

    @abstractmethod
    def get_table_name(self) -> str:
        pass

    @abstractmethod
    def add_config(self, config_dict: dict, config_name: str) -> bool:
        pass

    @abstractmethod
    def update_config(self, config_dict: dict, config_name: str) -> bool:
        pass

    @abstractmethod
    def delete_config(self, config_name) -> bool:
        pass

    @abstractmethod
    def get_config(self, config_name) -> dict:
        pass

    @abstractmethod
    def get_all_config_names(self) -> List[str]:
        pass
