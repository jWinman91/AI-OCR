import json
import sqlite3
import uuid

import sys
sys.path.append("../..")

from typing import List, Union
from loguru import logger
from src.handler.db_handler import DBHandler


class SqliteDBHandler(DBHandler):
    def __init__(self, table_name: str) -> None:
        super().__init__(table_name)
        self._db_path: str = "config.db"
        self._db_schema = {
            "config_name": "TEXT NOT NULL UNIQUE",
            "config_dict": "TEXT NOT NULL"
        }

        # Create the main config Table if not present
        self.create_table()

    def get_table_name(self) -> str:
        return self._table_name

    def connect(self) -> sqlite3.Connection:
        """
        Will connect the database
        Returns:
            connection
        """
        return sqlite3.connect(self._db_path)

    @staticmethod
    def close(con: sqlite3.Connection) -> None:
        """
        Will close the DB connection
        Args:
            con: sqlite Connection
        Returns:
            None
        """
        con.close()

    def create_table(self) -> bool:
        """
        Will create the main table for storing configs if not exists
        Returns:
            bool of successfully
        """
        con = self.connect()
        cursor = con.cursor()
        try:
            columns = ",".join([f"{field_name} {constraints}" for field_name, constraints in self._db_schema.items()])
            # create config table
            query = f"""CREATE TABLE IF NOT EXISTS {self._table_name} (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                       {columns});"""
            cursor.execute(query)
            con.commit()
        except sqlite3.OperationalError as e:
            logger.error(f"OperationalError aufgetreten: {e}")
            return False
        except sqlite3.DatabaseError as e:
            logger.error(f"DatabaseError aufgetreten: {e}")
            return False
        except Exception as e:
            logger.error(f"Unerwarteter Fehler aufgetreten: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
            if con:
                self.close(con)
        return True

    def get_config(self, config_name: str) -> Union[dict, None]:
        con = self.connect()
        cursor = con.cursor()
        try:
            query = f"""SELECT config_dict FROM {self._table_name} WHERE config_name = ?;"""
            cursor.execute(query, (config_name.lower(),))
            result = cursor.fetchone()
        except sqlite3.OperationalError as e:
            logger.error(f"OperationalError aufgetreten: {e}")
            return None
        except sqlite3.DatabaseError as e:
            logger.error(f"DatabaseError aufgetreten: {e}")
            return None
        except Exception as e:
            logger.error(f"Unerwarteter Fehler aufgetreten: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            if con:
                self.close(con)
        if result is None:
            return {}

        config_dict = json.loads(result[0])
        _ = config_dict.pop("uuid")
        return config_dict

    def add_config(self, config_dict: dict, config_name: str) -> bool:
        result = True
        con = self.connect()
        cursor = con.cursor()
        uid = str(uuid.uuid4())  # create standard conform uuid4
        query = f"""INSERT INTO {self._table_name} ({",".join(list(self._db_schema.keys()))}) VALUES (?, ?);"""
        config_dict["uuid"] = uid
        try:
            cursor.execute(query, (config_name.lower(), json.dumps(config_dict)), )
            con.commit()
        except sqlite3.OperationalError as e:
            logger.error(f"OperationalError aufgetreten: {e}")
            return False
        except sqlite3.DatabaseError as e:
            logger.error(f"DatabaseError aufgetreten: {e}")
            return False
        except Exception as e:
            logger.error(f"Unerwarteter Fehler aufgetreten: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
            if con:
                self.close(con)
        return result

    def update_config(self, config_dict: dict, config_name: str) -> bool:
        r = True
        con = self.connect()
        cursor = con.cursor()
        uid = str(uuid.uuid4())  # create standard conform uuid4
        config_dict["uuid"] = uid
        query = f"""SELECT config_dict FROM {self._table_name} WHERE config_name = ?;"""
        try:
            cursor.execute(query, (config_name.lower(),))
            result = cursor.fetchone()
            # If result is None - return False for nonexistent config
            if result is None:
                return False
        except sqlite3.OperationalError as e:
            logger.error(f"OperationalError aufgetreten: {e}")
            return False  # return an Error
        except sqlite3.DatabaseError as e:
            logger.error(f"DatabaseError aufgetreten: {e}")
            return False  # return an Error
        except Exception as e:
            logger.error(f"Unerwarteter Fehler aufgetreten: {e}")
            return False  # return an Error
        finally:
            if cursor:
                cursor.close()
            if con:
                self.close(con)

        # Check if json is inserted into string and create dict
        if isinstance(json.loads(result[0]), str):
            result = json.loads(json.loads(result[0]))
        else:
            result = json.loads(result[0])

        # check if there is a key not in the old config delete it
        d = []
        for key in result.keys():
            if key not in config_dict.keys():
                d.append(key)

        # remove deleted keys
        for key in d:
            result.pop(key)

        # Update settings
        result.update(config_dict)
        # create string
        result = json.dumps(result)
        # update values in the database
        query = f"""UPDATE {self._table_name} SET config_dict = ? WHERE config_name = ?;"""
        try:
            con = self.connect()
            cursor = con.cursor()
            cursor.execute(query, (result, config_name.lower(),))
            con.commit()
        except sqlite3.OperationalError as e:
            logger.error(f"OperationalError aufgetreten: {e}")
            return False
        except sqlite3.DatabaseError as e:
            logger.error(f"DatabaseError aufgetreten: {e}")
            return False
        except Exception as e:
            logger.error(f"Unerwarteter Fehler aufgetreten: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
            if con:
                self.close(con)
        return r

    def delete_config(self, config_name: str) -> bool:
        con = self.connect()
        cursor = con.cursor()
        try:
            query = f"DELETE FROM {self._table_name} WHERE config_name = ?"
            cursor.execute(query, (config_name.lower(),))
            con.commit()
        except sqlite3.OperationalError as e:
            logger.error(f"OperationalError aufgetreten: {e}")
            return False
        except sqlite3.DatabaseError as e:
            logger.error(f"DatabaseError aufgetreten: {e}")
            return False
        except Exception as e:
            logger.error(f"Unerwarteter Fehler aufgetreten: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
            if con:
                self.close(con)
        return True

    def get_all_config_names(self) -> List[str]:
        con = self.connect()
        cursor = con.cursor()
        try:
            query = f"""SELECT config_name FROM {self._table_name};"""
            cursor.execute(query)
            results = cursor.fetchall()
        except sqlite3.OperationalError as e:
            logger.error(f"OperationalError aufgetreten: {e}")
            return []
        except sqlite3.DatabaseError as e:
            logger.error(f"DatabaseError aufgetreten: {e}")
            return []
        except Exception as e:
            logger.error(f"Unerwarteter Fehler aufgetreten: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            if con:
                self.close(con)
        if results is None:
            return []

        return [result[0] for result in results]


if __name__ == "__main__":
    sqlite_handler = SqliteDBHandler(table_name="config_models")
    sqlite_handler_2 = SqliteDBHandler(table_name="unmodified_config_models")

    config_name_llama = "test_llama"
    config_dict_llama = {
        "model_wrapper": "llama_cpp",
        "repo_id": "test",
        "file_name": "test_file.gguf"
    }
    config_dict_llama_update = {
        "model_wrapper": "llama_cpp",
        "repo_id": "test_update"
    }

    config_name_openai = "test_openai"
    config_dict_openai = {
        "model_wrapper": "open_ai",
        "model_name": "chatgpt"
    }

    print("Add config: ", sqlite_handler.add_config(config_dict=config_dict_llama, config_name=config_name_llama))
    print("Config: ", sqlite_handler.get_config(config_name=config_name_llama))

    print("Update config: ", sqlite_handler.update_config(config_name=config_name_llama, config_dict=config_dict_llama_update))
    print("Config: ", sqlite_handler.get_config(config_name=config_name_llama))

    print("Add config: ", sqlite_handler.add_config(config_dict=config_dict_openai, config_name=config_name_openai))
    print("Config: ", sqlite_handler.get_config(config_name=config_name_openai))

    print("All config files: ", sqlite_handler.get_all_config_names())

    print("Delete config:", sqlite_handler.delete_config(config_name=config_name_openai))
    print("All config files: ", sqlite_handler.get_all_config_names())
    print("All config files in second table: ", sqlite_handler_2.get_all_config_names())
