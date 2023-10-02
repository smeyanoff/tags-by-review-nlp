from abc import ABC, abstractmethod
from os import environ
from pathlib import Path, PurePath

import sqlalchemy as sql
from dotenv import load_dotenv
from pandas import read_sql_query
from pyyaml import safeload

load_dotenv()
with open("config.yaml", "r", encoding="utf-8") as file:
    config = safeload(file)["pathes"]


class DataUtils(ABC):
    """
    class for downloading the data
    """
    @abstractmethod
    def download_data(
        self,
        download_script_path: Path,
        file_name: str,
    ) -> None:
        """
        download_scropt: Path - путь к sql-файлу для загрузки
        """
        url = "postgresql://{}:{}@{}:{}/{}".format(
            environ.get("DATABASE_LOGIN"),
            environ.get("DATABASE_PASSWORD"),
            environ.get("DATABASE_HOST"),
            5432,
            environ.get("DATABASE_NAME"),
        )
        engine = sql.create_engine(url=url)

        with open(download_script_path, "r", encoding="utf-8") as query:
            data = read_sql_query(query.read(), engine)

        data.to_parquet(
            PurePath(
                config["data"]["initial_data"],
                f"{file_name}.parquet",
            ),
        )
