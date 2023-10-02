import os
from abc import ABC, abstractmethod
from pathlib import Path, PurePath

import sqlalchemy as sql
from dotenv import load_dotenv
from pandas import DataFrame, read_sql_query

load_dotenv()


class DataUtils(ABC):
    @abstractmethod
    def download_data(download_script_path: Path, file_name: str) -> None:
        """
        download_scropt: Path - путь к sql-файлу для загрузки
        """
        url = "postgresql://{}:{}@{}:{}/{}".format(
            os.environ.get("DATABASE_LOGIN"),
            os.environ.get("DATABASE_PASSWORD"),
            os.environ.get("DATABASE_HOST"),
            5432,
            os.environ.get("DATABASE_NAME"),
        )
        engine = sql.create_engine(url=url)

        with open(download_script_path, "r", encoding="utf-8") as query:
            data = read_sql_query(query.read(), engine)

        data.to_parquet(
            PurePath(os.environ.get("INITIAL_DATA_PATH"),
                     f"{file_name}.parquet")
        )
