from abc import ABC, abstractmethod
from pathlib import Path, PurePath
from dotenv import load_dotenv

from pandas import DataFrame, read_sql_query
import sqlalchemy as sql

load_dotenv()

class DataUtils(ABC):

    @abstractmethod
    def download_data(
        DATABASE_LOGIN: str,
        DATABASE_PASSWORD: str,
        DATABASE_HOST: str,
        DATABASE_PORT: str,
        DATABASE_NAME: str,
        DOWNLOAD_SCRIPT: str,
        INITIAL_DATA_PATH: str,
        OUTPUT_FILE_NAME: str
    ) -> None:
        """
        download_scropt: Path - путь к sql-файлу для загрузки
        """
        url = "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
            DATABASE_LOGIN,
            DATABASE_PASSWORD,
            DATABASE_HOST,
            5432,
            DATABASE_NAME
        )
        engine = sql.create_engine(url=url)

        with open(DOWNLOAD_SCRIPT, "r") as query:
            comments = read_sql_query(query.read(), engine)
        
        comments.to_csv(PurePath(INITIAL_DATA_PATH, f"{OUTPUT_FILE_NAME}.csv"))

    # @abstractmethod
    # def prepare_data(initial_data_path: Path, intermediate_data_path) -> DataFrame:

        


