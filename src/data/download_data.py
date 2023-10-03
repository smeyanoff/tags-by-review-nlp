"В этом модуле реализованы утилиты для работы с данными"

from os import environ
from pathlib import Path, PurePath

import click
import sqlalchemy as sql
from dotenv import load_dotenv
from pandas import read_sql_query
from yaml import safe_load

load_dotenv()
with open("config.yaml", "r", encoding="utf-8") as file:
    config = safe_load(file)["pathes"]


@click.command()
@click.option(
    '--script_path',
    required=True,
    help="path/to/download_script.file",
)
@click.option(
    '--file_name',
    required=True,
    help="entire file name",
)
def download_data(
    script_path: Path,
    file_name: str,
) -> None:
    """
    download_scropt: Path - путь к sql-файлу для загрузки
    """
    url = "postgresql://{login}:{password}@{host}:{port}/{dbname}".format(
        login=environ.get("DATABASE_LOGIN"),
        password=environ.get("DATABASE_PASSWORD"),
        host=environ.get("DATABASE_HOST"),
        port=5432,
        dbname=environ.get("DATABASE_NAME"),
    )
    engine = sql.create_engine(url=url)

    with open(script_path, "r", encoding="utf-8") as query:
        data = read_sql_query(query.read(), engine)

    data.to_parquet(
        PurePath(
            config["data"]["initial_data"],
            f"{file_name}.parquet",
        ),
    )


if __name__ == "__main__":
    download_data()
