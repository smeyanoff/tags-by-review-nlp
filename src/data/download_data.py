"В этом модуле реализованы утилиты для работы с данными"

from os import environ

import click
import sqlalchemy as sql
from dotenv import load_dotenv
from pandas import read_sql_query

load_dotenv()


@click.command()
@click.option(
    '--script_path',
    required=True,
    help="path/to/download_script.sql",
)
@click.option(
    '--save_path',
    required=True,
    help="path/to/save_df.parquet",
)
def download_data(
    script_path: str,
    save_path: str,
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

    data.to_parquet(save_path)


if __name__ == "__main__":
    download_data()
