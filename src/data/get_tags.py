import importlib.util

import click
import pandas as pd
from yaml import safe_load

with open("config.yaml", "r") as file:
    config = safe_load(file)["roberta"]
spec = importlib.util.spec_from_file_location(
    "roberta",
    "src/models/roberta.py",
)
roberta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roberta)


def choose_list(list_data, col):
    try:
        return list(list_data[col])
    except IndexError:
        pass


@click.command()
@click.option(
    '--df_path',
    required=True,
    help="path/to/df.parquet",
)
@click.option(
    '--save_path',
    required=True,
    help="path/to/save_df.parquet",
)
def get_tagged_df(df_path: str, save_path: str) -> None:
    """
    helps get tags from df
    """
    df = pd.read_parquet(df_path).loc[:10]
    model = roberta.Pipeline()

    questions = config["questions"]
    proba = config["cut_off_proba"]

    for question in questions:
        df[question] = df.review_body.apply(
            model.get_tags,
            args=(question, proba),
        )

    answers = df.iloc[:, 8:].copy()

    questions_data = pd.DataFrame()
    for question in answers.columns:
        questions_data[question+"_data"] = answers[question].apply(
            choose_list, args=[0],
        )
        questions_data[question+"_proba"] = answers[question].apply(
            choose_list, args=[1],
        )
    data = df.iloc[:, :8].join(questions_data)
    data.to_parquet(save_path)


if __name__ == "__main__":
    get_tagged_df()
