import click
import pandas as pd
from yaml import safe_load

with open("config.yaml", "r") as file:
    config = safe_load(file)["roberta"]


@click.command()
@click.option(
    '--df_tagged_path',
    required=True,
    help="path/to/df_tagged.parquet",
)
@click.option(
    '--save_path',
    required=True,
    help="path/to/prepared_df.parquet",
)
def get_final_df(df_tagged_path: str, save_path: str) -> None:
    "Подготавливает df для анализа"

    questions = config["questions"]
    data = pd.read_parquet(df_tagged_path)

    index_array = []
    for ind in data.index:
        for question in questions:
            index_array.append([
                data.loc[ind, "place_id"],
                data.loc[ind, "review_body"],
                question,
                data.loc[ind, f"{question}_data"],
                data.loc[ind, f"{question}_proba"],
            ])
    columns = [
        "place_id",
        "review",
        "question",
        "tags",
        "proba",
    ]
    final_df = pd.DataFrame().from_records(index_array, columns=columns)

    final_df.to_parquet(save_path)


if __name__ == "__main__":
    get_final_df()
