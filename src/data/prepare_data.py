import click
import pandas as pd


@click.command()
@click.option(
    '--comments_path',
    required=True,
    help="path/to/comments.parquet",
)
@click.option(
    '--tags_path',
    required=True,
    help="path/to/tags.parquet",
)
@click.option(
    '--save_path',
    required=True,
    help="path/to/prepared_df.parquet",
)
def prepare_data(
    comments_path: str,
    tags_path: str,
    save_path: str,
) -> None:
    comments = pd.read_parquet(comments_path)
    comments.query("not comment.isna() & not place_id.isna()", inplace=True)

    tags = pd.read_parquet(tags_path)

    df = pd.DataFrame().from_records(comments.comment.values)
    df.insert(0, "place_id", comments.place_id.values)
    df = df.merge(tags, left_on="place_id", right_on="place_id", how="left")
    df.drop("id", axis=1, inplace=True)

    df.to_parquet(save_path)


if __name__ == "__main__":
    prepare_data()
