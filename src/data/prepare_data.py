import pandas as pd
from yaml import safe_load

with open("config.yaml", 'r') as conf:
    config = safe_load(conf)
config = config["pathes"]["data"]


comments = pd.read_parquet(f"{config['initial_data']}/comments.parquet")
comments.query("not comment.isna() & not place_id.isna()", inplace=True)

tags = pd.read_parquet(f"{config['initial_data']}/tags.parquet")

df = pd.DataFrame().from_records(comments.comment.values)
df.insert(0, "place_id", comments.place_id.values)
df = df.merge(tags, left_on="place_id", right_on="place_id", how="left")
df.drop("id", axis=1, inplace=True)

df.to_parquet(f"{config['intermediate_data']}/comments_tags.parquet")
