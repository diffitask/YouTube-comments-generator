import pandas as pd
import math
from typing import Tuple
import streamlit as st


@st.cache_data
def build_dataset(path_to_video_info_file: str = "data/USvideos.csv",
                  path_to_video_comments_file: str = "data/UScomments.csv"
                  ) -> pd.DataFrame:
    # video info
    df_video_info = pd.read_csv(path_to_video_info_file,
                                sep=',',
                                quotechar='"',
                                skipinitialspace=True,
                                on_bad_lines='skip',
                                header=0)
    df_video_info = df_video_info.drop(
        columns=["channel_title", "category_id", "tags", "views", "likes", "dislikes", "comment_total",
                 "thumbnail_link", "date"])
    # tags could be left as a feature

    # video comments
    df_comments = pd.read_csv(path_to_video_comments_file,
                              sep=',',
                              quotechar='"',
                              skipinitialspace=True,
                              on_bad_lines='skip',
                              header=0)
    df_comments = df_comments.drop(columns=["likes", "replies"])

    # concatenating dataframes
    dataframe = df_comments.merge(df_video_info, on="video_id", how="inner").drop_duplicates()

    return dataframe


@st.cache_data
def train_test_dataset_split(dataframe: pd.DataFrame, ratio: float = 0.8) -> Tuple[pd.Series, pd.Series]:
    rows_cnt = len(dataframe)
    train_rows_cnt = int(math.ceil(ratio * rows_cnt))

    train_frame = dataframe[:train_rows_cnt]
    test_frame = dataframe[train_rows_cnt:]

    return train_frame, test_frame
