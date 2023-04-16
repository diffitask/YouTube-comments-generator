import pandas as pd
import math
from typing import Tuple
from yt_dlp import YoutubeDL


def build_dataset(path_to_data_file: str = "../data/UScomments.csv", ratio: float = 0.8) -> Tuple[pd.Series, pd.Series]:
    dataframe = pd.read_csv(path_to_data_file,
                            sep=',',
                            quotechar='"',
                            skipinitialspace=True,
                            on_bad_lines='skip',
                            header=0)
    dataframe = dataframe.drop(columns=["likes", "replies"])

    rows_cnt = len(dataframe)
    train_rows_cnt = int(math.ceil(ratio * rows_cnt))

    train_frame = dataframe[:train_rows_cnt]
    test_frame = dataframe[train_rows_cnt:]

    return train_frame, test_frame


def get_youtube_video_info(youtube_video_id: str = "dQw4w9WgXcQ") -> Tuple[str, str]:
    video_link = "https://www.youtube.com/watch?v=" + youtube_video_id

    ydl = YoutubeDL({'outtmpl': '%(id)s.%(ext)s'})
    with ydl:
        info_dict = ydl.extract_info(video_link, download=False)
        video_title = info_dict['title']
        video_desc = info_dict['description']

    return video_title, video_desc
