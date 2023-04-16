import pandas as pd
import math
from typing import Tuple


def build_dataset(path_to_data_file: str = "../data/UScomments.csv", ratio: float = 0.8) -> Tuple[pd.Series, pd.Series]:
    dataframe = pd.read_csv(path_to_data_file,
                            sep='delimiter',
                            engine='python',
                            header=0)

    rows_cnt = len(dataframe)
    train_rows_cnt = int(math.ceil(ratio * rows_cnt))

    train_frame = dataframe[:train_rows_cnt]
    test_frame = dataframe[train_rows_cnt:]

    return train_frame, test_frame
