import pandas as pd
from typing import Tuple, List
from torch.utils.data import Dataset
import torch
import sklearn.model_selection as skm


class YTCommentsDataset(Dataset):
    def __init__(self, video_title_list, comments_list, tokenizer, max_length=1024):
        # variables
        self.input_ids = []
        self.attn_masks = []
        self.comments = []

        # iterate through the dataset
        for video_title, comment in zip(video_title_list, comments_list):
            # text of the request to model and its answer
            req_ans_text = f"<startoftext>The YouTube video named '{video_title}' may have the following comment:" \
                           f" {comment}<endoftext>"

            # tokenize text
            tokenized_text_dict = tokenizer(req_ans_text,
                                            truncation=True,
                                            max_length=max_length,
                                            padding=max_length)

            # append to lists
            self.input_ids.append(torch.tensor(tokenized_text_dict['input_ids']))
            self.attn_masks.append(torch.tensor(tokenized_text_dict['attention_mask']))
            self.comments.append(comment)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.comments[idx]


def build_dataset(path_to_video_info_file: str = "../data/USvideos.csv",
                  path_to_video_comments_file: str = "../data/UScomments.csv"
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
    dataframe.drop(columns=["video_id"])

    # resulting dataframe has 2 columns: 'title', 'comment_text'
    return dataframe


def train_test_dataset_split(dataframe: pd.DataFrame, test_size_perc: float = 0.05) -> Tuple[List[str], List[str], List[str], List[str]]:
    X_train, y_train, X_test, y_test = \
        skm.train_test_split(dataframe['title'].toList(),
                             dataframe['comment_text'].toList(),
                             shuffle=False,
                             test_size=test_size_perc,
                             stratify=dataframe['comment_text'])

    return X_train, y_train, X_test, y_test
