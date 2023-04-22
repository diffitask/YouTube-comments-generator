from datasetBuilding import YTCommentsDataset, build_dataset, train_test_dataset_split
from modelUtils import build_model, train_model, score_model, save_model
import torch


# creating, training and saving model
def fine_tune_gpt2_model():
    # seed
    torch.manual_seed(42)

    # building model
    device, tokenizer, model = build_model('gpt2')

    # building dataset
    dataframe = build_dataset()
    X_train, y_train, X_test, y_test = train_test_dataset_split(dataframe)
    train_dataset = YTCommentsDataset(X_train, y_train, tokenizer)

    # train model
    train_model(model, train_dataset)

    # score model
    score_model(model, tokenizer, device, X_test, y_test)

    # save model
    # save_model(model)


if __name__ == '__main__':
    fine_tune_gpt2_model()
