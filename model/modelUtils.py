import re
import pandas as pd
import torch
import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from sklearn.metrics import f1_score


def build_model(gpt2_type: str = 'gpt2'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type,
                                              bos_token='<startoftext',
                                              eos_token='<endoftext',
                                              pad_token='<pad>')
    model = GPT2LMHeadModel.from_pretrained(gpt2_type).to(device)
    model.resize_token_embeddings(len(tokenizer))

    return device, tokenizer, model


def train_model(model, train_dataset):
    # creating training arguments
    training_args = TrainingArguments(output_dir='trained-model',
                                      num_train_epochs=2,
                                      logging_steps=10,
                                      load_best_model_at_end=True,
                                      save_strategy="epoch",
                                      per_device_train_batch_size=2,
                                      per_device_eval_batch_size=2,
                                      warmup_steps=100,
                                      weight_decay=0.01,
                                      logging_dir='logs'
                                      )
    # start training
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                  'attention_mask': torch.stack([f[1] for f in data]),
                                                  'comments': torch.stack([f[0] for f in data])
                                                  # it is a text generation model that uses the prompt itself as the label.
                                                  })
    trainer.train()


def score_model(model, tokenizer, device, X_test, y_test):
    # set the model to eval mode
    _ = model.eval()

    # run model on all test data
    original_comments, predicted_comments, video_titles = [], [], []

    # iter over all test data
    for video_title, comment in zip(X_test, y_test):
        # create request to model (the same as request that was used in training)
        req_text = f"<startoftext>The YouTube video named '{video_title}' may have the following comment:"
        tokenized_req_text = tokenizer(f'{req_text}', return_tensors="pt").input_ids.to(device)

        # perform prediction
        sample_outputs = model.generate(tokenized_req_text,
                                        do_sample=False,
                                        top_k=50,
                                        max_length=512,
                                        top_p=0.9,
                                        temperature=0,
                                        num_return_sequences=0
                                        )

        # decode predicted tokens into text
        predicted_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
        # extract the predicted comment
        try:
            predicted_comment = re.findall("comment: (.*)", predicted_text)[-1]
        except:
            predicted_comment = "None"

        # append results
        original_comments.append(comment)
        predicted_comments.append(predicted_comment)
        video_titles.append(video_title)

    # transform result into dataframe
    df = pd.DataFrame({'video_title': video_titles,
                       'original_comment': original_comments,
                       'predicted_comment': predicted_comments})

    # calc the accuracy
    print(f1_score(original_comments, predicted_comments, average='macro'))
    # TODO: bad way to eval generator model


def save_model(model: torch.nn.Module, path_to_save_file: str = 'saved-model.pklz'):
    pickle.dump(model, open(path_to_save_file, 'wb'))
