import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, AdamW, get_linear_schedule_with_warmup
import numpy as np
from torch.utils.data import Dataset, DataLoader
import streamlit as st


def generate_text(model_req_text: str, model, tokenizer, device) -> str:
    tokens = tokenizer.encode(model_req_text)
    num_steps = 30

    for i in range(num_steps):
        with torch.no_grad():
            logits = model(torch.as_tensor([tokens], device=device))[0]
        p_next = torch.softmax(logits[0, -1, :] * 1.2, dim=-1).data.cpu().numpy()

        next_token_index = np.random.choice(len(p_next), p=p_next)
        tokens.append(int(next_token_index))

    return tokenizer.decode(tokens)


@st.cache_data
def build_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_prefix_space=True)
    model = GPT2LMHeadModel.from_pretrained('gpt2').train(False).to(device)
    return device, tokenizer, model


@st.cache_data
def train_model_stub():
    return


@st.cache_data
def train_model(train_dataframe, model, tokenizer, device,
                batch_size=16, epochs=5, lr=2e-5,
                max_seq_len=400, warmup_steps=200,
                gpt2_type="gpt2", output_dir=".", output_prefix="wreckgar",
                test_mode=False):
    # TODO
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(train_dataframe, batch_size=1, shuffle=True)
    loss = 0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):
        print(f"Training epoch {epoch}")
        print(loss)

    # saving model
    torch.save(model.state_dict(), 'model_dict.pt')

    return model


def score_model(test_dataframe, model):
    # TODO
    pass
