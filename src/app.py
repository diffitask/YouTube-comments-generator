import streamlit as st
from PIL import Image
from datasetBuilding import build_dataset, train_test_dataset_split
from model import generate_text, build_model, train_model_stub
from utils import get_youtube_video_info

st.title("YouTube comments generator")

image = Image.open('img/youtube-comments-img.jpeg')
st.image(image, width=200)

# building model
device, tokenizer, model = build_model()

# building dataset
dataframe = build_dataset()
train_dataframe, test_dataframe = train_test_dataset_split(dataframe)

# train model
train_model_stub()


def create_model_request(video_link: str):
    video_title, video_desc = get_youtube_video_info(video_link)

    # title_desc_req = f"The YouTube video named '{video_title}' that is described as '{video_desc}'" \
    #     + " may have the following comment: "

    title_req = f"The YouTube video named '{video_title}' may have the following comment: "
    return title_req


def generate_comment(video_link: str):
    model_req_text = create_model_request(video_link)
    res_str = generate_text(model_req_text, model, tokenizer, device) + '"'
    st.markdown("### Generated comment")
    st.write(res_str)


@st.cache_data
def empty_link_error():
    st.write("Please, enter a non-empty video link")


st.markdown("### Data entry")
with st.form("Comment generating"):
    user_video_link = st.text_input("**Insert YouTube video link here**")
    # comments_cnt = st.text_input("**How many comments to generate?**")
    generate_button_res = st.form_submit_button("Generate comment")

if generate_button_res:
    if not user_video_link:
        empty_link_error()
    else:
        generate_comment(user_video_link)
