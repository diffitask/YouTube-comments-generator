import streamlit as st
from PIL import Image

st.title("YouTube comments generator")

image = Image.open('img/youtube-comments-img.jpeg')
st.image(image, width=200)


def generate_comment(link):
    st.write("some comment for " + link)


def empty_link_error():
    st.write("Please, enter a non-empty link to the video")


with st.form("Comment generating"):
    user_video_link = st.text_input("**Insert YouTube video link here**")
    comments_cnt = st.text_input("**How many comments to generate?**")
    generate_button_res = st.form_submit_button("Generate comment")

if generate_button_res:
    if not user_video_link:
        empty_link_error()
    else:
        generate_comment(user_video_link)
