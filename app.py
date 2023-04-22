import streamlit as st
from PIL import Image
import yt_dlp
from transformers import pipeline

# load model
comments_generator = pipeline(task='text-generation', model='gpt2')


def get_yt_video_title(yt_video_link: str) -> str:
    ydl = yt_dlp.YoutubeDL({'outtmpl': '%(id)s.%(ext)s'})
    with ydl:
        try:
            info_dict = ydl.extract_info(yt_video_link, download=False)
            video_title = info_dict['title']
            return video_title
        except yt_dlp.utils.DownloadError as err:
            raise err


def invalid_yt_link_error():
    st.write("Entered link is incorrect. Please, enter a valid YouTube video link")


def generate_comment(yt_video_link: str, comments_cnt: int = 1, num_beams: int = 2, length_penalty: int = 5):
    try:
        video_title = get_yt_video_title(yt_video_link)
    except yt_dlp.utils.DownloadError:
        invalid_yt_link_error()
        return

    req_text = f"The YouTube video named '{video_title}' may have the following comment: "
    generator_output = comments_generator(req_text,
                                          do_sample=True,
                                          num_return_sequences=comments_cnt,
                                          num_beams=num_beams,
                                          length_penalty=length_penalty)

    st.markdown("### Generated comments")

    comments_str = ""
    for i in range(comments_cnt):
        comment_str = generator_output[i]['generated_text'].replace(req_text, '').replace('"', '')
        comments_str += str(i + 1) + ". " + comment_str + "\n\n"

    st.write(comments_str)


# ----- application ------
st.title("YouTube comments generator")

image = Image.open('img/youtube-comments-img.jpeg')
st.image(image, width=200)

st.markdown("### Generation parameters")
with st.form("Comment generating"):
    user_video_link = st.text_input("Insert YouTube video link here")

    comments_cnt_col, num_beams_col, len_penalty_col = st.columns(3)

    with comments_cnt_col:
        user_comments_cnt = st.slider("Comments to generate:", 1, 10, 1)

    with num_beams_col:
        user_num_beams = st.slider("Number of beams:", 1, 10, 1)

    with len_penalty_col:
        user_len_penalty = st.slider("Penalty length:", 0, 10, 1)

    generate_button_res = st.form_submit_button("Generate comments")

if generate_button_res:
    generate_comment(user_video_link, user_comments_cnt, user_num_beams, user_len_penalty)
