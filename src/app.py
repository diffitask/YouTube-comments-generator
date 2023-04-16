import streamlit as st

st.title("YouTube comments generator")
st.markdown(
    "<img width=200px src='https://drive.google.com/file/d/1Bmkawh_bCedwkMapw6IALhurF0sPTu3q/view?usp=sharing'>",
    unsafe_allow_html=True)


def process(text):
    return text + "555"


text = st.text_area("TEXT HERE")

st.markdown(f"{process(text)}")
