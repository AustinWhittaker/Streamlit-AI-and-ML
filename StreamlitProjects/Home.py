import streamlit
import streamlit as st

st.set_page_config(
    page_title= "Streamlit Projects"
)

st.sidebar.success('Select a page above')

st.markdown("<h1 style='text-align: center;'> Home </h1>", unsafe_allow_html=True)
st.image('Images/streamlit.jpg')
streamlit.markdown('---')

st.text('')

