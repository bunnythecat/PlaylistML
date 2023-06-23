import urllib.request
import numpy as np
import requests
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import streamlit.components.v1 as components
import time
from servicerequest import TokenGenerator
import pickle

st.set_page_config(page_title="Spotify Music Recommendation App", layout="wide")

SERVICE_URL = "http://localhost:3000/lmrec_predict"
# Define the maximum sentence
MAX_SENT_LENGTH = 128

# Load track dictionary
with open(("../data/track_dictionary.p"), "rb") as f:
    track_dict = pickle.load(f)

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (0.1, 2, 0.2, 1, 0.1)
)

row0_1.title("Spotify Music Recommendation App")


with row0_2:
    add_vertical_space()

row0_2.subheader(
    "A Streamlit web app by [Ailin Wang](https://github.com/bunnythecat)"
)

row1_spacer1, row1_1, row1_spacer2 = st.columns((0.1, 3.2, 0.1))

with row1_1:
    st.markdown(
        "Welcome to the Jungle! This app utilizes auto-regressive Language Model to generate music recommandation. Give it a go!"
    )
    st.markdown(
        "**To begin, please enter track ids and seperate them with comma) (or just use the sample!).** 👇"
    )

row2_spacer1, row2_1, row2_spacer2 = st.columns((0.1, 3.2, 0.1))
with row2_1:
    user_input = st.text_input(
        "Input your own track ids (e.g. 57bgtoPSgt236HzfBOd8kj,0G21yYKMZoHa30cYVi1iA8,6J17MkMmuzBiIOjRH6MOBZ,7snQQk1zcKl8gZ92AnueZW,07q0QVgO56EorrSGHC48y3)"
    )
    need_help = st.expander("Here's how to find any Track IDs in Spotify 👉")
    with need_help:
            st.write(""" 
                - Search for a song on the Spotify app
                - Right Click on the song you like
                - Click "Share"
                - Choose "Copy Song Link"
                - Track ID comes after /track/
                - (E.g., 57bgtoPSgt236HzfBOd8kj for https://open.spotify.com/track/57bgtoPSgt236HzfBOd8kj)
            """)
            st.markdown("<br>", unsafe_allow_html=True)
            # st.image('./assets/images/.png')

    if not user_input:
        user_input = "57bgtoPSgt236HzfBOd8kj,0G21yYKMZoHa30cYVi1iA8,6J17MkMmuzBiIOjRH6MOBZ,7snQQk1zcKl8gZ92AnueZW,07q0QVgO56EorrSGHC48y3"

    st.warning(
        """Model prediction may take a while."""
    )

tokenGenerator = TokenGenerator(MAX_SENT_LENGTH, track_dict, SERVICE_URL)

row3_spacer1, row3_1, row3_spacer2 = st.columns((0.1, 5, 0.1))
with row3_1:
    get_song = st.button("Get Songs")

if get_song:
    st.subheader("User's Songs")
    st.markdown('---')
    track_uris = user_input.split(",")
    for track_uri in track_uris:
        uri_link = 'https://open.spotify.com/embed/track/' + track_uri + "?utm_source=generator&theme=0"
        components.iframe(uri_link, height=80)
    st.markdown("<br>", unsafe_allow_html=True)

row4_spacer1, row4_1, row4_spacer2 = st.columns((0.1, 5, 0.1))
with row4_1:
    get_rec = st.button("Get Recommandations")
if get_rec:
    start_tokens = user_input.split(",")
    with st.spinner('Getting Recommendations...'):
        text = tokenGenerator.generate(start_tokens, 5)

    if text is not None:
        st.success('Here are top 5 recommendations!')
        st.write("Recommendation tokens: {}".format(text))
        st.markdown('---')
        for track_uri in text.split(","):
            uri_link = 'https://open.spotify.com/embed/track/' + track_uri + "?utm_source=generator&theme=0"
            components.iframe(uri_link, height=80)
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.error("Error! Either the songs entered are not in the database or the ids are invalid. Please try again!")
                