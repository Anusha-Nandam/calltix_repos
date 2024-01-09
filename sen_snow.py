from config import *
import snowflake.connector
import pandas as pd
import json
from transformers import pipeline
import matplotlib.pyplot as plt
import streamlit as st


conn = {
    "user"  : snowflake_user,
    "password": snowflake_password,
    "account": snowflake_account,
    "warehouse": snowflake_warehouse,
    "database": snowflake_database,
    "schema": snowflake_schema
}

connection = snowflake.connector.connect(**conn)

cur = connection.cursor()
cur.execute("SELECT * FROM DIARIZED_DATA")
columns = [column[0] for column in cur.description]
results = cur.fetchall()

# st.write(results)

df = pd.DataFrame(results, columns=columns)


st.write(df)

sentiment_analysis = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

for j in range(1,5):
    st.write("**********************************************************")
    st.write(df['FILE_NAME'][j])
    st.write("**********************************************************")

    dictionary_of_dicts = json.loads(df['AUDIO_DIARIZATION_TEXT'][j])
    for i in dictionary_of_dicts['diarization']['full_transcribe_data']:
        predictions = sentiment_analysis(i['transcription'])

    for i in dictionary_of_dicts['diarization']['full_transcribe_data']:
        i['predictions']= sentiment_analysis(i['transcription'])


    speaker_sentiments = {}
    for segment in dictionary_of_dicts["diarization"]["full_transcribe_data"]:
        speaker_id = segment["speaker"]
        sentiment_score = segment["predictions"][0]["score"]  # Taking the first prediction score
        if segment["predictions"][0]["label"]=='negative':
            sentiment_score = -sentiment_score
        elif segment["predictions"][0]["label"]=='neutral':
            sentiment_score = 0
        if speaker_id not in speaker_sentiments:
            speaker_sentiments[speaker_id] = []
        speaker_sentiments[speaker_id].append(sentiment_score)

    plt.figure(figsize=(10, 6))
    for speaker, sentiments in speaker_sentiments.items():
        plt.plot(sentiments, label=f'Speaker {speaker}', marker='o', linestyle='-', markersize=8)

        # Annotate each point with sentiment score
        for i, score in enumerate(sentiments):
            plt.annotate(f'{score:.2f}', (i, score), textcoords="offset points", xytext=(0,10), ha='center')

    plt.xlabel('Conversation Segment')
    plt.ylabel('Sentiment Score')
    plt.title('Sentiment Analysis of Each Speaker')
    plt.legend()
    plt.tight_layout()

    # plt.gcf().patch.set_facecolor('#00FF00') 
    # plt.set_facecolor('#f0f0f0')
    # plt.show()
    st.pyplot(plt)

    st.text(df['AUDIO_DIARIZATION_TEXT'][j])
    st.write("**********************************************************")
    
    st.write("**********************************************************")
    st.write("**********************************************************")


