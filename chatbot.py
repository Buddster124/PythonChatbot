import os
import webbrowser
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
# Speech Recon
import speech_recognition as sr
import pyaudio
# TTS
import pyttsx3
# Time
from datetime import datetime

engine = pyttsx3.init()

nltk.download('wordnet')
nltk.download('omw-1.4')
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def speech_recon():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak Anything :")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except:
            return "Sorry could not recognize what you said"


def lockdown_system():
    print('Activating Lockdown Sequence...')
    os.system("help")


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print("Bot is Running!")

while True:

    message = speech_recon()
    # message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)

    if res == "Initiating Lockdown Sequence...":
        lockdown_system()

    if res == "Time":
        res = datetime.now()
        res = res.strftime("%H:%M:%S")
        res = "The Current Time Is " + str(res)

    print(res)
    engine.say(res)
    engine.runAndWait()
