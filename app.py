#  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Candidate number 181485
#  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# IMPORTS

import string
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request
import nltk
from nltk.stem import WordNetLemmatizer
import json
import random
import pickle
import numpy as np
from keras.models import load_model

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

app = Flask(__name__, static_url_path='/templates', static_folder='/templates')
nltk.download('wordnet')
nltk.download('punkt')

warnings.filterwarnings('ignore')
lemmatizeSTR = nltk.stem.WordNetLemmatizer()


# lemmatization for each token in string


def LemmatizeT(tokens):
    return [lemmatizeSTR.lemmatize(token) for token in tokens]


# text normalisation: punctuation is removed


punctuation_remove = dict((ord(punct), None) for punct in string.punctuation)


# all tokens are now lowercased

def LemNormalize(text):
    return LemmatizeT(nltk.word_tokenize(text.lower().translate(punctuation_remove)))


# greeting inputs that can be expected from users as well as appropriate responses

Salutations = ("hello", "hi", "greetings", "sup", "what's up", "hey",)

SaluteResponse = ["hi", "hey", "*nods*", "hi there", "hello", "how can i help?", "wassup bruv"]


# this provision randomises the output


def greeting(strings):
    for word in strings.split():
        if word.lower() in Salutations:
            return random.choice(SaluteResponse)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Return Responses Via Predictive Model
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


model = load_model('model (8).h5')

lemmatizeSTR2 = WordNetLemmatizer()
# load JSON and H5 file

sentenceData = json.loads(open('intents.json').read())
derived_tokens = pickle.load(open('tokens (8).pkl', 'rb'))
derived_classes = pickle.load(open('class (8).pkl', 'rb'))


def normalize_stn(strings):
    partsofsentence = nltk.word_tokenize(strings)
    partsofsentence = [lemmatizeSTR2.lemmatize(word.lower()) for word in partsofsentence]
    return partsofsentence


# return binary result of whether word exists in bag

def bow(strings, tokens, bin=True):
    # tokenize
    partsofsentence = normalize_stn(strings)
    # matrix of words
    matrix = [0] * len(tokens)
    for s in partsofsentence:
        for i, w in enumerate(tokens):
            if w == s:
                # should word exist at coordinate val = 1
                matrix[i] = 1
                if bin:
                    print("found in matrix: %s" % w)
    return np.array(matrix)


def probability_cl(strings, h5):
    # predictions below threshold 0.3 are excluded
    p = bow(strings, derived_tokens, bin=False)
    res = h5.predict(np.array([p]))[0]
    permitted_error = 0.3
    permitted = [[i, r] for i, r in enumerate(res) if r > permitted_error]
    # ordered by probability value
    permitted.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in permitted:
        return_list.append({"intent": derived_classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(probs, json_ints):
    tag = probs[0]['intent']
    text_data_list = json_ints['intents']
    for i in text_data_list:
        if i['tag'] == tag:
            found = random.choice(i['responses'])
            break
    return found


def chat_reply(msg):
    ints = probability_cl(msg, model)
    res = getResponse(ints, sentenceData)
    return res


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# APP.ROUTE sends returns response to webApp
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


@app.route("/get")
def get_bot_response():
    flag = True
    ans = string
    userText = request.args.get('msg')

    user_response = userText.lower()

    if user_response != 'bye':
        if user_response == 'thanks$' or user_response == 'thank you$':
            flag = False
            ans = "You are welcome.."
        else:
            if greeting(user_response) is not None:
                ans = greeting(user_response)
            else:
                ans = chat_reply(user_response)
    else:
        ans = "Bye! take care.."

    return str(ans)


# APP.ROUTE RENDER DISCLAIMER PAGE


@app.route("/")
def disclaimer():
    return render_template("Disclaimer.html")


# APP.ROUTE RENDER HOME PAGE


@app.route("/home/")
def home():
    return render_template("home200.html")


if __name__ == "__main__":
    app.run(port=80, threaded=False)
