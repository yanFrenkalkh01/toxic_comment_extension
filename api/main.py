import os
import re # Make sure re is imported
import json
import numpy as np
import tensorflow as tf # Import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences # For pad_sequences

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords_fallback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# --- Download NLTK resources if not already present ---
try:
    nltk.data.find('corpora/wordnet.zip')
except Exception:
    print("Downloading WordNet resource...")
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/stopwords.zip')
except Exception:
    print("Downloading NLTK stopwords resource...")
    nltk.download('stopwords')

# --- Global Variables & Model Loading ---
CATEGORIES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MAX_PAD_LEN = 200
TOXICITY_THRESHOLD = 0.5
MODEL_FILE = "toxic_comment_lstm_final.h5"
TOKENIZER_FILE = "tokenizer.json"

loaded_best_model = None
loaded_tokenizer = None

# Load the best model (LSTM)
if os.path.exists(MODEL_FILE):
    try:
        loaded_best_model = load_model(MODEL_FILE)
        print(f"Successfully loaded '{MODEL_FILE}'")
    except Exception as e:
        print(f"Error loading LSTM model '{MODEL_FILE}': {e}")
else:
    print(f"Model file '{MODEL_FILE}' not found.")

# Load the tokenizer
if os.path.exists(TOKENIZER_FILE):
    try:
        with open(TOKENIZER_FILE) as f:
            data = json.load(f)
            loaded_tokenizer = tokenizer_from_json(data)
        print(f"Successfully loaded '{TOKENIZER_FILE}'")
    except Exception as e:
        print(f"Error loading tokenizer '{TOKENIZER_FILE}': {e}")
else:
    print(f"Tokenizer file '{TOKENIZER_FILE}' not found.")


# --- Preprocessing Function Definitions (Copied from your script) ---
regex_patterns = {
    ' american ': ['amerikan'], ' adolf ': ['adolf'], ' hitler ': ['hitler'],
    ' fuck': ['(f)(u|[^a-z0-9 ])(c|[^a-z0-9 ])(k|[^a-z0-9 ])([^ ])*', '(f)([^a-z]*)(u)([^a-z]*)(c)([^a-z]*)(k)', r'f[!@#$%^&*]*u[!@#$%^&*]*k', 'f u u c', '(f)(c|[^a-z ])(u|[^a-z ])(k)', r'f\*', 'feck ', ' fux ', r'f\*\*', r'f\*\*k',r'fu\*k', r'f\-ing', r'f\.u\.', 'f###', ' fu ', 'f@ck', 'f u c k', 'f uck', 'f ck'],
    ' ass ': [r'[^a-z]ass ', r'[^a-z]azz ', 'arrse', ' arse ', r'@\$\$', r'[^a-z]anus', r' a\*s\*s', r'[^a-z]ass[^a-z ]', r'a[@#$%^&*][@#$%^&*]', r'[^a-z]anal ', 'a s s','a55', r'@\$\$'],
    ' ass hole ': [r' a[s|z]*wipe', r'a[s|z]*[w]*h[o|0]+[l]*e', r'@\$\$hole', r'a\*\*hole'],
    ' bitch ': [r'b[w]*i[t]*ch', 'b!tch', r'bi\+ch', r'b!\+ch', '(b)([^a-z]*)(i)([^a-z]*)(t)([^a-z]*)(c)([^a-z]*)(h)', 'biatch', r'bi\*\*h', 'bytch', 'b i t c h', 'l3itch'],
    ' bastard ': [r'ba[s|z]+t[e|a]+rd'], ' trans gender': ['transgender'], ' gay ': ['gay'],
    ' cock ': [r'[^a-z]cock', 'c0ck', r'[^a-z]cok ', 'c0k', r'[^a-z]cok[^aeiou]', ' cawk', '(c)([^a-z ])(o)([^a-z ]*)(c)([^a-z ]*)(k)', 'c o c k'],
    ' dick ': [r' dick[^aeiou]', 'deek', 'd i c k', 'dik'],
    ' suck ': ['sucker', '(s)([^a-z ]*)(u)([^a-z ]*)(c)([^a-z ]*)(k)', 'sucks', '5uck', 's u c k'],
    ' cunt ': ['cunt', 'c u n t'], ' bull shit ': [r'bullsh\*t', r'bull\$hit'],
    ' homo sex ual': ['homosexual'], ' jerk ': ['jerk'],
    ' idiot ': [r'i[d]+io[t]+', '(i)([^a-z ]*)(d)([^a-z ]*)(i)([^a-z ]*)(o)([^a-z ]*)(t)', 'idiots', 'i d i o t'],
    ' dumb ': ['(d)([^a-z ]*)(u)([^a-z ]*)(m)([^a-z ]*)(b)'],
    ' shit ': ['shitty', '(s)([^a-z ]*)(h)([^a-z ]*)(i)([^a-z ]*)(t)', 'shite', r'\$hit', 's h i t', r'\$h1t'],
    ' shit hole ': ['shythole'], ' retard ': ['returd', 'retad', 'retard', 'wiktard', 'wikitud'],
    ' rape ': [' raped'], ' dumb ass': ['dumbass', 'dubass'], ' ass head': ['butthead'],
    ' sex ': ['sexy', 's3x', 'sexuality'],
    ' nigger ': ['nigger', r'ni[g]+a', ' nigr ', 'negrito', 'niguh', 'n3gr', 'n i g g e r'],
    ' shut the fuck up': ['stfu', r'st\*u'], ' pussy ': [r'pussy[^c]', 'pusy', r'pussi[^l]', 'pusses', r'p\*ssy'],
    ' faggot ': ['faggot', r' fa[g]+[s]*[^a-z ]', 'fagot', 'f a g g o t', 'faggit', '(f)([^a-z ]*)(a)([^a-z ]*)([g]+)([^a-z ]*)(o)([^a-z ]*)(t)', r'fau[g]+ot', r'fae[g]+ot'],
    ' mother fucker': [' motha ', ' motha f', ' mother f', 'motherucker'],
    ' whore ': [r'wh\*\*\*', 'w h o r e'], ' fucking ': [r'f\*$%-ing']
}

def clean_text(text, repeat_text=True, patterns_text=True, is_lower=True):
    text = str(text)
    if is_lower:
        text = text.lower()
    if patterns_text:
        for target, patterns in regex_patterns.items():
            for pat in patterns:
                # Using re.sub for regex patterns, and str.replace for simple strings
                # For simplicity, assuming all 'pat' are intended as regex if they contain special chars
                # A more robust way would be to pre-compile regexes
                try:
                    text = re.sub(pat, target, text)
                except re.error: # Fallback to string replace if it's not a valid regex
                    text = text.replace(pat, target)
    if repeat_text:
        text = re.sub(r'(.)\1{2,}', r'\1', text)
    text = text.replace("\n", " ")
    text = re.sub(r'[^\w\s]', ' ', text) # Remove punctuation
    text = re.sub(r'[0-9]', '', text)    # Remove numbers
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # Remove non-ASCII
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with single
    return text.strip()

wordnet_lemmatizer = WordNetLemmatizer()
def lemma(text, lemmatization=True):
    output = ""
    if lemmatization:
        words = text.split(" ")
        lemmatized_words = []
        for word in words:
            if word: # Ensure word is not empty
                word1 = wordnet_lemmatizer.lemmatize(word, pos="n")
                word2 = wordnet_lemmatizer.lemmatize(word1, pos="v")
                word3 = wordnet_lemmatizer.lemmatize(word2, pos="a") # 'a' for adjective
                word4 = wordnet_lemmatizer.lemmatize(word3, pos="r") # 'r' for adverb
                lemmatized_words.append(word4)
        output = " ".join(lemmatized_words)
    else:
        output = text
    return output.strip()

# Use NLTK stopwords as a fallback if spaCy isn't installed or preferred
try:
    from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
    print("Using spaCy stopwords.")
except ImportError:
    print("spaCy not found, using NLTK stopwords.")
    spacy_stopwords = set(nltk_stopwords_fallback.words('english'))


def remove_stopwords(text, rem_stop_wrds=True):
    output_text = ""
    if rem_stop_wrds:
        words = text.split(" ")
        # Ensure word is not empty before lowercasing
        output_words = [word for word in words if word and word.lower() not in spacy_stopwords]
        output_text = " ".join(output_words)
    else:
        output_text = text
    return output_text.strip()

def predict_toxicity_probabilities(text_input, model, tokenizer_object, max_len=MAX_PAD_LEN):
    if model is None:
        raise ValueError("Model not loaded. Cannot predict.")
    if tokenizer_object is None:
        raise ValueError("Tokenizer not loaded. Cannot predict.")

    cleaned_text = clean_text(text_input)
    lemmatized_text = lemma(cleaned_text)
    processed_text = remove_stopwords(lemmatized_text)

    if not processed_text.strip(): # Handle empty string after preprocessing
        return np.zeros(len(CATEGORIES))


    sequence = tokenizer_object.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(
        sequence, maxlen=max_len, padding='post', truncating='post'
    )
    prediction_probs = model.predict(padded_sequence)[0]
    return prediction_probs

# --- FastAPI App ---
app = FastAPI()

# CORS (Cross-Origin Resource Sharing)
# MODIFICATION FOR DEBUGGING: Allow all origins temporarily
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# END OF MODIFICATION

class TextInput(BaseModel):
    text: str

class TextOutput(BaseModel):
    processed_text: str

@app.post("/process_text", response_model=TextOutput)
async def process_text_endpoint(item: TextInput):
    original_text = item.text
    if not loaded_best_model or not loaded_tokenizer:
        print("Model or Tokenizer not available. Returning original text.")
        return TextOutput(processed_text=original_text)

    try:
        prediction_probs = predict_toxicity_probabilities(
            original_text,
            loaded_best_model,
            loaded_tokenizer,
            max_len=MAX_PAD_LEN
        )
        is_toxic_overall = False
        print(f"\nInput: \"{original_text[:100]}...\"")
        print("Probabilities:")
        for i, category_name in enumerate(CATEGORIES):
            print(f"- {category_name}: {prediction_probs[i]:.4f}")
            if prediction_probs[i] > TOXICITY_THRESHOLD:
                is_toxic_overall = True
        
        if is_toxic_overall:
            return TextOutput(processed_text="[Toxic]")
        else:
            return TextOutput(processed_text=original_text)

    except ValueError as ve:
        print(f"ValueError during prediction: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error processing text")

@app.get("/")
async def root():
    return {"message": "Toxic Comment Detection API is running."}
