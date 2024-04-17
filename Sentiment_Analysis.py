import numpy as np
import pickle
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
import streamlit as st

def sentiment_analysis(review):    

    model_checkpoint = 'distilbert-base-uncased'

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

    # define label maps
    id2label = {0: "Negative", 1: "Positive"}
    label2id = {"Negative":0, "Positive":1}

    # This function will create an instance of distilbert-base-uncased model and from_pretrained will also load the weights of this model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)

    # add pad token if none exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    loaded_model=pickle.load(open('/Users/saunvidganbavale/Downloads/trained_model.sav','rb'))

    # defined examples
    #text_list = [ "Not a fan, don't recommed.", "An artistic marvel, elevating experiences to profound realms of sublime beauty.", "This is not worth watching even once.", "I was shocked by how amazing it was"]
    #for text in text_list:
    inputs = tokenizer.encode(review, return_tensors="pt")
    # compute logits
    logits = loaded_model(inputs).logits
    # convert logits to label
    predictions = torch.argmax(logits)
    #print(review + " - " + id2label[predictions.tolist()])
    return id2label[predictions.tolist()]

def main():

    # giving title
    st.title('Sentiment Analysis for movie reviews')

    # getting input data from user
    Movie_review=st.text_input('Enter the movie review')

    #code for prediction
    predictions=''
    #creating button for prediction

    if st.button('Predict sentiment'):
        predictions=sentiment_analysis(Movie_review)

    st.success(predictions)

if __name__=='__main__':
    main()



