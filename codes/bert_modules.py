import numpy as np
import os
import shutil
import tarfile
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm




def text_cleaning(text):
    soup = BeautifulSoup(text, "html.parser")
    text = re.sub(r'\[[^]]*\]', '', soup.get_text())
    pattern = r"[^a-zA-Z0-9\s,']"
    text = re.sub(pattern, '', text)
    return text

def train_bert(df,df_test, text="text", label="sentiment", path = 'models'):
    df = df.copy()   
    df["cleaned"] = df[text].apply(text_cleaning)
    df_test["cleaned"] = df_test[text].apply(text_cleaning)

    x_train, x_val, y_train, y_val = train_test_split(df["cleaned"],
												      df[label],
												      test_size=0.2)
    x_test = df_test["cleaned"]
    y_test = df_test[label]
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    max_len= 128

    X_train_encoded = tokenizer.batch_encode_plus(x_train.tolist(),
                                                padding=True, 
                                                truncation=True,
                                                max_length = max_len,
                                                return_tensors='tf')

    X_val_encoded = tokenizer.batch_encode_plus(x_val.tolist(), 
                                                padding=True, 
                                                truncation=True,
                                                max_length = max_len,
                                                return_tensors='tf')

    X_test_encoded = tokenizer.batch_encode_plus(x_test.tolist(), 
                                                padding=True, 
                                                truncation=True,
                                                max_length = max_len,
                                                return_tensors='tf')
    
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    history = model.fit(
        [X_train_encoded['input_ids'], X_train_encoded['token_type_ids'], X_train_encoded['attention_mask']],
        y_train,
        validation_data=(
        [X_val_encoded['input_ids'], X_val_encoded['token_type_ids'], X_val_encoded['attention_mask']],y_val),
        batch_size=32,
        epochs=3
    )

    test_loss, test_accuracy = model.evaluate(
        [X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']],
        y_test
    )
    print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')
    tokenizer.save_pretrained(path +'/Tokenizer')
    model.save_pretrained(path +'/Model')

    return model, tokenizer

def Get_sentiment(Review, path="model", Model=BertTokenizer.from_pretrained("models" + '/Tokenizer'), Tokenizer= TFBertForSequenceClassification.from_pretrained("models" + '/Model')):

    Review = Review.apply(text_cleaning)

    if not isinstance(Review, list):
        Review = [Review]

    Input_ids, Token_type_ids, Attention_mask = Tokenizer.batch_encode_plus(Review,
																			padding=True,
																			truncation=True,
																			max_length=128,
																			return_tensors='tf').values()

    prediction = Model.predict([Input_ids, Token_type_ids, Attention_mask])

    pred_labels = tf.argmax(prediction.logits, axis=1)
    label = ["Negative", "Positive"]

    pred_labels = [label[i] for i in pred_labels.numpy().tolist()]

    return pred_labels

def make_predictions(reviews, model = TFBertForSequenceClassification.from_pretrained("models" + '/Model'), tokenizer = BertTokenizer.from_pretrained("models" + '/Tokenizer')):

    reviews = reviews.apply(text_cleaning)

    max_len = 128
    inputs = tokenizer.batch_encode_plus(
        reviews.tolist(), padding=True, truncation=True, max_length=max_len, return_tensors='tf'
    )
    predictions = model.predict([inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']])
    predicted_labels = tf.argmax(predictions.logits, axis=1).numpy()
    return predicted_labels

def get_sentiment_pretrained(df, text="text", label="sentiment"):

    df = df.copy()   
    df["cleaned"] = df[text].apply(text_cleaning)

    tokenizer = AutoTokenizer.from_pretrained("MarieAngeA13/Sentiment-Analysis-BERT")
    model = AutoModelForSequenceClassification.from_pretrained("MarieAngeA13/Sentiment-Analysis-BERT")

    def predict_sentiment(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get the sorted probabilities and their corresponding indices
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # If the highest probability is neutral (class 1), pick the next highest
        if sorted_indices[0][0].item() == 1:
            predicted_label = sorted_indices[0][1].item()
        else:
            predicted_label = sorted_indices[0][0].item()
        
        return predicted_label

    tqdm.pandas()
    df['predicted_sentiment'] = df['cleaned'].progress_apply(predict_sentiment)

    if label not in df.columns:
        return df

    result_dict_adj = {"TP":0, "TN":0,"FP":0,"FN":0}

    for i,data in df.iterrows():
        if (data["predicted_sentiment"] in (1,2)) & (data["sentiment"]== 1):
            result_dict_adj["TP"] += 1
        elif (data["predicted_sentiment"] in (1,0)) & (data["sentiment"]== 0):
            result_dict_adj["TN"] += 1
        elif (data["predicted_sentiment"] == 2) & (data["sentiment"]== 0):
            result_dict_adj["FP"] += 1
        elif (data["predicted_sentiment"] == 0) & (data["sentiment"]== 1):
            result_dict_adj["FN"] += 1

    TP_2, TN_2, FP_2, FN_2 = result_dict_adj["TP"], result_dict_adj["TN"], result_dict_adj["FP"], result_dict_adj["FN"]

    accuracy_2 = (TP_2 + TN_2) / ( TP_2 + TN_2 + FP_2 + FN_2 )
    precision_2 = TP_2 / (TP_2 + FP_2)
    recall_2 = TP_2 / (TP_2 + FN_2)

    print("accuracy: ",accuracy_2)
    print("precision: ",precision_2)
    print("recall: ",recall_2)

    df["predicted_sentiment"] = df["predicted_sentiment"].replace(2,1)

    return df["predicted_sentiment"]






