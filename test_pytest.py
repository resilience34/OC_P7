# import des librairies
import pytest
import asyncio
import requests
import numpy as np
import pandas as pd 
import pickle

# import modlèle et fichier pour le test 
data_clean = pd.read_csv(('df_test_imputed.csv'), index_col='SK_ID_CURR', encoding ='utf-8')
all_id_client = data_clean.index
lgbm = pickle.load(open('best_final_prediction.pickle', 'rb'))

# fonction de prediction
def classify_client(model, ID, df): # fonction via streamlit
    ID = int(ID)
    X = df[df.index == ID]
    #X = X.drop(['TARGET'], axis=1) #if df_train_imputes.csv
    predict = model.predict_proba(X)[:, 1]
    return predict

def predict_loan_approval(id_client, url):
    id_client = int(id_client)
    id = "/{id_client}"
    url2 = url+id
    api_url = url2.replace("{id_client}", str(id_client))
    response = requests.get(api_url)
    result = response.text
    df = pd.read_json(result)
    return df

# Prediction avec les données local
id_client = data_clean.index.tolist()[0]
id_client_2 = 100245
predict_locl = classify_client(lgbm, id_client_2, data_clean)

# Prediction avec les données via l'API
url_predict_client = " https://apiopc-fbf4eb881e94.herokuapp.com/predict"

data_api = predict_loan_approval(id_client, url_predict_client)
data_api = data_api.set_index('SK_ID_CURR')
predict_id_api = classify_client(lgbm, id_client, data_api)

# initialisation des tests 
def return_predict():
    return predict_id_api

def test_return_predict() : 
    assert return_predict() == predict_locl
