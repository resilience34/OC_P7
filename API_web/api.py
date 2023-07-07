#to run :
#en mode local :dans le dossier du fichier api.py faire python app.py puis dans le navigateur aller à http://127.0.0.1:5000/

# imporeter les packages et librairies 
import pandas as pd
import numpy as np
from flask import Flask, render_template, url_for, request,jsonify
import pickle
import math
import base64
from zipfile import ZipFile
from lightgbm import LGBMClassifier
import uvicorn


# Création de l'instance de l'application Flask
app = Flask(__name__)

# Charger le fichier de données à partir de l'archive zip et le lire en tant que dataframe
z = ZipFile("df_test_imputed.zip") # le dataset final avec Standardisation et encodage
dataframe = pd.read_csv(z.open('df_test_imputed.csv'), encoding='utf-8')

# extrait tous les ID client uniques du dataframe et les stocke dans la liste
all_id_client = list(dataframe['SK_ID_CURR'].unique())

# Charger le modèle entraîné à partir du fichier pickle
model = pickle.load(open('best_final_prediction.pickle', 'rb'))
seuil = 0.6


'''@app.get("/predict/{client_id}")
async def predict(client_id : int):
    predictions = model.predict_proba(dataframe).tolist()
    predict_proba = []
    for pred, ID in zip(predictions, all_id_client):
        if ID == client_id:
            predict_proba.append(pred[1])
    return predict_proba[0]'''

# Définir une autre route pour la prédiction
@app.route('/predict', methods=['POST'])#, methods=['POST']
def predict():

    #Pour le rendu des résultats sur l'interface graphique HTML

    # récupère la valeur du champ de formulaire HTML(dashboard) avec le nom 'id_client' 
    # et l'assigne à la variable ID puis le converti en entier
    ID = request.form["id_client"] # request est utilisée pour accéder aux données du formulaire
    ID = int(ID)

    # Vérifier si l'ID client est présent dans la liste des ID client du jeu de données
    if ID not in all_id_client:
        prediction = "Ce client n'est pas répertorié"
    else:
        # Sélectionner les données correspondantes à l'ID client
        X = dataframe[dataframe['SK_ID_CURR'] == ID] # qui contient uniquement les colonnes à correspond à l'ID client spécifié
        X = X.drop(['SK_ID_CURR'], axis=1) # supprime la colonne 'SK_ID_CURR'est utilisée uniquement pour filtrer les données et n'est pas nécessaire pour la prédiction

        # Effectuer la prédiction en utilisant le modèle, en utilisant la probabilité de défaut de paiement
        probability_default_payment = model.predict_proba(X)[:, 1]

        # Appliquer un seuil de 0.6 pour décider si le prêt est accordé ou non
        if probability_default_payment >= seuil:
            prediction = "Prêt NON Accordé"
        else:
            prediction = "Prêt Accordé"

        # Renvoyer la prédiction au format JSON
        return jsonify({'prediction_text': prediction, 'probability_default_payment': probability_default_payment})
if __name__ == "__main__":
    app.run(debug=True)

