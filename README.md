# Projet7-OpenClassrooms
Parcours Data Scientist - Projet 7: Implémentez un modèle de scoring

**Source de données : https://www.kaggle.com/c/home-credit-default-risk/data**

**Dashboard déployé sur streamlit : https://dashboard-loan-prediction.streamlit.app/**

**Contexte:**

Nous sommes Data Scientist au sein d'une société financière **"Prêt à dépenser"**, qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.
L’entreprise souhaite mettre en œuvre un **outil de “scoring crédit”** pour calculer la probabilité qu’un client rembourse son crédit, puis **classifie la demande en crédit accordé ou refusé**. Elle souhaite donc développer un **algorithme de classification** en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.
Prêt à dépenser décide donc de développer un **dashboard interactif** pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement. 

**Notre mission :**

* Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
* Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.
* Mettre en production le modèle de scoring de prédiction à l’aide d’une API, ainsi que le dashboard interactif qui appelle l’API pour les prédictions.

**Description des dossiers et fichiers dans le dossier OC_P7:**
* **Notebook preprocessing** : contient le code de la préparation des données, de l'analyse exploratoire.
* **Notebook model** contient le code de la modélisation.
* **Dossier API** : dossier contenant les fichiers liés au focntionnement de l'API de prédiction.
    * api.py : fichier API de prédiction réalisée avec flask. Contient les différents endpoints.
    * fichier CSV zipé : fichier utilisé pour la prediction.
    * fFichier nécessaire au déploiement sur Heroku de l'API. Ce fichier contient la liste des packages requis pour le projet.
    * Procfile* : Fichier necessaire au déploiement sur Heroku de l'API. Il décrit les processus (dyno) à exécuter lors du lancement de l'application Heroku
* **Dossier Dashboard** : dossier contenant les fichiers liés au fonctionnement du dashboard. 
    * dashboard.py : tableau de bord réalisé avec le framework Streamlit. Il comprend la partie interface utilisateur qui interagit avec l'API de prédiction.
    * setup_streamlit.sh : fichier necessaire au lancement de streamlit. Précise l'option "headless" (mode sans écran) ainsi que le port d'écoute.
    * Procfile* : Fichier necessaire si déploiement sur Heroku de dashboard. Il décrit les processus (dyno) à exécuter lors du lancement de l'application Heroku
    * Requirements.txt : Fichier nécessaire si déploiement sur Heroku du dashboard. Ce fichier contient la liste des packages requis pour le projet.
    * le modèle : modèle pour la prédiction exporté en pickle 
* **data_drift_analysis.hml** : rapport au format .html de data drift entre les données d'entraînement et les données de "production".
