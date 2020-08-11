# Fake_Anime_Generator
This is a repo that contains the code used to build a simple Machine Learning Web App that generates artificial anime titles and descriptions using neural networks. Visit http://zhijingeu.pythonanywhere.com to see the finished product

There is an accompanying Medium article here: 
https://medium.com/@zhijingeu/building-an-amateur-machine-learning-web-app-what-i-learnt-d6a89bddb025 

and series of youtube videos here:
https://youtu.be/4B8sGQgTUmg
https://youtu.be/R_lVjGwUhIA
https://youtu.be/_VNsYQfGz_I

Description Of Files & Folders:

Building_and_Deploying_A_Simple_ML_WebApp.pdf - is a walk thru of the development and deployment process 

FakeAnimeGenerator_Full.ipynb -contains the full set of code including the extracting of the training image dataset , training of the neural networks and launching of a local Flask webapp

FakeAnimeGenerator_RunOnSpellML.py -contains only the code required to train the Generative Adversarial Network on SPELL ML a cloud based ML Ops service

FakeAnimeGenerator_FlaskApp.py -contains only the code required to reload the trained models and launch a local Flask WebApp

ParseHubExtract.csv - contains the training data for the text generation models as extracted from MyAnimeList.com

ImageDataset - is a folder that contains the raw images used for GAN Image model

Tensorboard - is the folder containing the saved logs for the Text Model output from FakeAnimeGenerator_Full.ipynb

ImageModel_checkpoints - is a folder that contains the tensorflow H5 files for 1000, 5000 and 10,000 Epoch runs for the GAN models

TextModel_chkpoints- is a folder that contains the saved ckpt files for the Title and Synopses generation models 

Static + Templates - are the folders with the HTML template and images that are referenced when the Flask app is run either from FakeAnimeGenerator_Full.ipynb or FakeAnimeGenerator_FlaskApp.py
