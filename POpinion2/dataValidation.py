from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# import spacy
import re
import numpy as np
import pandas as pd
import pickle

df = pd.read_excel("Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx") # Lee el excel con las opiniones 
x = df.drop(['Polarity', 'Attraction'], axis=1).values # Corpus sin etiquetas 
y = df.drop(['Title', 'Opinion'], axis=1).values # Etiquetas
print(df)
# print("Datos:\n", x[0], x.shape)
# print("Etiquetas:\n", y[0], y.shape)

# Normaliza titulo y opinion
# nlp = spacy.load("es_core_news_sm")
# patron = re.compile("\n+")
# dataNormalize = []
# for row in x:
#     data = []
#     titleNormalize = ""
#     opinionNormalize = ""

#     title = re.sub(patron, "", str(row[0])) # Quita saltos de linea 
#     title = nlp(title)
#     opinion = re.sub(patron, "", str(row[1]))
#     opinion = nlp(opinion)

#     # Tokeniza y lematiza 
#     for token in title:
#         if token.pos_ not in {'ADP', 'PRON', 'CONJ', 'DET', 'PUNCT', 'CCONJ', 'SYM'}: # Quita stop words y simbolos 
#             titleNormalize += token.lemma_ + " "
#     data.append(titleNormalize)

#     for token in opinion:
#         if token.pos_ not in {'ADP', 'PRON', 'CONJ', 'DET', 'PUNCT', 'CCONJ', 'SYM'}:
#             opinionNormalize += token.lemma_ + " "
#     data.append(opinionNormalize)

#     dataNormalize.append(np.array(data))
# dataNormalize = np.array(dataNormalize)
# print("\n")
# print(dataNormalize[0], dataNormalize.shape)

# datasetFile = open('dataset.pkl', 'wb') # Guardamos el dataset normalizado 
# pickle.dump(dataNormalize, datasetFile)
# datasetFile.close()
             
datasetFile = open('dataset.pkl', 'rb') # Carga dataset guardado 
dataNormalize = pickle.load(datasetFile)
df = pd.DataFrame(data=dataNormalize, columns=["Title", "Opinion"])
print(df)
# print("\nDatos normalizados:\n", dataNormalize[0], dataNormalize.shape)

# Saca conjunto de prueba y entrenamiento 
xTrain, xTest, yTrain, yTest = train_test_split(dataNormalize, y, test_size=0.2, shuffle=True, random_state=0)	# 20% prueba 80% entrenamineto
print("\nxTrain:", len(xTrain))
print("yTrain:", len(yTrain))
print("xTest:", len(xTest))
print("yTest:", len(yTest))

testData = (xTest, yTest)
datasetFile = open('testData.pkl', 'wb') # Guardamos el conjunto de prueba 
pickle.dump(testData, datasetFile)
datasetFile.close()

# Saca k pliegues de los datos 
validationSets = []
kf = KFold(n_splits=5) # Numero de pliegues
for trainIndex, testIndex in kf.split(xTrain):
    xTrainV, xTestV = xTrain[trainIndex], xTrain[testIndex]
    yTrainV, yTestV = yTrain[trainIndex], yTrain[testIndex]
    validationSets.append((xTrainV, yTrainV, xTestV, yTestV)) # Agrega el pliegue creado a la lista
# print("\nxTrainV:", validationSets[0][0].shape)
# print("yTrainV:", validationSets[0][1].shape)
# print("xTestV:", validationSets[0][2].shape)
# print("yTestV:", validationSets[0][3].shape)

# datasetFile = open('kPliegues.pkl', 'wb') # Guardamos el dataset normalizado 
# pickle.dump(validationSets, datasetFile)
# datasetFile.close()