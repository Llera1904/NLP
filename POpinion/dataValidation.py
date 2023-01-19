import spacy
import re
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_excel("Rest_Mex_2022_Sentiment_Analysis_Track_Train.xlsx") # Lee el excel con las opiniones 
x = df.drop(['Polarity', 'Attraction'], axis=1).values # Corpus sin etiquetas 
y = df.drop(['Title', 'Opinion'], axis=1).values # Etiquetas

# count de categorias
yPolarity = df.drop(['Title', 'Opinion', 'Attraction'], axis=1)
yAtracttion = df.drop(['Title', 'Opinion', 'Polarity'], axis=1)
count = yPolarity.value_counts()
count.plot.bar()
plt.ylabel('Number of records')
plt.xlabel('Target Class')
plt.show()
print(df)

# Concatenamos titulo y opinion 
data = []
for i in range(len(x)): 
    data.append(str(x[i][0]) + " " + str(x[i][1]))
x = np.array(data)
df = pd.DataFrame(data=x, columns=["Title and Opinion"])
print("\n", df)

# Normaliza titulo y opinion
nlp = spacy.load("es_core_news_sm")
patron = re.compile("\n+")
dataNormalize = []
for row in x:
    textNormalize = ""
    text = re.sub(patron, "", row) # Quita saltos de linea 
    text = nlp(text)

    # Tokeniza y lematiza 
    for token in text:
        # if token.pos_ not in {'SYM'}: 
            textNormalize += token.lemma_ + " " 
            # textNormalize += token.text + " "
    dataNormalize.append(textNormalize)
dataNormalize = np.array(dataNormalize)
np.save("dataNormalize.npy", dataNormalize) # Guarda el dataset normalizado
# print("\n", dataNormalize[0], dataNormalize.shape)

dataNormalize = np.load("dataNormalize.npy")
df = pd.DataFrame(data=dataNormalize, columns=["Title and Opinion"])
print(df)
# print("\nDatos normalizados:\n", dataNormalize[0], dataNormalize.shape)

# Guardamos las etiquetas 
np.save("samples.npy", y)
datasetFile = open("samples.pkl", "wb") 
pickle.dump(y, datasetFile)
datasetFile.close()