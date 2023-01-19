from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from scipy.spatial import distance
import numpy as np
import pandas as pd
import re
import math 
from pandas import *
import matplotlib.pyplot as plt
import seaborn as sns 
import random

def cosine(x, y):
	val = sum(x[index] * y[index] for index in range(len(x)))
	sr_x = math.sqrt(sum(x_val**2 for x_val in x))
	sr_y = math.sqrt(sum(y_val**2 for y_val in y))
	res = val/(sr_x*sr_y)
	return (res)

def graphHeatMap(documentSimilarity, documentSimilarity100):
    documentSimilarity10 = documentSimilarity100[0:10]
    # print("Top 10: ", documentSimilarity10)

    coordenadasX=[]
    coordenadasY=[]
    for tupla in documentSimilarity10:
        coordenadasX.append(tupla[0])
        coordenadasY.append(tupla[1])
    # print("CoordenadasX: ", coordenadasX)
    # print("CoordenadasY: ", coordenadasY)

    dictionarySimilarity = {}
    for tupla in documentSimilarity:
        dictionarySimilarity[(tupla[0], tupla[1])] = tupla[2]

    dictionarySimilarity10 = {}
    for i in coordenadasX:
        similarity = []
        for j in coordenadasY:
            if i < j :
                similarity.append(dictionarySimilarity[(i, j)])
            elif i > j:
                similarity.append(dictionarySimilarity[(j, i)])  
            elif i == j:
                similarity.append(1) 
        dictionarySimilarity10[i] = similarity.copy()  
    # print(dictionarySimilarity10)

    df = pd.DataFrame(dictionarySimilarity10)
    # print (df)

    # colors=['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #         'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #         'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

    ax = sns.heatmap(data=df, cmap="PuRd", annot=True,  xticklabels=coordenadasY, yticklabels=coordenadasX)
    ax.set(xlabel="", ylabel="")
    plt.show()

def similitudDocs(representacion):
    print("\nSimilitudes")
    documentSimilarity = []
    documentSimilarity100 = []
    k = 0
    for i in range(len(representacion)):
        for j in range(len(representacion)):
            if i != j:
                if j >= k:
                    similitud = 1 - distance.cosine(representacion[i], representacion[j])
                    documentSimilarity.append((i + 1, j + 1, similitud))
        k += 1
    # print(documentSimilarity)
    print("\nComparaciones totales: ", len(documentSimilarity))

    documentSimilarity100 = sorted(documentSimilarity, reverse=True, key=lambda similarity : similarity[2])[:100]
    # print(documentSimilarity100)

    return documentSimilarity, documentSimilarity100

patron = re.compile(r'(?u)\w\w+|\w\w+\n|\.|\?|\,|\;|\:|\¿|\?|\¡|\!')
patronOriginal = re.compile(r'(?u)\w\w+|\w\w+\n|\.')

corpusFiltrado = []
with open("corpusFiltrado.txt", encoding="utf8") as archivoEntrada:
	for line in archivoEntrada:
		corpusFiltrado.append(line)
        # print(line)
# print(corpusFiltrado)

corpus = ['El niño corre velozmente por el camino a gran velocidad .',
          'El coche rojo del niño es grande .',
          'El coche tiene un color rojo brillante y tiene llantas nuevas .',
          '¿ Las nuevas canicas del niño son color rojo ?']

# Representación vectorial binarizada
vectorizadorBinario = CountVectorizer(binary=True, token_pattern=patron)
X = vectorizadorBinario.fit_transform(corpusFiltrado)
print("\nRepresentación vectorial binarizada")

binarizada = X.toarray()
documentSimilarityBinarizada, documentSimilarityBinarizada100 = similitudDocs(binarizada)
# print(documentSimilarity100)

with open("similarityBinarizada100.txt", "w") as similarity:
    for noticia in documentSimilarityBinarizada100:
        similarity.write("Noticia" + str(noticia[0]) + "-" + "Noticia" + str(noticia[1]) + " " + str(noticia[2]) + "\n")

# Representación vectorial por frecuencia
vectorizadorFrecuencia = CountVectorizer(token_pattern=patron)
X = vectorizadorFrecuencia.fit_transform(corpusFiltrado)
print('\nRepresentación vectorial por frecuencia')

frecuencia = X.toarray()
documentSimilarityFrecuencia, documentSimilarityFrecuencia100 = similitudDocs(frecuencia)
# print(documentSimilarity)

with open("similarityFecuencia100.txt", "w") as similarity:
    for noticia in documentSimilarityFrecuencia100:
        similarity.write("Noticia" + str(noticia[0]) + "-" + "Noticia" + str(noticia[1]) + " " + str(noticia[2]) + "\n")

# Representación vectorial tf-idf
vectorizadorTfIdf = TfidfVectorizer(token_pattern=patron)
X = vectorizadorTfIdf.fit_transform(corpusFiltrado)
print ('\nRepresentación vectorial tf-idf')

tfIdf = X.toarray()
documentSimilarityTfIdf, documentSimilarityTfIdf100 = similitudDocs(tfIdf)
# print(documentSimilarity)

with open("similarityTfIdf100.txt", "w") as similarity:
    for noticia in documentSimilarityTfIdf100:
        similarity.write("Noticia" + str(noticia[0]) + "-" + "Noticia" + str(noticia[1]) + " " + str(noticia[2]) + "\n")

# Graficar mapas de calor
graphHeatMap(documentSimilarityBinarizada, documentSimilarityBinarizada100)
graphHeatMap(documentSimilarityFrecuencia, documentSimilarityFrecuencia100)
graphHeatMap(documentSimilarityTfIdf, documentSimilarityTfIdf100)