from asyncore import write
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import re
import csv

# 147414 Palabras totales de todos los ducumentos
# 16488 Palabras distintas encontradas
corpusFiltrado=[]
with open("corpusFiltrado.txt", encoding="utf8") as archivoEntrada:
	for line in archivoEntrada:
		corpusFiltrado.append(line)
        # print(line)
# print(corpusFiltrado)

# corpus = ['El niño corre velozmente por el camino a gran velocidad .',
#           'El coche rojo del niño es grande .',
#           'El coche tiene un color rojo brillante y tiene llantas nuevas .',
#           '¿ Las nuevas canicas del niño son color rojo ?'
# ]
# print(corpus)

# Representación vectorial binarizada
patron = re.compile(r'(?u)\w\w+|\w\w+\n|\.|\,|\:|\;|\¿|\?|\¡|\!')
patron2 = re.compile(r'(?u)\w\w+|\w\w+\n|\.') 

vectorizadorBinario = CountVectorizer(binary=True, token_pattern=patron)
X = vectorizadorBinario.fit_transform(corpusFiltrado)
# print (vectorizadorBinario.get_feature_names_out())
# print (X) # Sparse matrix
# print (type(X)) # Sparse matrix
# print (type(X.toarray())) 
# print("\n")

# Le damos formato y lo pasamos a un csv
A = csr_matrix(X.toarray())
df = pd.DataFrame.sparse.from_spmatrix(A, columns=vectorizadorBinario.get_feature_names_out())
print(df)
# df.to_csv("binarizado.csv", index=False, sep=',', encoding='utf-8') 

# Representación vectorial por frecuencia
vectorizadorFrecuencia = CountVectorizer(binary=False, token_pattern=patron)
X = vectorizadorFrecuencia.fit_transform(corpusFiltrado)
# print (vectorizador_binario.get_feature_names_out())
# print (X) # Sparse matrix
# print (type(X)) # Sparse matrix
# print (type(X.toarray())) 
print("\n")

A = csr_matrix(X.toarray())
df = pd.DataFrame.sparse.from_spmatrix(A, columns=vectorizadorFrecuencia.get_feature_names_out())
print(df)
# df.to_csv("frecuencia.csv", index=False, sep=',', encoding='utf-8') 

# Representación vectorial tf-idf
vectorizadorTfIdf = TfidfVectorizer(token_pattern=patron)
X = vectorizadorTfIdf.fit_transform(corpusFiltrado)
# print (vectorizador_binario.get_feature_names_out())
# print (X) # Sparse matrix
# print (type(X)) # Sparse matrix
# print (type(X.toarray())) 
print("\n")

A = csr_matrix(X.toarray())
df = pd.DataFrame.sparse.from_spmatrix(A, columns=vectorizadorTfIdf.get_feature_names_out())
print(df)
df.to_csv("tf-idf.csv", index=False, sep=',', encoding='utf-8') 



