import numpy as np
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt

datasetFile = open('kPliegues.pkl', 'rb') # Carga los k pliegues 
validationSets = pickle.load(datasetFile)

for i in range(len(validationSets)): # Pasamos las tuplas a listas 
    validationSets[i] = list(validationSets[i])
validationSets[4][2] = np.append(validationSets[4][2], [validationSets[4][0][-1]], axis=0) 
validationSets[4][3] = np.append(validationSets[4][3], [validationSets[4][1][-1]], axis=0)
validationSets[4][0] = np.delete(validationSets[4][0], [-1], axis=0)
validationSets[4][1] = np.delete(validationSets[4][1], [-1], axis=0)

print("\nxTrainV:", validationSets[4][0].shape)
print("yTrainV:", validationSets[4][1].shape)
print("xTestV:", validationSets[4][2].shape)
print("yTestV:", validationSets[4][3].shape)

# Crea diccionario de palabras 
patron = re.compile("\s")
opinionPolarityWords = {}
with open('SEL_full.txt', mode= "r" , encoding= "utf-8") as file:
    for line in file:
        line = patron.split(line)
        opinionPolarityWords[line[0]] = (line[5], line[6])
# print("\n", opinionPolarityWords["abundancia"])

# folds = [] # Separamos los k pliegues
# for fold in validationSets:
#     folds.append([fold[0], fold[1]]) # xTrain, yTrain
#     # print(fold[0].shape)
# # print("\n", folds[0].shape)

# positiveWords = ["Alegría", "Sorpresa"]
# negativeWords = ["Tristeza", "Repulsión", "Miedo", "Enojo"]
# differenceFolds = [] # Diferencias por pliegue 
# for fold in folds: 
#     differenceOpinions = []
#     for data in fold[0]: # Titulo y opinion de xTrain
#         positiveWord = 0
#         negativeWord = 0
#         for word in data[1].split(" "): # Opinion
#             # print(word)
#             if word.lower() in opinionPolarityWords.keys():
#                 if opinionPolarityWords[word.lower()][1] in positiveWords:
#                     positiveWord += float(opinionPolarityWords[word.lower()][0])
#                 elif opinionPolarityWords[word.lower()][1] in negativeWords:
#                     negativeWord += float(opinionPolarityWords[word.lower()][0])
#         # print(positiveWord - negativeWord)
#         differenceOpinions.append(positiveWord - negativeWord) # Agrega la diferencia   
#     differenceFolds.append(np.array(differenceOpinions, dtype=np.float32))
# differenceFolds = np.array(differenceFolds)
# print("\n", differenceFolds, differenceFolds.shape)

# datasetFile = open('differenceFolds.pkl', 'wb') # Guardamos el dataset normalizado 
# pickle.dump(differenceFolds, datasetFile)
# datasetFile.close()

# datasetFile = open('trainDataFolds.pkl', 'wb') # Guardamos el dataset normalizado 
# pickle.dump(folds, datasetFile)
# datasetFile.close()

datasetFile = open('differenceFolds.pkl', 'rb')  
differenceFolds = pickle.load(datasetFile)
print("\n", differenceFolds, differenceFolds.shape)

plt.scatter(range(len(differenceFolds[0])), differenceFolds[0], s=0.75) # Graficamos para buscar un espacio de busqueda
plt.show()



    






