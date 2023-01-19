import numpy as np
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

datasetFile = open('testData.pkl', 'rb') # Carga el conjunto de prueba 
dataTest = pickle.load(datasetFile)

print("xTest:", dataTest[0].shape)
print("yTest:", dataTest[1].shape)

# Crea diccionario de palabras 
patron = re.compile("\s")
opinionPolarityWords = {}
with open('SEL_full.txt', mode= "r" , encoding= "utf-8") as file:
    for line in file:
        line = patron.split(line)
        opinionPolarityWords[line[0]] = (line[5], line[6])
# print("\n", opinionPolarityWords["abundancia"])

positiveWords = ["Alegría", "Sorpresa"]
negativeWords = ["Tristeza", "Repulsión", "Miedo", "Enojo"]
differences = [] # Diferencias 
for xTest in dataTest[0]: 
    positiveWord = 0
    negativeWord = 0
    for word in xTest[1].split(" "): # Opinion
        # print(word)
        if word.lower() in opinionPolarityWords.keys():
            if opinionPolarityWords[word.lower()][1] in positiveWords:
                positiveWord += float(opinionPolarityWords[word.lower()][0])
            elif opinionPolarityWords[word.lower()][1] in negativeWords:
                negativeWord += float(opinionPolarityWords[word.lower()][0])
    # print(positiveWord - negativeWord)
    differences.append(positiveWord - negativeWord) # Agrega la diferencia   
differences = np.array(differences)
# print("\n", differences, differences.shape)

yPred = []
yReal = []
# threshold = [-1.557, -0.92, -0.3833, 0] # Categoriza mas proporcionalmente
threshold = [-1.608, -1.548, -1.101, -0.7314]
for difference, yTest in zip(differences, dataTest[1]): # Diferencias y yTest
    yReal.append(yTest[0])
    if difference <= threshold[0]:    
            yPred.append(1)
    elif difference > threshold[0] and difference <= threshold[1]:    
            yPred.append(2)
    elif difference > threshold[1] and difference <= threshold[2]:    
            yPred.append(3)
    elif difference > threshold[2] and difference <= threshold[3]:    
            yPred.append(4)
    elif difference > threshold[3]:    
            yPred.append(5)
# normalizedAccuracy = accuracy_score(yReal, yPred)
# print("Accuracy:", normalizedAccuracy)

# Metricas
yPredCluster = np.zeros((5), dtype=np.int32)
for cluster in yPred:
    if cluster == 1:
        yPredCluster[0] += 1
    elif cluster == 2:
        yPredCluster[1] += 1
    elif cluster == 3:
        yPredCluster[2] += 1
    elif cluster == 4:
        yPredCluster[3] += 1
    elif cluster == 5:
        yPredCluster[4] += 1

yRealCluster = np.zeros((5), dtype=np.int32)
for cluster in yReal:
    if cluster == 1:
        yRealCluster[0] += 1
    elif cluster == 2:
        yRealCluster[1] += 1
    elif cluster == 3:
        yRealCluster[2] += 1
    elif cluster == 4:
        yRealCluster[3] += 1
    elif cluster == 5:
        yRealCluster[4] += 1

print("Etiquetas reales", yRealCluster, np.sum(yRealCluster))
print("Etiquetas predichas",yPredCluster, np.sum(yPredCluster))

normalizedAccuracy = accuracy_score(yReal, yPred)
accuracy = accuracy_score(yReal, yPred, normalize=False)
print("Accuracy normalizado:", normalizedAccuracy)
print("Accuracy:", accuracy)

cm = confusion_matrix(yReal, yPred, labels=[1,2,3,4,5])
print("\n", classification_report(yReal, yPred, target_names=["1","2","3","4","5"]))
print (cm)

disp1 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,2,3,4,5])
disp1.plot()
plt.show()
