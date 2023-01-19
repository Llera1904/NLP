from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def logisticRegression(xTrain, xTest, yTrain, yTest):
    # solucionador {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}
    # multi_clase {'auto', 'ovr', 'multinomial'}
    # clf = LogisticRegression(C=5, max_iter=2000)
    clf = LogisticRegression(solver="liblinear", C=0.5, random_state=0, max_iter=1000, multi_class="ovr")
    clf.fit(xTrain, yTrain)
    yPredict = clf.predict(xTest)

    return metrics(yTest, yPredict, clf)

def multinomialNaiveBayes(xTrain, xTest, yTrain, yTest):
    clf = MultinomialNB()
    clf.fit(xTrain, yTrain)
    yPredict = clf.predict(xTest)

    return metrics(yTest, yPredict, clf)

def metrics(yTest, yPredict, clf):
    cm = confusion_matrix(yTest, yPredict)
    print("\nMatriz de confusión:\n", cm)
    print(classification_report(yTest, yPredict))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    
    normalizedAccuracy = accuracy_score(yTest, yPredict)
    f1Score = f1_score(yTest, yPredict, average="micro") # weighted
    recall = recall_score(yTest, yPredict, average="micro")
    precision = precision_score(yTest, yPredict,average="micro")
    print ("\nnormalizedAccuracy:", normalizedAccuracy * 100, "%")
    print ("f1Score:", f1Score * 100, "%")
    print ("recall:", recall * 100, "%")
    print ("precision:", precision * 100, "%\n")

    return normalizedAccuracy, f1Score, recall, precision, clf

dataNormalize = np.load("dataNormalize.npy") # Carga dataset normalizado 
df = pd.DataFrame(data=dataNormalize, columns=["Title and Opinion"])
print(df)

datasetFile = open('samples.pkl', 'rb') # Carga el conjunto de prueba 
y = pickle.load(datasetFile)

target = 1 # [polaridad, atraccion]
y = y[:, target].tolist()# Obtenemos una columna de las etiquetas [estrellas, atraccion]
# y = np.array(y, np.int8)
y = np.array(y)

# # Balancear data
# dataNormalizeResample = dataNormalize.reshape(-1, 1)
# yResample = y.reshape(-1, 1)
# dataBalancing = RandomOverSampler(random_state=0)
# dataNormalizeResample, yResample = dataBalancing.fit_resample(dataNormalizeResample, yResample)
# dataNormalize = dataNormalizeResample.reshape(-1)
# y = yResample.reshape(-1)
# print("\n", dataNormalize.shape, y.shape)

# Vectoriza el dataset 
vectorizer = CountVectorizer(binary=True, dtype=np.int8) # True para binarizado, False para frecuencia
x = vectorizer.fit_transform(dataNormalize)
dataVectorizer = x.toarray() # Le damos formato 
df = pd.DataFrame(dataVectorizer, columns=vectorizer.get_feature_names_out())
print("\n", df)
   
# Saca conjunto de prueba y entrenamiento 
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0) # 20% prueba 80% entrenamiento
print("\nxTrain:", xTrain.shape)
print("yTrain:", yTrain.shape)
print("xTest:", xTest.shape)
print("yTest:", yTest.shape, "\n")

# Transforma las características escalándolas a un rango dado, por defecto (0,1). Este tipo de escalado suele denominarse frecuentemente "normalización" de los datos
# scaler = MinMaxScaler()
# xTrain = scaler.fit_transform(xTrain.todense())
# xTest = scaler.transform(xTest.todense())

# Saca k pliegues de los datos vectorizados 
accuracies = []
f1Scores = []
recalls = []
precisions = []
models = []
p = 0
kf = KFold(n_splits=5) # Numero de pliegues
for trainIndex, testIndex in kf.split(xTrain):
    p += 1
    print("Pliegue:", p)
    xTrainV, xTestV = xTrain[trainIndex], xTrain[testIndex]
    yTrainV, yTestV = yTrain[trainIndex], yTrain[testIndex]

    accuracy, f1Score, recall, precision, clf = logisticRegression(xTrainV, xTestV, yTrainV, yTestV)
    # accuracy, f1Score, recall, precision, clf = multinomialNaiveBayes(xTrainV, xTestV, yTrainV, yTestV)
    accuracies.append(accuracy)
    f1Scores.append(f1Score)
    precisions.append(precision)
    recalls.append(recall)
    models.append(clf)

accuracies = np.array(accuracies)
f1Scores = np.array(f1Scores)
recalls = np.array(recalls)
precisions = np.array(precisions)
print("\nAccuracy promedio:", np.mean(accuracies))
print("F1Score promedio:", np.mean(f1Scores))
print("Recall promedio:", np.mean(recalls))
print("Precision promedio:", np.mean(precisions))
print("Mejor modelo:", f1Scores.tolist().index(max(f1Scores)))

# np.save("model.npy", models[f1Scores.tolist().index(max(f1Scores))]) # Guarda el modelo

# Aplicar mejor modelo 
clf = models[f1Scores.tolist().index(max(f1Scores))]
yPredict = clf.predict(xTest)
metrics(yTest, yPredict, clf)