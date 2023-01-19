import spacy
import re

nlp = spacy.load("es_core_news_sm")
with open('Ejercicio1/archivo_ejercicio_entrada.txt', encoding="utf8") as archivo_entrada:
	dataset = archivo_entrada.read()

# print("\n\n" + dataset + "\n\n")

doc = nlp(dataset)

normalizado = ""
for token in doc:
	# print(token.text, token.pos_, token.dep_, token.lemma_)
	normalizado = normalizado + token.lemma_ + " "

# print(normalizado)

patron1 = re.compile("\s\n")
patron2 = re.compile("\n\s")
normalizado = re.sub(patron1, "", normalizado)
normalizado = re.sub(patron2, "\n", normalizado)

# print(normalizado)

with open("Ejercicio1/textoNormalizado.txt", "w", encoding="utf8") as txt:
	txt.write(normalizado) 
                



