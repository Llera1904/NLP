import spacy
import re

nlp = spacy.load("es_core_news_sm")

with open("corpusFiltrado.txt", "w", encoding="utf8") as txt:
	txt.write("")

patron = re.compile("&{8}")
with open("corpus_noticias.txt", encoding="utf8") as archivoEntrada:
	for line in archivoEntrada:
		resultado = patron.split(line)
		# print(resultado)
		# print(resultado[2] + "\n")

		doc = nlp(resultado[2])

		stopwords = {'PREP': "allí, allá, ahí, algún, algúnos, algúnas, demás, acá, el"}
		normalizado = ""
		etiquetas = ""
		for token in doc:
			# print(token.text, token.pos_, token.dep_, token.lemma_)
			if ((token.pos_ not in {'ADP', 'PRON', 'CONJ', 'DET'}) and (token.lemma_ not in stopwords['PREP'])):
					normalizado = normalizado + token.lemma_ + " "
					# normalizado = normalizado + token.text + " "
					# etiquetas = etiquetas + token.pos_ + " " 
        
		normalizado = re.sub("\s\él", "", normalizado)
		normalizado = re.sub("\s\s", "", normalizado)
        
		with open("corpusFiltrado.txt", "a", encoding="utf8") as txt:
			txt.write(normalizado + "\n")
