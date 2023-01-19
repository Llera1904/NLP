from gensim.models.doc2vec import Doc2Vec, TaggedDocument # https://radimrehurek.com/gensim/models/doc2vec.html
import spacy
import re

def trainModel(data):
	taggedData = [TaggedDocument(d, [i]) for i, d in enumerate(data)]
	# print(taggedData)

	# Train doc2vec model
	model = Doc2Vec(taggedData, dm=1, vector_size=300, window=10)
	# print(model.dv[0])

	return model

def loadModel(model):
	# Load saved doc2vec model
	model = Doc2Vec.load(model)
	similitudesTop = []
	combinacionesRepetidas = []
	for i in range(635):
		similitud = model.dv.most_similar(model.dv[i])[1]
		combinacionesRepetidas.append((i, similitud[0]))
		if (similitud[0], i) not in combinacionesRepetidas:
			similitudesTop.append((i, similitud[0], similitud[1]))

	documentSimilarity10 = sorted(similitudesTop, reverse=True, key=lambda similarity : similarity[2])[:10]
	print("\nTop10:\n", documentSimilarity10)

	return documentSimilarity10
    
sentences = [['i', 'like', 'apple', 'pie', 'for', 'dessert'],
           ['i', 'dont', 'drive', 'fast', 'cars'],
           ['data', 'science', 'is', 'fun'],
           ['chocolate', 'is', 'my', 'favorite'],
           ['my', 'favorite', 'movie', 'is', 'predator'],
           ['vanilla', 'is', 'my', 'favorite'],
           ['chocolate', 'is', 'delicious'],
           ['vanilla', 'is', 'delicious']]

nlp = spacy.load("es_core_news_sm")

patron = re.compile("&{8}")
noticiasTokenizadas = []
noticiasNormalizadas = []
stopwords = ['allí', 'allá', 'ahí', 'algún', 'algúnos', 'algúnas', 'demás', 'acá']
with open("corpus_noticias.txt", encoding="utf8") as archivoEntrada:
	for line in archivoEntrada:
		resultado = patron.split(line)
		# print(resultado)
		# print(resultado[2] + "\n")

		doc = nlp(resultado[2])

		tokenizado = []
		normalizado = []
		for token in doc:
			tokenizado.append(token.text)
			if (token.pos_ not in {'ADP', 'PRON', 'CONJ', 'DET'}) and (token.text not in stopwords):
				normalizado.append(token.lemma_)
		# print(tokenizado)
		noticiasTokenizadas.append(tokenizado)
		noticiasNormalizadas.append(normalizado)
# print("Tokenizado:", noticiasTokenizadas)
# print("NoticiasNormalizadas:", noticiasNormalizadas)

model = trainModel(noticiasTokenizadas)
# Save trained doc2vec model
model.save("doc2vecTokenizado.model")
      
model = trainModel(noticiasNormalizadas)
# Save trained doc2vec model
model.save("doc2vecNormalizado.model")

# Load saved doc2vec model
documentSimilarity10 = loadModel("doc2vecTokenizado.model")
with open("similarityTop10Tokenizado.txt", "w") as similarity:
    for noticia in documentSimilarity10:
        similarity.write("Noticia" + str(noticia[0]) + "-" + "Noticia" + str(noticia[1]) + " " + str(noticia[2]) + "\n")

# Load saved doc2vec model
documentSimilarity10 = loadModel("doc2vecNormalizado.model")
with open("similarityTop10Normalizado.txt", "w") as similarity:
    for noticia in documentSimilarity10:
        similarity.write("Noticia" + str(noticia[0]) + "-" + "Noticia" + str(noticia[1]) + " " + str(noticia[2]) + "\n")








