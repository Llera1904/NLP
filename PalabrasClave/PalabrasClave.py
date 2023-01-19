import spacy
import re
# import pandas as pd

patron = re.compile("&{8}")
nlp = spacy.load("es_core_news_sm")

# frasesNoticias = []
# frecuenciasPalabrasNoticias = []
# concurrenciasNoticias = []
# puntuacionGradoPalabrasNoticias = []
# puntuacionFinalNoticias = []
# frasesTopNoticias = []
numeroDeNoticia = 0

file = open("tabla.txt", "w", encoding="utf8")
file.write("Noticia" + "\t" + "|" + "Palabras clave" + "\n")
with open("corpus_noticias.txt", encoding="utf8") as archivoEntrada:
	for line in archivoEntrada:
		numeroDeNoticia += 1
		resultado = patron.split(line)
		doc = nlp(resultado[2])

		# Listas donde guardaremos temporalmente las frases y palabras de cada noticia
		listaFrases = []
		listaPalabras = []
		frase = []
		for token in doc: # Sacamos las frases candidatas 
			# Recorremos las noticias palabra por palabra
			if (token.pos_ not in {'ADP', 'PRON', 'CONJ', 'CCONJ', 'DET', 'PUNCT'}):
				listaPalabras.append(token.text)
				frase.append(token.text) 	
			else:
				if frase != []:
					listaFrases.append(frase)
				frase = []
		if frase != []:
			listaFrases.append(frase)
		# frasesNoticias.append(listaFrases)

		frecuenciaPalabrasDiccionario = {} # Sacamos la frecuencia de las palabras 
		matrizConcurrenciaDiccionario = {} # Sacamos la matriz de concurrencia 
		puntuacionGradoPalabrasDiccionario = {} # Sacamos la puntacion del grado
		puntuacionFinalDiccionario = {} # Sacamos la puntacion final (Puntuacion acumulada)
		puntuacionFinalDiccionarioTop = {} # Sacamos las mejores puntaciones finales

		for palabra in listaPalabras: # Sacamos la frecuencia de las palabras
			frecuenciaPalabrasDiccionario[palabra] = listaPalabras.count(palabra)
			matrizConcurrenciaDiccionario[palabra] = 0
		# frecuenciasPalabrasNoticias.append(frecuenciaPalabrasDiccionario)

		# Calcula la matriz de concurrencia 
		# De aqui se obtiene el grado de la palabra
		for frase in listaFrases:
			for palabra in matrizConcurrenciaDiccionario: # Itera sobre las llaves del diccionario
				if palabra in frase:
					matrizConcurrenciaDiccionario[palabra] += len(frase) # Grado de la palabra 
		# concurrenciasNoticias.append(matrizConcurrenciaDiccionario)

		for palabra in matrizConcurrenciaDiccionario: # Itera sobre las llaves del diccionario 
			puntuacion = matrizConcurrenciaDiccionario[palabra] / frecuenciaPalabrasDiccionario[palabra]
			puntuacionGradoPalabrasDiccionario[palabra] = int(puntuacion)
		# puntuacionGradoPalabrasNoticias.append(puntuacionGradoPalabrasDiccionario)

		# Obtenemos la puntuacion acumulada 
		for frase in listaFrases:
			puntuacionFinal = 0
			fraseString = ""
			for palabra in frase:
				fraseString = fraseString + palabra + " "
				puntuacionFinal += puntuacionGradoPalabrasDiccionario[palabra]
			if puntuacionFinal > 1:
				puntuacionFinalDiccionarioTop[fraseString] = puntuacionFinal # Puntuacion acumulada (mejores)
			puntuacionFinalDiccionario[fraseString] = puntuacionFinal # Puntuacion acumulada 
		
		# Ordenamos las puntuaciones de forma descendente
		sortedTop = dict(sorted(puntuacionFinalDiccionarioTop.items(), key=lambda item:item[1], reverse=True))
		sortedPuntuaciones = dict(sorted(puntuacionFinalDiccionario.items(), key=lambda item:item[1], reverse=True))
		# puntuacionFinalNoticias.append(sortedPuntuaciones)
		# frasesTopNoticias.append(sortedTop)

		# Imprime en un txt
		for key in sortedTop.keys():
			file.write(str(numeroDeNoticia) + "\t" + "|" + key + "(" + str(sortedTop[key]) + ")" + "\n")
		file.write("\n")	
file.close()
# print("\nFrases:\n", frasesNoticias[0])
# print("\nFrecuencia palabras:\n", frecuenciasPalabrasNoticias[0])
# print("\nConcurrencias:\n", concurrenciasNoticias[0]) # Grado de la palabra 
# print("\nPuntuacionDelGrado:\n", puntuacionGradoPalabrasNoticias[0])
# print("\nPuntuacion Final:\n", puntuacionFinalNoticias[0])
# print("\nFrases Top:\n", frasesTopNoticias[0])