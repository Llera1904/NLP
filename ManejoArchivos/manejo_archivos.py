import spacy

nlp = spacy.load("es_core_news_sm")

archivo_entrada = open('archivo_entrada.txt', 'r')

for line in archivo_entrada:
	print (line, end='')

archivo_entrada.close()

#Aquí no es necesario el llamado a close ya que lo hace automáticamente
with open('archivo_entrada.txt') as archivo_entrada:
	dataset = archivo_entrada.read()

print (dataset)

doc = nlp(dataset)

normalizado = ""
for token in doc:
    # print(token.text, token.pos_, token.dep_, token.lemma_)
    normalizado = normalizado + token.lemma_ + " "
    
print (normalizado)

archivo_salida = open('archivo_salida.txt', 'w')
archivo_salida.write(normalizado)
archivo_salida.close()








