import pandas as pd
import re

def filas(x):
    filas = []
    for i in range(x):
        filas.append(str(i)) # definimos los nombres de las filas
    
    return filas

# (?:subexpresion) Define una subexpresion regular
# ({n,m}) Coencide al menos n veces pero menos de m

patron1 = re.compile("\#\w+")
patron2 = re.compile("\@\w+")
patron3 = re.compile("\d{1,2}\:\d{2}(?:\:\d{2})?") 
patron4 = re.compile("\d{1,2}\-\d{1,2}\-\d{1,4}")
patron5 = re.compile("\d{1,2}\/\d{1,2}\/\d{1,4}")
patron6 = re.compile("\d{1,2}\s(?:de)\s\w+\s(?:del)\s\d{1,4}")
patron7 = re.compile("[Xx:.][XxDdPpVv()]")

hashtag = []
usuario = []
hora = []
fecha = []
emoticon = []

with open("tweets.txt", encoding="utf8") as archivo:
    for linea in archivo:
       hashtag += patron1.findall(linea)
       usuario += patron2.findall(linea)
       hora += patron3.findall(linea)
       fecha += patron4.findall(linea)
       fecha += patron5.findall(linea)
       fecha += patron6.findall(linea)
       emoticon += patron7.findall(linea)

# print(hashtag)
print("Cantidad hashtags: " + str(len(hashtag)) + "\n")

# print(usuario)
print("Cantidad usuarios: " + str(len(usuario)) + "\n")

# print(hora)
print("Cantidad horas: " + str(len(hora)) + "\n")

# print(fecha)
print("Cantidad fechas: " + str(len(fecha)) + "\n")

# print(emoticon)
print("Cantidad emoticones: " + str(len(emoticon)) + "\n")

# Tablas 
datos1 = {
    'Frecuencia' : [len(hashtag)],
    'hashtags' : hashtag
}

datos2 = {
    'Frecuencia' : [len(usuario)],
    'usuarios' : usuario
}

datos3 = {
    'Frecuencia' : [len(hora)],
    'horas' : hora
}

datos4 = {
    'Frecuencia' : [len(fecha)],
    'fechas' : fecha
}

datos5 = {
    'Frecuencia' : [len(emoticon)],
    'emoticones' : emoticon
}

filas1 = filas(len(hashtag))
filas2 = filas(len(usuario))
filas3 = filas(len(hora))
filas4 = filas(len(fecha))
filas5 = filas(len(emoticon))
 
df1 = pd.DataFrame(datos1, index=filas1)
df2 = pd.DataFrame(datos2, index=filas2)
df3 = pd.DataFrame(datos3, index=filas3)
df4 = pd.DataFrame(datos4, index=filas4)
df5 = pd.DataFrame(datos5, index=filas5)

# print(df1)
# print(df2)
# print(df3)
# print(df4)
# print(df5)

# with pd.ExcelWriter('df.xlsx') as writer:
#     df1.to_excel(writer, sheet_name = 'df1')
#     df2.to_excel(writer, sheet_name = 'df2')
#     df3.to_excel(writer, sheet_name = 'df3')
#     df4.to_excel(writer, sheet_name = 'df4')
#     df5.to_excel(writer, sheet_name = 'df5')