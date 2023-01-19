#Search
import re
res = re.search("c", "abcdef")
print('Search: ', res)

#Findall
res = re.findall("\s", "esta es una cadena.")
print ('Findall: ', res)

#split
res = re.split("\s", "esta es una cadena.")
print ('Split: ', res)

#sub
res = re.sub("\s", "\n", "esta es una cadena.")
print ('sub: ', res)

#compile
patron = re.compile(",")
resultado = patron.findall("Cadena1, Cadena2, Cadena3, Cadena4, Cadena5")
print(resultado)
resultado2 = patron.split("Cadena1, Cadena2, Cadena3, Cadena4, Cadena5")
print(resultado2)


#Uso de operadores
patron = re.compile("\d+\.?\d+")
resultado = patron.findall("Esta es una cadena con los números 14, 15.5 y 0.25, 1")
print(resultado)
































