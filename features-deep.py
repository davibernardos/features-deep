#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 06:30:21 2018

@author: davibernardo
"""

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os

#n_caracteristicas_coletadas = 512 + 1
n_caracteristicas_coletadas = 3585 + 1
total_de_imagens = 1000 + 1
intervalo_de_escrita = 2
texto = ""
caminho_img = os.path.abspath("imagens")

try:
    os.remove("saidaFeatures.csv")
    print("\n << O arquivo csv antigo foi removido >>")
except FileNotFoundError:
    print("\n> A cada {} imagens o arquivo csv será atualizado.".format(intervalo_de_escrita))


print("\n> Iniciando o treinamento do modelo...")
model = VGG16(weights='imagenet', include_top=False)

print("\n> Criando o cabeçalho do aquivo csv com {} colunas...".format(n_caracteristicas_coletadas))
for i in range(1,n_caracteristicas_coletadas):
    texto +=  str(i) + ","
texto += "class\n"

if os.path.exists(caminho_img):
    print("\n> Diretório de imagens ok.")
    
    print("\n> Iniciando a extração de caracteristicas de {} imagens...".format(total_de_imagens))
    for a in range(1,total_de_imagens):
        
        #as imagens estão soltas no diretório corrente 
        #para evitar as diferenças de sistema operacional
        if os.name == "posix":
            img_path = "{}/{}.jpg".format(caminho_img, str(a-1))
        else:
            img_path = "{}\{}.jpg".format(caminho_img, str(a-1))
       
        print("processando: {}".format(os.path.basename(img_path)))
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
            
        features = model.predict(x)
        
        #extrai características (apenas parte delas)
        for i in range(len(features[0][0][0])):
            #para rodar com 512 caracteristicas precisa comentar esse for
            for j in range(len(features[0][0])):
                texto += (str(features[0][0][j][i]))
                texto += (",")
            
        classe = int(a/100)
        print("classe: " + str(classe))
        texto += (str(classe) + "\n")
            
        if a % intervalo_de_escrita == 0:
            arq = open("saidaFeatures.csv", "a")
            arq.write(str(texto))
            arq.close()
            texto = ""
            
else:
    print("\n << Você precisa criar um diretório com o nome 'imagens'!! >>") 
    print("\n> Cuidado, o arquivo que o professor disponibilizou tem uma extensão '.org' que poderá causar problemas. >>")



'''

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
'''