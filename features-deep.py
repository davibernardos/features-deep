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

#treinamento do modelo
model = VGG16(weights='imagenet', include_top=False)

#armazenamento temporário de características
texto = ""
textoin = ""

#passa pelo repositório de imagens
for a in range(1,1001):
    
    #as imagens estão soltas no diretório corrente 
    #para evitar as diferenças de sistema operacional 
    img_path = str(a-1) + '.jpg'
    print("processando: " + img_path)
    
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    features = model.predict(x)


    for i in range(1,len(features[0][0][0])):
        texto +=  str(i) + ";"
    texto += "class\n"
    
    
    #extrai características (apenas parte delas)
    for i in range(len(features[0][0][0])):
        texto += (str(features[0][0][0][i]))
        texto += (";")
    
    classe = int(a/100)
    print("classe: " + str(classe))
    texto += (str(classe) + "\n")
    
    if a % 100 == 0:
        arqin = open("saidaFeatures.csv", "r")
        textoin = arqin.read()
        textoin += texto
        texto = "" 

        arqout = open("saidaFeatures.csv", "w")
        arqout.write(str(textoin))
        arqout.close()


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