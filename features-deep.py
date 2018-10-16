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
for a in range(151,201):
    
    #as imagens estão soltas no diretório corrente 
    #para evitar as diferenças de sistema operacional 
    img_path = str(a-1) + '.jpg'
    print(img_path)
    
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    features = model.predict(x)
    
    #extrai características (apenas parte delas)
    for i in range(len(features[0][0][0])):
        texto += (str(features[0][0][0][i]))
        texto += (";")
    
    #verifica a classe da imagem
    print(str(int(a/100)))
    texto += (str(int(a/100)) + "\n")        

arqin = open("saidaFeatures.csv", "r")
textoin = arqin.read()
textoin += texto

arq = open("saidaFeatures.csv", "w")
arq.write(str(textoin))
arq.close()


#img_path = "image/" + str(a) + '.jpg'
    
#print(features[0][0][0][0])
    
#    for i in features:
#        for j in i:
#            for k in j:
#                for l in k:
#                    arq.write(str(l)+";")
                    
#    if cont == 99:
#        cont = 0
#        arq.write(str(class_cont) + "\n")
#        class_cont += 1
#    else:
#        cont += 1

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