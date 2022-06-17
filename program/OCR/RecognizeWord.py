import cv2
import numpy as np
import string
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional,Dropout
from keras.models import Model
import keras.backend as K

from spellchecker import SpellChecker
import tensorflow as tf
from PIL import Image

spell = SpellChecker()




punclist='.?:;"'
punclist2="-+/()[]!`,|*&^%$#@'"


smallletters=string.ascii_lowercase
capitalletters=string.ascii_uppercase
digits=string.digits
char_list=smallletters+capitalletters+digits+punclist+punclist2#+'`'	#dla interpunkcji
#char_list = string.ascii_letters+string.digits		#dla podstawowego






inputs = Input(shape=(32,128,1))
 

conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)

pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
 
conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
 
conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
 
conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)

pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
 
conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)

batch_norm_5 = BatchNormalization()(conv_5)
 
conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
 
conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
 
squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
 

blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)
 
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)


act_model = Model(inputs, outputs)




act_model.load_weights('wszystkie_gen2.hdf5') # wszystkie znaki bez przecinka , nie działa wypisuje SSSSS jako odstepy
#act_model.load_weights('CRNN_model_downloaded.hdf5')# male litery podstawowy pobrany
#act_model.load_weights('my_CRNN_model.hdf5')# male litery podstawowy przeszkolony samemu, tutaj powiela  eee na końcu wyrazów
#act_model.load_weights('myn_best_model.hdf5')# male litery podstawowy przeszkolony samemu, tutaj działa




def recognize_words(line_indicator,word_array,n_lines):

    file=open('recognized_texts.txt','w')

    line_rec=[]
    for listidx in range(n_lines):
        line_rec.append([])


    predictions=act_model.predict(word_array)
   

    out = K.get_value(K.ctc_decode(predictions, input_length=np.ones(predictions.shape[0])*predictions.shape[1],
                         greedy=True)[0][0])

    lw_idx=0
   
    for wordidxs in out:
        word=[]
        for char in wordidxs:
            if int(char)!=-1:
                word.append(char_list[int(char)])
        #word=spell.correction(''.join(word))
        word=''.join(word)
        line_rec[line_indicator[lw_idx]].append(word)
        lw_idx+=1

    for listidx in range(n_lines):
        line=' '.join(line_rec[listidx])
        print(line)
        file.writelines(line+'\n')
    file.close()

   







    




