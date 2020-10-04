import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
CSV_COLUMN_NAMES = ['MesoNet, FaceForensics++, EfficientNetB4Att']
LABELS = ['FAKE, REAL']
train = pd.read_csv('COMBINED_Train.csv', names=CSV_COLUMN_NAMES, header=0)
train_labels = pd.read_csv('metadata_Train.json', names=LABELS, header=0)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(3, 1)),
    keras.layers.Dense(9, activation='relu'),  
    keras.layers.Dense(2, activation='softmax') 
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train, train_labels, epochs=10)
output = pd.read_csv('COMBINED_Test.csv', names=CSV_COLUMN_NAMES, header=0)
output_labels = pd.read_csv('metadata_Test.json', names=LABELS, header=0)
res = []
df_test=pd.read_csv('DNN.csv')
for index, rows in train.iterrows():  
    prediction = model.predict(rows)
    if(prediction[0] > prediction[1]):
        res.append(1-prediction[0])
    else:
        res.append(prediction[1])
df_test['label']=res
df_test.to_csv('DNN.csv',index=False)
    
    