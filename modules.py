from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import Counter
from sklearn.utils import resample
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import *
from tensorflow.keras import layers
from keras.layers import Conv2D,MaxPooling2D,AvgPool2D,GlobalAveragePooling2D,Dense,Dropout,BatchNormalization,Flatten,Input
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input,Activation,Add
from keras.models import Sequential,load_model,Model
import cv2
import json

def saveHistory(filePath, history):
  with open(filePath, "w") as history_file:
    json.dump(history.history, history_file)  

def getConverters():
  return {'image_id': str}

def loadDataFrame(fileName, converters={}):
   return read_csv(fileName, converters=converters)

def saveDataFrame(data, fileName):
  data.to_csv(fileName, index=False)

def encodeDataFrame(df):
  le = LabelEncoder()
  for column in df.columns:
    df[column] = le.fit_transform(df[column])
  return df

def resample(df, feature):
    from sklearn.utils import resample
    import pandas as pd
    majority_class = df[df[feature] == 0]
    minority_class = df[df[feature] == 1]
    minority_upsampled = resample(minority_class, n_samples=len(majority_class), random_state=42)
    resampled_df= pd.concat([majority_class, minority_upsampled])
    resampled_df = resampled_df.sample(frac=1, random_state=42)
    return resampled_df

def getSecondItemsAboveQualityThreshold(arr):
  secondItems = []
  threshold = 0.6
  for sub_array in arr:
      secondItems.append((sub_array[1] >= threshold).astype(int))
  return secondItems

def evaluate(model, x_test, y_test, classNames):
    predicted = model.predict(x_test)
    
    predicted = getSecondItemsAboveQualityThreshold(predicted)

    report = classification_report(y_test, predicted, target_names=classNames)

    print("Classification report")
    print(report)

    matrix = confusion_matrix(y_test, predicted)
    print("Confusion matrix")
    print(matrix)

def createOptimizer(type, learn_rate, momentum):
  match type:
    case 'Adadelta':
      return Adadelta(learning_rate=learn_rate, ema_momentum=momentum)
    case 'Adafactor':
      return Adafactor(learning_rate=learn_rate, ema_momentum=momentum)
    case 'Adagrad':
      return Adagrad(learning_rate=learn_rate, ema_momentum=momentum)
    case 'Adam':
      return Adam(learning_rate=learn_rate, ema_momentum=momentum)
    case 'AdamW':
      return AdamW(learning_rate=learn_rate, ema_momentum=momentum)
    case 'Adamax':
      return Adamax(learning_rate=learn_rate, ema_momentum=momentum)
    case 'Nadam':
      return Nadam(learning_rate=learn_rate, ema_momentum=momentum)
    case 'rmsprop':
      return RMSprop(learning_rate=learn_rate, ema_momentum=momentum)
    case 'sgd':
      return SGD(learning_rate=learn_rate, ema_momentum=momentum)

def create_model(num_conv_layers, num_dense_layers, num_units, optimizer, learn_rate, momentum):
  IMAGE_WIDTH = 178
  IMAGE_HEIGHT = 218
  model = keras.Sequential()

  model.add(layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))

  for _ in range(num_conv_layers):
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

  model.add(layers.Flatten())

  for _ in range(num_dense_layers):
    model.add(layers.Dense(num_units, activation='relu'))
  
  model.add(layers.Dense(1, activation='sigmoid'))

  model.compile(
    createOptimizer(optimizer,learn_rate, momentum),
    loss='binary_crossentropy',
    metrics=['accuracy']
  )

  return model

def create_model_v2(learn_rate, momentum):
  IMAGE_WIDTH = 178
  IMAGE_HEIGHT = 218
  model = keras.Sequential([
      keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(16, (3, 3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(32, (3, 3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(64, (3, 3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.Conv2D(128, (3, 3), activation='relu'),
      keras.layers.MaxPooling2D((2, 2)),
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Dense(64, 'relu', kernel_regularizer=keras.regularizers.l2(0.01)),
      keras.layers.BatchNormalization(),
      keras.layers.Dropout(0.6),
      keras.layers.Dense(1, activation='sigmoid')
  ])
  optimizer = keras.optimizers.legacy.Adam(learning_rate=learn_rate)
  model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
  return model

def create_model_v3(optimizer):
  input = Input(shape = (218,178,3))

  conv1 = Conv2D(32,(3, 3), padding = 'same', strides=(1, 1), kernel_regularizer=l2(0.001))(input)
  conv1 = Dropout(0.1)(conv1)
  conv1 = Activation('relu')(conv1)
  pool1 = MaxPooling2D(pool_size = (2,2)) (conv1)

  conv2 = Conv2D(64,(3, 3), padding = 'same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool1)
  conv2 = Dropout(0.1)(conv2)
  conv2 = Activation('relu')(conv2)
  pool2 = MaxPooling2D(pool_size = (2,2)) (conv2)

  conv3 = Conv2D(128,(3, 3), padding = 'same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool2)
  conv3 = Dropout(0.1)(conv3)
  conv3 = Activation('relu')(conv3)
  pool3 = MaxPooling2D(pool_size = (2,2)) (conv3)

  conv4 = Conv2D(256,(3, 3), padding = 'same', strides=(1, 1), kernel_regularizer=l2(0.001))(pool3)
  conv4 = Dropout(0.1)(conv4)
  conv4 = Activation('relu')(conv4)
  pool4 = MaxPooling2D(pool_size = (2,2)) (conv4)

  flatten = Flatten()(pool4)
  dense_1 = Dense(512,activation='relu')(flatten)
  drop_1 = Dropout(0.2)(dense_1)
  output = Dense(2,activation="sigmoid")(drop_1)

  model = Model(inputs=input,outputs=output)
  model.compile(optimizer=optimizer,loss=["sparse_categorical_crossentropy"],metrics=['accuracy'])
  return model

def csvFileToTrainTestData(data, dirPath, labelColumnName):
  pixels = []
  gender = []
  print('Loading images into memory...')
  for index, row in data.iterrows():
    img = cv2.imread(dirPath+row["image_id"])
    pixels.append(np.array(img))
    if (row[labelColumnName] == -1):
      gender.append(np.array(0))
    else:
      gender.append(np.array(1))
    # print(row['image_id'])
  pixels = np.array(pixels)
  gender = np.array(gender,np.uint64)
  return pixels, gender
 
