from sklearn.model_selection import train_test_split
from keras.models import load_model
import cv2
from modules import *

def generate(csvPath, columnName, imagesDirPath, modelPath):
  print(f"Generating classifier examples for: {columnName}")
  data = loadDataFrame(csvPath)
  pixels, gender = csvFileToTrainTestData(data, imagesDirPath, columnName)

  X_train, X_test, y_train, y_test = train_test_split(pixels, gender, test_size=0.2, random_state=42)

  model = load_model(modelPath)
  predicted = model.predict(X_test)
  predicted = getSecondItemsAboveQualityThreshold(predicted)

  matrix = confusion_matrix(y_test, predicted)
  print("Confusion matrix")
  print(matrix)

  for i in range(len(predicted)):
    if (predicted[i] == 1 and y_test[i] == 1):
      cv2.imwrite(f"./tested_data/smiling/tp/{i}.jpg", X_test[i])
    if (predicted[i] == 0 and y_test[i] == 0):
      cv2.imwrite(f"./tested_data/smiling/tn/{i}.jpg", X_test[i])
    if (predicted[i] == 1 and y_test[i] == 0):
      cv2.imwrite(f"./tested_data/smiling/fp/{i}.jpg", X_test[i])
    if (predicted[i] == 0 and y_test[i] == 1):
      cv2.imwrite(f"./tested_data/smiling/fn/{i}.jpg", X_test[i])

generate('./prepped_data/gender_data_raw.csv', 'Male', 'prepped_data/gender_images/', 'gender_model.h5')
generate('./prepped_data/attractive_data_raw.csv', 'Attractive', 'prepped_data/attractive_images/', 'attractive_model.h5')
generate('./prepped_data/smiling_data_raw.csv', 'Smiling', 'prepped_data/smiling_images/', 'smiling_model.h5')