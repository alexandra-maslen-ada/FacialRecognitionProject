from keras.models import load_model
import cv2
import numpy as np

gender_model = load_model("gender_model.h5")
attractive_model = load_model("attractive_model.h5")
smiling_model = load_model("smiling_model.h5")

pixels = []
filePath = './tested_data/gender/tp/9.jpg'
image = cv2.imread(filePath)
pixels.append(image)
pixels = np.array(pixels)

gender_result = gender_model.predict(pixels)[ 0 ][ 1 ]
attractive_result = attractive_model.predict(pixels)[ 0 ][ 1 ]
smiling_result = smiling_model.predict(pixels)[ 0 ][ 1 ]

def classifier(percentage=0):
  if percentage >= 0.8:
    return 'is'
  if percentage >= 0.6:
    return 'is likely'
  return 'could be'

print(filePath)
print(f"Subject {classifier(gender_result)} male: {round(gender_result * 100)}%")
print(f"Subject {classifier(attractive_result)} attractive: {round(attractive_result * 100)}%")
print(f"Subject {classifier(smiling_result)} smiling: {round(smiling_result * 100)}%")