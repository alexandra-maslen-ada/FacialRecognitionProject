import cv2
import os
from modules import *

def grey_scale_images(max = -1):

  input_directory = './original_data/sample_images/'

  output_directory = './prepped_data/sample_images/'

  os.makedirs(output_directory, exist_ok=True)

  image_files = os.listdir(input_directory)

  for image_file in image_files:
    if image_file.endswith('.jpg') or image_file.endswith('.png'):
      # Load the color image
      image_path = os.path.join(input_directory, image_file)
      color_image = cv2.imread(image_path)
      
      # Convert the color image to grayscale
      grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
      
      # Save the grayscale image to the output directory
      output_path = os.path.join(output_directory, image_file)
      cv2.imwrite(output_path, grayscale_image)

def chooseImagesForAttributeTraining(data, outputDirPath):
  input_directory = './original_data/images/'
  for index, row in data.iterrows():
    print(index)

    # load image
    image_path = os.path.join(input_directory, row['image_id'])
    color_image = cv2.imread(image_path)

    # copy image
    output_path = os.path.join(outputDirPath, row['image_id'])
    color_image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    print(output_path)
    cv2.imwrite(output_path, grayscale_image)

# grey_scale_images()
data = loadDataFrame('prepped_data/smiling_data_raw.csv', getConverters())
chooseImagesForAttributeTraining(data, './prepped_data/gender_images/')
chooseImagesForAttributeTraining(data, './prepped_data/attractive_images/')
chooseImagesForAttributeTraining(data, './prepped_data/smiling_images/')