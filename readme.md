# Facial Recognition Project

## Prerequisites

1) Create the following sub directories in this project...

```
./original_data/
./prepped_data/
./prepped_data/attractive_images/
./prepped_data/gender_images/
./prepped_data/smiling_images/
./tested_data/
/tested_data/attractive/
/tested_data/attractive/FN/
/tested_data/attractive/FP/
/tested_data/attractive/TN/
/tested_data/attractive/TP/
/tested_data/gender/
/tested_data/gender/FN/
/tested_data/gender/FP/
/tested_data/gender/TN/
/tested_data/gender/TP/
/tested_data/smiling/
/tested_data/smiling/FN/
/tested_data/smiling/FP/
/tested_data/smiling/TN/
/tested_data/smiling/TP/
```

2) Download the contents of the Kaggle project [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and put it into the directory `original_data`.

3) Python3

4) Install the following python libraries via pip.

* `pandas`
* `scikit-learn`
* `numpy`
* `keras`
* `opencv-python`

## How to use

1) To review and explore the dataset, open `explore_data.ipynb` in an application that supports interactive python notebooks.  I used Visual Studio for this.

2) Prep the CSV data by running `prep_csv_data.py`. This will create a file containing 50,000 entries of just the image column and label column for the 3 classification tasks.

3) Prep the image file data by running `prep_images.py`. This will create a grey scaled copy of each image chosen in the previous step and store it in a new directory.

4) Create the models by running `train_model.py`.  There are 3 commented out lines at the bottom of the script - uncomment the one for the model you want to train.  This script takes about 1 hour to run. It will generate a `.h5` file - the model, and a `.json` file which is a report on how the training went.

5) Run `classifier_examples.py` to create examples of how the test data was classified. This will copy images and place them into the `tested_data`.

6) Run `agent.py` - editing the line `filePath = './tested_data/gender/tp/9.jpg'` to point at the file you want to profile.

7) Generate "loss / accuracy" graphs by running the interactive pythong notebook `loss_accuracy_graph_by_epochs.ipynb`.
