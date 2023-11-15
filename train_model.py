from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from modules import *

def train(csvFilePath, columnName, imagesDirPath, classNames, modelPath):
  # Load dataset
  data = loadDataFrame(csvFilePath)
  pixels, gender = csvFileToTrainTestData(data, imagesDirPath, columnName)

  # Split your dataset into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(pixels, gender, test_size=0.2, random_state=42)

  # Experiment with different numbers of layers
  num_conv_layers_list = [5]
  num_dense_layers_list = [1]
  num_units_list = [64]
  batch_sizes_list = [5]
  epochs_list = [10]
  optimizers_list = ['Adamax']
  learn_rates_list = [0.001]
  momentums_list = [0.0]
  # num_conv_layers_list = [1, 2]
  # num_dense_layers_list = [1, 2]
  # num_units_list = [64, 128]
  # batch_sizes_list = [10, 20, 40, 60, 80, 100]
  # epochs_list = [10, 50, 100]
  # optimizers_list = ['Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam', 'RMSprop', 'SGD']
  # learn_rates_list = [0.001, 0.01, 0.1, 0.2, 0.3]
  # momentums_list = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

  for num_conv_layers in num_conv_layers_list:
    for num_dense_layers in num_dense_layers_list:
      for num_units in num_units_list:
        for batch_size in batch_sizes_list:
          for epoch in epochs_list:
            for optimizer in optimizers_list:
              for learn_rate in learn_rates_list:
                for momentum in momentums_list:
                  # model = create_model(num_conv_layers, num_dense_layers, num_units, optimizer, learn_rate, momentum)
                  model = create_model_v3(optimizer)
                  model_path = modelPath
                  checkpointer = ModelCheckpoint(model_path, monitor='loss',verbose=1,save_best_only=True,
                                save_weights_only=False, mode='auto',save_freq='epoch')
                  callback_list=[checkpointer]
                  history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(X_test, y_test),callbacks=[callback_list])
                  saveHistory(f"./{columnName}_history.json", history)
                  print(f"Conv Layers: {num_conv_layers}, Dense Layers: {num_dense_layers}, Units: {num_units}, Batch size: {batch_size}, Epoch: {epoch}, Optimizer: {optimizer}, Learn Rate: {learn_rate}, Momentum: {momentum}")
                  test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
                  print(f"Test accuracy: {test_acc}\n")
                  print(f"Test loss: {test_loss}\n")
                  evaluate(model, X_test, y_test, classNames)


# train('./prepped_data/gender_data_raw.csv', 'Male', 'prepped_data/gender_images/', ['male', 'female'], './gender_model2.h5')
# train('./prepped_data/attractive_data_raw.csv', 'Attractive', 'prepped_data/attractive_images/', ['Attractive', 'Average'], './attractive_model.h5')
# train('./prepped_data/smiling_data_raw.csv', 'Smiling', 'prepped_data/smiling_images/', ['smiling', 'not smiling'], './smiling_model.h5')