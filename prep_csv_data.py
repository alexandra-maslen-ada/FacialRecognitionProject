from modules import *
import random

def reduce(df, amount, columnName):
  classA_df = df[df[columnName] == 1]
  classB_df = df[df[columnName] == -1]

  print(f'Reducing the data by {amount} rows, while auto-balancing the data for: {columnName}')
  print('Before reducing...')
  print(f'Rows with "1": {len(classA_df)}')
  print(f'Rows with "1": {len(classB_df)}')
 
  sampled_classA_df = classA_df.sample(n=amount, random_state=1)
  sampled_classB_df = classB_df.sample(n=amount, random_state=1)

  print('After reducing...')
  print(f'Rows with "1": {len(sampled_classA_df)}')
  print(f'Rows with "1": {len(sampled_classB_df)}')

  balanced_df = pd.concat([sampled_classA_df, sampled_classB_df])
  return balanced_df.sample(frac=1, random_state=1).reset_index(drop=True)

# load CSV
data = loadDataFrame('original_data/list_attr_celeba.csv', getConverters())

# isolate gender and file path columns
gender_data = data[['image_id', 'Male']]
attractive_data = data[['image_id', 'Attractive']]
smiling_data = data[['image_id', 'Smiling']]

# resample
gender_data = reduce(gender_data, 25000, 'Male')
attractive_data = reduce(attractive_data, 25000, 'Attractive')
smiling_data = reduce(smiling_data, 25000, 'Smiling')

# save CSV
saveDataFrame(gender_data, 'prepped_data/gender_data_raw.csv')
saveDataFrame(attractive_data, 'prepped_data/attractive_data_raw.csv')
saveDataFrame(smiling_data, 'prepped_data/smiling_data_raw.csv')