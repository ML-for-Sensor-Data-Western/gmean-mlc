import pandas as pd
import os

image_dir = "/mnt/datassd0/sewer-data/images/" 

df_training = pd.read_csv("/mnt/datassd0/sewer-data/full_dataset_2/SewerML_Train.csv")
df_val = pd.read_csv("/mnt/datassd0/sewer-data/full_dataset_2/SewerML_Val.csv")

'''df_training2 = [] # df_training[os.path.isfile(image_dir + df_training.Filename)]
df_val2 = [] # df_val[os.path.isfile(image_dir + df_val.Filename)]

for index, row in df_val.iterrows():
    print(row.Filename, type(row.Filename))
    if type(row.Filename) ==str:
        file_path = os.path.join(image_dir, row.Filename) 
        if os.path.isfile(file_path):
            df_val2.append(row)

print("created validation set...")
df_val2 = pd.DataFrame(df_val2)

for index, row in df_training.iterrows():
    if type(row.Filename) ==str:
        file_path = os.path.join(image_dir, row.Filename)
        if os.path.isfile(file_path):
            df_training2.append(row)

df_training2 = pd.DataFrame(df_training2)
'''

train_subset = df_training.sample(n=400000)
val_subset = df_val.sample(n=50000)
test_subset = df_val.sample(n=50000)




train_subset.to_csv("/mnt/datassd0/sewer-data/training_sub_set/500k_subset/SewerML_Train.csv", index=False)
val_subset.to_csv("/mnt/datassd0/sewer-data/training_sub_set/500k_subset/SewerML_Val.csv", index=False)
test_subset.to_csv("/mnt/datassd0/sewer-data/training_sub_set/500k_subset/testing/SewerML_Val.csv", index=False)