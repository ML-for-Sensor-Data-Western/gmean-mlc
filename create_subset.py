import pandas as pd
import os

image_dir = "/mnt/datassd0/sewer-data/images/" 

df_training = pd.read_csv("/mnt/datassd0/sewer-data/annotations/SewerML_Train.csv")
df_val = pd.read_csv("/mnt/datassd0/sewer-data/annotations/SewerML_Val.csv")
df_test = pd.read_csv("/mnt/datassd0/sewer-data/annotations/SewerML_Test.csv")

num_images = len(os.listdir(image_dir))

num_training = len(df_training)
num_val = len(df_val)
# num_test = len(df_test)

num_data = num_training + num_val# + num_test

print("Number of files in image_dir: ", num_images)

print("\nNumber of images in sheets: ", num_data)
print("Number of training images: ", num_training)
print("Number of validation images: ", num_val)
# print("Number of test images: ", num_test)

print("\nNumber of missing files: ", num_data - num_images)

'''# Check missing files and write filename to txt file
missing_filenames = []

for df in [df_training, df_val]:
    for index, row in df.iterrows():
            file_path = os.path.join(image_dir, row.Filename)
            if not os.path.isfile(file_path):
                missing_filenames.append(row.Filename)

print("Number of missing files from search:", len(missing_filenames))

with open("missing_filenames.txt", "w") as f:
    for filename in missing_filenames:
        f.write(filename + "\n")
'''

''' # Create new dataframes with only the files that exist
df_training2 = [] # df_training[os.path.isfile(image_dir + df_training.Filename)]
df_val2 = [] # df_val[os.path.isfile(image_dir + df_val.Filename)]

for index, row in df_val.iterrows():
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

# Create a subset of the dataframes
train_subset = df_training.sample(n=400000, random_state=1)
df_val_shuffled = df_val.sample(frac=1, random_state=1)
val_subset = df_val_shuffled.iloc[:50000]
test_subset = df_val_shuffled.iloc[50000:100000]

train_subset.to_csv("/mnt/datassd0/sewer-data/annotations_500k/SewerML_Train.csv", index=False)
val_subset.to_csv("/mnt/datassd0/sewer-data/annotations_500k/SewerML_Val.csv", index=False)
test_subset.to_csv("/mnt/datassd0/sewer-data/annotations_500k/SewerML_Test.csv", index=False)