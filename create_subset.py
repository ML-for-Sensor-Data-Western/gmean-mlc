import pandas as pd
import os

IMAGE_DIR = "/mnt/datassd0/sewer-data/images/"
TRAIN_ANN = "/mnt/datassd0/sewer-data/annotations/SewerML_Train.csv"
VAL_ANN = "/mnt/datassd0/sewer-data/annotations/SewerML_Val.csv"

NUM_DEFECT = 50000
DEFECT_TO_NORMAL_RATIO = 10
TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.8, 0.1, 0.1


num_train_defect = int(NUM_DEFECT * TRAIN_SPLIT)
num_val_defect = int(NUM_DEFECT * VAL_SPLIT)
num_test_defect = int(NUM_DEFECT * TEST_SPLIT)
num_train_normal = num_train_defect * DEFECT_TO_NORMAL_RATIO
num_val_normal = num_val_defect * DEFECT_TO_NORMAL_RATIO
num_test_normal = num_test_defect * DEFECT_TO_NORMAL_RATIO

OUTPUT_DIR = f"/mnt/datassd0/sewer-data/defect_{int(NUM_DEFECT/1000)}k_1_to_{DEFECT_TO_NORMAL_RATIO}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df_train = pd.read_csv(TRAIN_ANN)
df_val = pd.read_csv(VAL_ANN)

df_train_defect = df_train[df_train["Defect"] == 1]
df_train_normal = df_train[df_train["Defect"] == 0]

df_val_defect = df_val[df_val["Defect"] == 1]
df_val_normal = df_val[df_val["Defect"] == 0]

# Create a subset of the dataframes
# Create train subset
train_subset_defect = df_train_defect.sample(n=num_train_defect, random_state=1)
train_subset_normal = df_train_normal.sample(n=num_train_normal, random_state=1)
train_subset = pd.concat([train_subset_defect, train_subset_normal])

# val and test subsets from original val set
df_val_defect_shuffled = df_val_defect.sample(frac=1, random_state=1)
df_val_normal_shuffled = df_val_normal.sample(frac=1, random_state=1)

val_subset_defect = df_val_defect_shuffled.iloc[:num_val_defect]
val_subset_normal = df_val_normal_shuffled.iloc[:num_val_normal]
val_subset = pd.concat([val_subset_defect, val_subset_normal])

test_subset_defect = df_val_defect_shuffled.iloc[
    num_val_defect : num_val_defect + num_test_defect
]
test_subset_normal = df_val_normal_shuffled.iloc[
    num_val_normal : num_val_normal + num_test_normal
]
test_subset = pd.concat([test_subset_defect, test_subset_normal])

train_subset.to_csv(os.path.join(OUTPUT_DIR, "SewerML_Train.csv"), index=False)
val_subset.to_csv(os.path.join(OUTPUT_DIR, "SewerML_Val.csv"), index=False)
test_subset.to_csv(os.path.join(OUTPUT_DIR, "SewerML_Test.csv"), index=False)


"""# Check missing files and write filename to txt file
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
"""

""" # Create new dataframes with only the files that exist
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
"""
