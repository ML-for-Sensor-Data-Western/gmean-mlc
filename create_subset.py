import pandas as pd

df_training = pd.read_csv("E:\quinn\sewer_ml_dataset\SewerML_Train.csv")
df_val = pd.read_csv("E:\quinn\sewer_ml_dataset\SewerML_Val.csv")

train_subset = df_training.sample(n=80000)
val_subset = df_val.sample(n=20000)

train_subset.to_csv("", index=False)
val_subset.to_csv("", index=False)