import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Read the raw dataset
raw_df = pd.read_csv("raw_df.csv")

# First split off train (70%) and temp (30%), stratified on the target
train_df, temp_df = train_test_split(
    raw_df, test_size=0.3, stratify=raw_df.stroke, random_state=42
)

# Split temp into val (15%) and test (15%), also stratified
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df.stroke, random_state=42
)

# Now apply up/down-sampling only on the TRAIN set
# Separate majority/minority in train
train_majority = train_df[train_df.stroke == 0]
train_minority = train_df[train_df.stroke == 1]

# Downsample majority to size of minority
n_min = len(train_minority)
maj_down = resample(train_majority, replace=False, n_samples=n_min, random_state=42)

# Upsample both classes back to original majority size in train
n_maj = len(train_majority)
maj_up = resample(maj_down, replace=True, n_samples=n_maj, random_state=42)
min_up = resample(train_minority, replace=True, n_samples=n_maj, random_state=42)

balanced_train = pd.concat([maj_up, min_up])
balanced_train = balanced_train.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
balanced_train.to_csv("train_df.csv", index=False)
val_df.to_csv("val_df.csv", index=False)
test_df.to_csv("test_df.csv", index=False)
