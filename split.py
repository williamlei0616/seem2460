import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Read the raw dataset
raw_df = pd.read_csv("raw_df.csv")

# Assume the target is in the 'stroke' column
# Identify majority and minority classes
df_majority = raw_df[raw_df.stroke == 0]
df_minority = raw_df[raw_df.stroke == 1]

# Downsample the majority class to the number of minority samples
n_minority = len(df_minority)
df_majority_down = resample(
    df_majority,
    replace=False,  # sample without replacement
    n_samples=n_minority,
    random_state=42,
)

# Now upsample both classes to the original majority size.
n_majority = len(df_majority)
df_majority_up = resample(
    df_majority_down,
    replace=True,  # upsample with replacement
    n_samples=n_majority,
    random_state=42,
)

df_minority_up = resample(
    df_minority, replace=True, n_samples=n_majority, random_state=42
)

# Combine upsampled classes to form a balanced dataset
balanced_df = pd.concat([df_majority_up, df_minority_up])
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(
    drop=True
)  # shuffle

# Split into train, validation and test sets (70%/15%/15%)
train_df, temp_df = train_test_split(
    balanced_df, test_size=0.3, stratify=balanced_df.stroke, random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df.stroke, random_state=42
)

# Save to CSV files
train_df.to_csv("train_df.csv", index=False)
val_df.to_csv("val_df.csv", index=False)
test_df.to_csv("test_df.csv", index=False)
