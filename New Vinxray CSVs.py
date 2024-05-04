import pandas as pd


def modify_and_save_csv(original_csv_path, new_csv_path, prefix_to_remove):
    # Load the original CSV file
    df = pd.read_csv(original_csv_path)

    # Check if 'full_path' column exists
    if 'full_path' not in df.columns:
        print(f"'full_path' column not found in {original_csv_path}.")
        return

    # Print the first few rows before modification for verification
    print(f"Before modification in {original_csv_path}:")
    print(df.head())

    # Remove the prefix from the entries in 'full_path' column
    df['full_path'] = df['full_path'].apply(lambda x: x.replace(prefix_to_remove, '').lstrip('/'))

    # Print the first few rows after modification for verification
    print(f"\nAfter modification in {original_csv_path}:")
    print(df.head())

    # Save the modified DataFrame to a new CSV file
    df.to_csv(new_csv_path, index=False)
    print(f"\nModified CSV saved as: {new_csv_path}")


# Paths of the original CSVs
train_csv_path = '/vinxray/vietnam_train.csv'
test_csv_path = '/vinxray/vietnam_test.csv'

# Prefixes to remove (adjust these if necessary)
train_prefix = "./vinxray/train"
test_prefix = "./vinxray/train" ##### under the "train file"

# Modify and save the train.csv without the prefix
modify_and_save_csv(train_csv_path, '../train_vt.csv', train_prefix)

# Modify and save the test.csv without the prefix
modify_and_save_csv(test_csv_path, '../test_vt.csv', test_prefix)
