# Import necessary libraries
import pandas as pd
import os
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from features import Features
from functools import reduce
import joblib  # For saving and loading the model
import random
# Define the key and target column
key = "row_index"
target_column = 'piezo_groundwater_level_category'

# Paths for the raw data and the preprocessed data
raw_file_path = 'X_train_Hi5.csv'
processed_data_path = 'preprocessed_data.csv'  # File to save/load preprocessed data
processed_target_path = 'preprocessed_target.csv'  # File to save/load target variable
model_path = 'trained_model.pkl'  # File to save/load the trained model
label_mapping_path = 'label_mapping.pkl'  # File to save/load label mappings

# Define summer months (June, July, August)
summer_months = [6, 7, 8,9]

# Check if preprocessed data exists
if os.path.exists(processed_data_path) and os.path.exists(processed_target_path):
    print("Loading preprocessed data...")
    X = pd.read_csv(processed_data_path)
    y_encoded = pd.read_csv(processed_target_path)['target']
else:
    print("Preprocessing data...")
    # Desired number of samples
    n_samples = 100000
 
    # First, get the total number of lines in the CSV file
    with open(raw_file_path, 'r') as f:
        total_lines = sum(1 for line in f) - 1  # Subtract 1 for the header

    # Ensure that the number of samples is less than the total lines
    n_samples = min(n_samples, total_lines)

    # Generate a sorted list of line numbers to skip
    skip_lines = sorted(random.sample(range(1, total_lines + 1), total_lines - n_samples))

    # Read the CSV, skipping the randomly selected lines
    data = pd.read_csv(raw_file_path, skiprows=skip_lines)

    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)
    # Drop rows with missing target values
    data = data.dropna(subset=[target_column])
    y = data[target_column]

    # Construct features using the Features class
    features = Features(data, key)
    meteo_features = features.construct_meteo_features()
    piezo_features = features.construct_piezo_features()
    seasonality_features = features.construct_seasonality_features()
    withdrawal_features = features.construct_withdrawal_features()
    
    # Merge all features on the key
    dfs = [meteo_features, piezo_features, withdrawal_features, seasonality_features]
    X = reduce(lambda left, right: pd.merge(left, right, on=key), dfs)
    
    # Create 'is_summer' flag
    X['is_summer'] = X['month'].isin(summer_months)
    
    # Encode target variable
    y_encoded = y.astype('category').cat.codes
    # Save the label mapping (code to category)
    label_mapping = dict(enumerate(y.astype('category').cat.categories))
    # Save label mapping to file
    joblib.dump(label_mapping, label_mapping_path)

    # Save the preprocessed data for future use
    X.to_csv(processed_data_path, index=False)
    pd.DataFrame({'target': y_encoded}).to_csv(processed_target_path, index=False)
    print("Preprocessed data saved.")


# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove the target and key columns from features if present
for col in [target_column, key, 'is_summer', 'month']:
    if col in numerical_cols:
        numerical_cols.remove(col)
    if col in categorical_cols:
        categorical_cols.remove(col)

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create a pipeline that includes the preprocessor and the classifier
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=100,
        random_state=42,
        device="cuda",
        use_label_encoder=False,
        eval_metric='mlogloss'
    ))
])

# Now, split the data focusing on summer months

# Separate summer and non-summer data
summer_data = X[X['is_summer']]
non_summer_data = X[~X['is_summer']]

# Define the desired proportion of summer data in the training set
desired_prop_summer_train = 0.8  # Adjust this value as needed

# Total number of samples
total_samples = len(X)

# Define the size of the training set (e.g., 80% for training)
train_size = 0.8
N_train = int(train_size * total_samples)

# Calculate the number of summer and non-summer samples for training
N_summer_train = int(desired_prop_summer_train * N_train)
N_non_summer_train = N_train - N_summer_train

# Shuffle the data
summer_data = summer_data.sample(frac=1, random_state=42)
non_summer_data = non_summer_data.sample(frac=1, random_state=42)

# Ensure we don't exceed available samples
N_summer_train = min(N_summer_train, len(summer_data))
N_non_summer_train = min(N_non_summer_train, len(non_summer_data))

# Split data for training
summer_train = summer_data.iloc[:N_summer_train]
non_summer_train = non_summer_data.iloc[:N_non_summer_train]
train_data = pd.concat([summer_train, non_summer_train])
train_data = train_data.sample(frac=1, random_state=42)

# Remaining data for validation
summer_val = summer_data.iloc[N_summer_train:]
non_summer_val = non_summer_data.iloc[N_non_summer_train:]
val_data = pd.concat([summer_val, non_summer_val])
val_data = val_data.sample(frac=1, random_state=42)

# Extract features and labels
X_train = train_data.drop(columns=['is_summer', 'month'])
y_train = y_encoded[train_data.index]
X_val = val_data.drop(columns=['is_summer', 'month'])
y_val = y_encoded[val_data.index]

# Check if model is already trained and saved
if os.path.exists(model_path):
    print("Loading trained model...")
    clf = joblib.load(model_path)
else:
    print("Training the model...")
    # Train the model
    clf.fit(X_train, y_train)
    # Save the trained model
    joblib.dump(clf, model_path)
    print("Model saved.")

# Predict on validation set
y_pred = clf.predict(X_val)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_val, y_pred))

# Calculate the F1 score
f1 = f1_score(y_val, y_pred, average='weighted')
print("\nF1 Score:")
print(f1)

# Now, predict on the test data and save the predictions
test_file_path = 'X_test_Hi5.csv'
output_file_path = 'predictions.csv'

# Load the test data
print("\nLoading test data...")
test_data = pd.read_csv(test_file_path, low_memory=False)

# Construct features for the test data
features_test = Features(test_data, key)
meteo_features_test = features_test.construct_meteo_features()
piezo_features_test = features_test.construct_piezo_features()
withdrawal_features_test = features_test.construct_withdrawal_features()
seasonality_features_test = features_test.construct_seasonality_features()

# Merge all features on the key
dfs_test = [meteo_features_test, piezo_features_test, withdrawal_features_test, seasonality_features_test]
X_test = reduce(lambda left, right: pd.merge(left, right, on=key), dfs_test)

# Drop unnecessary columns
X_test = X_test.drop(columns=[key, 'is_summer', 'month'], errors='ignore')

# Ensure that the columns in X_test match those in X_train
print("\nAligning test data columns with training data...")
# Add missing columns in X_test
missing_cols = set(X_train.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0  # You can choose an appropriate default value

# Remove extra columns in X_test
extra_cols = set(X_test.columns) - set(X_train.columns)
if extra_cols:
    X_test = X_test.drop(columns=extra_cols)

# Ensure the order of columns matches
X_test = X_test[X_train.columns]

# Predict on the test data
print("Making predictions on test data...")
predictions_encoded = clf.predict(X_test)

# Load label mapping to convert codes back to categories
label_mapping = joblib.load(label_mapping_path)
# Map the encoded predictions back to original labels
predictions = pd.Series(predictions_encoded).map(label_mapping)

# Prepare the output DataFrame
output_df = pd.DataFrame({
    'row_index': X_test.index,
    'piezo_groundwater_level_category': predictions
})

# Save the predictions to a CSV file
output_df.to_csv(output_file_path, index=False)
print(f"Predictions saved to {output_file_path}")
