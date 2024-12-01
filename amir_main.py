# Import necessary libraries
import pandas as pd
import os
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from functools import reduce
import joblib  # For saving and loading the model
from sklearn.ensemble import RandomForestClassifier

from features import Features

# Define the key and target column
key = "row_index"
target_column = 'piezo_groundwater_level_category'

# Paths for the raw data and the preprocessed data
raw_file_path = 'X_train_Hi5.csv'
processed_data_path = 'preprocessed_data.csv'  # File to save/load preprocessed data
processed_target_path = 'preprocessed_target.csv'  # File to save/load target variable
model_path = 'trained_model.pkl'  # File to save/load the trained model
label_mapping_path = 'label_mapping.pkl'  # File to save/load label mappings
A
# Check if preprocessed data exists
if os.path.exists(processed_data_path) and os.path.exists(processed_target_path):
    print("Loading preprocessed data...")
    X = pd.read_csv(processed_data_path)
    y_encoded = pd.read_csv(processed_target_path)['target']
else:
    print("Preprocessing data...")
    # Desired number of samples
    n_samples = 500000

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
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    # Drop rows with missing target values
    data = data.dropna(subset=[target_column])
    y = data[target_column]

    # Construct features using the Features class
    features = Features(data, key)
    meteo_features = features.construct_meteo_features()
    piezo_features = features.construct_piezo_features()
    withdrawal_features = features.construct_withdrawal_features()
    seasonality_features = features.construct_seasonality_features()

    # Merge all features on the key
    dfs = [meteo_features, piezo_features, withdrawal_features, seasonality_features]
    X = reduce(lambda left, right: pd.merge(left, right, on=key), dfs)

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

# Drop 'key' column from X
if key in X.columns:
    X = X.drop(columns=[key])

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove target_column and key if present
for col in [target_column, key]:
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
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define parameter grid for Grid Search
param_grid = {
    'classifier__n_estimators': [100, 150],
    'classifier__max_depth': [None, 3,10, 15],
}

# Check if model is already trained and saved
if os.path.exists(model_path):
    print("Loading trained model...")
    clf = joblib.load(model_path)
else:
    print("Training the model with Grid Search...")
    # Set up GridSearchCV
    clf = GridSearchCV(pipeline, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
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
X_test = reduce(lambda left, right: pd.merge(left, right, on=key, how='left'), dfs_test)

# Preserve 'key' column for output
if key in X_test.columns:
    X_test_key = X_test[key]
    X_test = X_test.drop(columns=[key])
else:
    X_test_key = pd.Series(range(len(X_test)))

# Predict on the test data
print("Making predictions on test data...")
predictions_encoded = clf.predict(X_test)

# Load label mapping to convert codes back to categories
label_mapping = joblib.load(label_mapping_path)
# Map the encoded predictions back to original labels
predictions = pd.Series(predictions_encoded).map(label_mapping)

# Prepare the output DataFrame
output_df = pd.DataFrame({
    'row_index': X_test_key,
    'piezo_groundwater_level_category': predictions
})

# Save the predictions to a CSV file
output_df.to_csv(output_file_path, index=False)
print(f"Predictions saved to {output_file_path}")
