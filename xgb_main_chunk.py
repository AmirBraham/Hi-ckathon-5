# Import necessary libraries
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from features import Features
from functools import reduce


# check if GPU is available
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(os.environ['CUDA_VISIBLE_DEVICES'])

key = "row_index"

# Define the target column
target_column = 'piezo_groundwater_level_category'

# Load the data
file_path = 'X_train_Hi5.csv'
data = pd.read_csv(file_path, nrows=500_000)

# Drop rows with missing target values
data = data.dropna(subset=[target_column])
y = data[target_column]

# Separate features and target variable
features = Features(data, key)
meteo_features = features.construct_meteo_features()
piezo_features = features.construct_piezo_features()
withdrawal_features = features.construct_withdrawal_features()

dfs = [meteo_features, piezo_features, withdrawal_features]

data = reduce(lambda left, right: pd.merge(left, right, on=key), dfs)

data.drop('row_index', axis=1, inplace=True)

X = data
print(X.shape)
print(X.columns)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Check if 'meteo_if_snow' is categorical and adjust columns accordingly
if 'meteo_if_snow' in X.columns and X['meteo_if_snow'].dtype == 'object':
    categorical_cols.append('meteo_if_snow')
    numerical_cols.remove('meteo_if_snow')

# Handle missing values and encoding
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

# Replace the classifier with XGBClassifier
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        device='cuda',  # Use GPU for training (if available)
        predictor='gpu_predictor',  # Use GPU for predictions (if available)
        n_estimators=300,  # Number of boosting rounds
        learning_rate=0.1,  # Step size shrinkage
        max_depth=5,  # Maximum depth of a tree
        colsample_bytree=0.8,  # Subsample ratio of columns for each tree
        subsample=0.8,  # Subsample ratio of the training instances
        random_state=42  # Ensures reproducibility
    ))
])

# Encode target variable
y_encoded = y.astype('category').cat.codes

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Access the trained XGBClassifier from the pipeline
model = clf.named_steps['classifier']

# Feature importance from XGBoost
importances = model.feature_importances_
print("\nFeature Importances:")
print(importances)

# Get the numerical feature names
numerical_feature_names = numerical_cols
print("\nNumerical Feature Names:")
print(numerical_feature_names)


# Extract feature names after preprocessing
feature_names = []
for transformer_name, transformer, columns in preprocessor.transformers_:
    if transformer_name == 'num':
        feature_names.extend(columns)
    elif transformer_name == 'cat':
        ohe = transformer.named_steps['encoder']
        encoded_feature_names = ohe.get_feature_names_out(columns)
        feature_names.extend(encoded_feature_names)

# Create a dictionary mapping feature names to their importances
feature_importance_dict = dict(zip(feature_names, importances))

# Print the dictionary
print("\nFeature Importances:")
print(feature_importance_dict)
