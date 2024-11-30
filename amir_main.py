# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from features import Features
from functools import reduce

# Define the key and target column
key = "row_index"
target_column = 'piezo_groundwater_level_category'

# Load the data
file_path = 'X_train_Hi5.csv'
data = pd.read_csv(file_path, nrows=500_000)

# Drop rows with missing target values
data = data.dropna(subset=[target_column])
y = data[target_column]

# Construct features using the Features class
features = Features(data, key)
meteo_features = features.construct_meteo_features()
piezo_features = features.construct_piezo_features()
withdrawal_features = features.construct_withdrawal_features()

# Merge all features on the key
dfs = [meteo_features, piezo_features, withdrawal_features]
X = reduce(lambda left, right: pd.merge(left, right, on=key), dfs)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Handle 'meteo_if_snow' if it's categorical
if 'meteo_if_snow' in X.columns and X['meteo_if_snow'].dtype == 'object':
    categorical_cols.append('meteo_if_snow')
    numerical_cols.remove('meteo_if_snow')

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
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
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

# Calculate the F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
print("\nF1 Score:")
print(f1)

# Access the trained RandomForestClassifier from the pipeline
model = clf.named_steps['classifier']

# Get the feature importances
importances = model.feature_importances_
print("\nFeature Importances:")
print(importances)

# Get the feature names from the preprocessor
# Numerical feature names remain the same
numerical_feature_names = numerical_cols
print("\nNumerical Feature Names:")
print(numerical_feature_names)
