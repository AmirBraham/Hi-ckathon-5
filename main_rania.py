# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from features import Features

key = "row_index"

# define the target column
target_column = 'piezo_groundwater_level_category'

# Load the data
# If your dataset is large, you can read it in chunks. For simplicity, we'll read it all at once here.
# Replace 'X_train_Hi5.csv' with your actual file path
file_path = 'X_train_Hi5.csv'

# Read the data
data = pd.read_csv(file_path,nrows=100000)

# Drop rows with missing target values
data = data.dropna(subset=[target_column])

# Separate features and target variable
features = Features(data,key)
water_features = features.construct_water_features()
withdrawal_features = features.construct_withdrawal_features()

X = pd.merge(water_features,
             withdrawal_features,
             on=key)
y = data[target_column]

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# For this dataset, let's check which columns are categorical
print("Categorical columns:", categorical_cols)

# If 'meteo_if_snow' is not numerical, treat it as categorical
# Let's assume 'meteo_if_snow' is categorical (e.g., 'Yes', 'No')
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

# Create a pipeline that includes the preprocessor and the classifier
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Encode target variable
# Since it's categorical (e.g., 'very low', 'low', etc.), we need to encode it
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

# Access the trained RandomForestClassifier from the pipeline
model = clf.named_steps['classifier']

# Get the feature importances
importances = model.feature_importances_
print(importances)
# Get the feature names from the preprocessor
# Numerical feature names remain the same
numerical_feature_names = numerical_cols
print(numerical_feature_names)