# Import necessary libraries
import dask.dataframe as dd
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
from xgboost import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(os.environ['CUDA_VISIBLE_DEVICES'])

key = "row_index"
target_column = 'piezo_groundwater_level_category'

# Load the data using Dask
file_path = 'X_train_Hi5.csv'
data = dd.read_csv(file_path)

# Drop rows with missing target values and persist to ensure faster operations
data = data.dropna(subset=[target_column]).persist()

# Extract the target variable
y = data[target_column]

# Construct features using your Features class
features = Features(data, key)
meteo_features = features.construct_meteo_features()
piezo_features = features.construct_piezo_features()
withdrawal_features = features.construct_withdrawal_features()

dfs = [meteo_features, piezo_features, withdrawal_features]

# Merge the feature sets
data = reduce(lambda left, right: dd.merge(left, right, on=key), dfs).persist()

X = data

# Convert to pandas for compatibility with scikit-learn
X = X.compute()
y = y.compute()

print(X.shape)
print(X.columns)

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Check for categorical columns like 'meteo_if_snow'
if 'meteo_if_snow' in X.columns and X['meteo_if_snow'].dtype == 'object':
    categorical_cols.append('meteo_if_snow')
    numerical_cols.remove('meteo_if_snow')

# Handle missing values and encoding
numerical_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Replace GradientBoostingClassifier with XGBClassifier (GPU version)
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor',
                                 n_estimators=100, random_state=42))
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
print(importances)

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
