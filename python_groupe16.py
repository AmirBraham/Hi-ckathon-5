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
import numpy as np
import logger
from geopy.distance import geodesic  # For geospatial calculations
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


class Features:
    def __init__(self, data, key):
        self.data = data
        self.key = key
    def construct_meteo_features(self):
        """
        Constructs water-related features from the dataset.
        
        Returns:
            pd.DataFrame: Dataset with selected water-related features and `self.key`.
        """
        # Define the columns to use
        meteo_important_cols = [
            'meteo_rain_height',
            'meteo_evapotranspiration_grid',
            'meteo_sunshine_%',
            'meteo_temperature_avg',
            'meteo_temperature_min',
            'meteo_temperature_max',
            'meteo_amplitude_tn_tx',
            'meteo_humidity_avg',
            'meteo_humidity_min',
            'meteo_pressure_avg',
            'meteo_cloudiness',
            'meteo_frost_duration',
            'meteo_wetting_duration',
            'meteo_humidity_duration_below_40%',
            'meteo_humidity_duration_above_80%',
            'meteo_radiation_direct',
            'meteo_radiation',
            'meteo_radiation_IR',
            'meteo_if_snow',
            'meteo_snow_height',
            'meteo_snow_thickness_6h',
            'meteo_snow_thickness_max'
        ]
        # Select relevant columns along with the key
        features = self.data[[self.key] + meteo_important_cols]
        features['meteo_amplitude_tn_tx'] = features['meteo_amplitude_tn_tx'].fillna(features['meteo_temperature_max'] - features['meteo_temperature_min'])
        features['meteo_temperature_min'] = features['meteo_temperature_min'].fillna(features['meteo_temperature_max'] - features['meteo_amplitude_tn_tx'])
        features['meteo_temperature_max'] = features['meteo_temperature_max'].fillna(features['meteo_temperature_min'] + features['meteo_amplitude_tn_tx'])
        features.drop(columns=['meteo_radiation_IR','meteo_cloudiness','meteo_wetting_duration','meteo_amplitude_tn_tx','meteo_snow_thickness_6h'], inplace = True)
        
        # Définir des tranches pour meteo_temperature_max et meteo_temperature_min
        bins_min = [-30, 5, 10, 15, 20, 40]  # Tranches pour la température minimale
        bins_max = [-10, 20, 30, 40, 60]      # Tranches pour la température maximale

        labels_min = ['-30-5', '5-10', '10-15', '15-20', '20-40']
        labels_max = ['-10-20', '20-30', '30-40', '40-60']

        # Ajouter des colonnes de tranches
        features['temp_range_min'] = pd.cut(features['meteo_temperature_min'], bins=bins_min, labels=labels_min, right=False)
        features['temp_range_max'] = pd.cut(features['meteo_temperature_max'], bins=bins_max, labels=labels_max, right=False)

        # Regrouper par les tranches de température minimale et maximale et calculer les moyennes de meteo_temperature_avg
        group_cols_temp = ['temp_range_min', 'temp_range_max']
        temperature_avg = features.groupby(group_cols_temp)['meteo_temperature_avg'].mean()

        # Remplir les NaN de meteo_temperature_avg avec les moyennes calculées pour les tranches correspondantes
        features['meteo_temperature_avg'] = features.apply(
            lambda row: temperature_avg.get(
                (row['temp_range_min'], row['temp_range_max']),
                row['meteo_temperature_avg']
            ) if pd.isna(row['meteo_temperature_avg']) else row['meteo_temperature_avg'],
            axis=1
        )
        # Définir des tranches pour meteo_evapotranspiration_grid
        bins_evaporation = [0, 2, 4, 6, 8, 10]  # Tranches pour meteo_evapotranspiration_grid

        labels_evaporation = ['0-2', '2-4', '4-6', '6-8', '8-10']

        # Ajouter des colonnes de tranches
        features['evapotranspiration_range'] = pd.cut(features['meteo_evapotranspiration_grid'], bins=bins_evaporation, labels=labels_evaporation, right=False)

        # Regrouper par les tranches de evapotranspiration_range et temp_range_max et calculer les moyennes de meteo_radiation
        group_cols_radiation = ['evapotranspiration_range', 'temp_range_max']
        radiation_avg = features.groupby(group_cols_radiation)['meteo_radiation'].mean()

        # Remplir les NaN de meteo_radiation avec les moyennes calculées pour les tranches correspondantes
        features['meteo_radiation'] = features.apply(
            lambda row: radiation_avg.get(
                (row['evapotranspiration_range'], row['temp_range_max']),
                row['meteo_radiation']
            ) if pd.isna(row['meteo_radiation']) else row['meteo_radiation'],
            axis=1
        )
        # Définir des tranches pour meteo_frost_duration et meteo_temperature_max
        bins_frost = [0, 100, 1000, 1500]  # Tranches pour meteo_frost_duration
        labels_frost = ['0-100', '100-1000', '1000-1500']

        # Ajouter des colonnes de tranches
        features['frost_range'] = pd.cut(features['meteo_frost_duration'], bins=bins_frost, labels=labels_frost, right=False)

        # Calculer la proportion de 1 dans chaque groupe
        group_cols_snow = ['frost_range', 'temp_range_max']
        snow_proportion = features.groupby(group_cols_snow)['meteo_if_snow'].mean()

        # Définir un seuil pour classer en 0 ou 1
        threshold = 0.5  # Si la proportion est > 0.5, on assigne 1, sinon 0
        snow_binary = snow_proportion.apply(lambda x: 1 if x > threshold else 0)

        # Remplir les NaN dans meteo_if_snow
        features['meteo_if_snow'] = features.apply(
            lambda row: snow_binary.get(
                (row['frost_range'], row['temp_range_max']),
                row['meteo_if_snow']
            ) if pd.isna(row['meteo_if_snow']) else row['meteo_if_snow'],
            axis=1
        )
        
        # Mettre à zéro les colonnes meteo_snow_thickness_max et meteo_snow_height quand meteo_if_snow est nul
        features.loc[features['meteo_if_snow'] == 0, ['meteo_snow_thickness_max', 'meteo_snow_height']] = 0

        # Définir des tranches pour meteo_radiation
        bins_radiation = [0, 500, 1000, 1500, 3500]  # Tranches pour la radiation infrarouge
        labels_radiation = ['0-500', '500-1000', '1000-1500', '1500-3500']

        # Ajouter une colonne de tranches pour meteo_radiation
        features['radiation_range'] = pd.cut(features['meteo_radiation'], bins=bins_radiation, labels=labels_radiation, right=False)

        # Regrouper par les tranches de radiation infrarouge et calculer les moyennes de meteo_radiation_direct
        group_cols_radiation = ['radiation_range']
        radiation_avg = features.groupby(group_cols_radiation)['meteo_radiation_direct'].mean()

        # Remplir les NaN de meteo_radiation_direct avec les moyennes calculées pour les tranches correspondantes
        features['meteo_radiation_direct'] = features.apply(
            lambda row: radiation_avg.get(row['radiation_range'], row['meteo_radiation_direct'])
            if pd.isna(row['meteo_radiation_direct']) else row['meteo_radiation_direct'],
            axis=1
        )
        # Définir des tranches pour meteo_radiation_direct
        bins_radiation_direct = [0, 700, 1000, 2000, 5000]  # Tranches pour la radiation directe
        labels_radiation_direct = ['0-700', '700-1000', '1000-2000', '2000-5000']

        # Ajouter une colonne de tranches pour meteo_radiation_direct
        features['radiation_direct_range'] = pd.cut(features['meteo_radiation_direct'], bins=bins_radiation_direct, labels=labels_radiation_direct, right=False)

        # Regrouper par les tranches de radiation directe et calculer les moyennes de meteo_sunshine_%
        group_cols_sunshine = ['radiation_direct_range']
        sunshine_avg = features.groupby(group_cols_sunshine)['meteo_sunshine_%'].mean()

        # Remplir les NaN de meteo_sunshine_% avec les moyennes calculées pour les tranches correspondantes
        features['meteo_sunshine_%'] = features.apply(
            lambda row: sunshine_avg.get(row['radiation_direct_range'], row['meteo_sunshine_%'])
            if pd.isna(row['meteo_sunshine_%']) else row['meteo_sunshine_%'],
            axis=1
        )

        # Définir des tranches pour meteo_rain_height
        bins_rain_height = [0, 0.2, 1, 250]  # Tranches pour la hauteur de pluie
        labels_rain_height = ['0-0.2', '0.2-1', '1-250']

        # Ajouter une colonne de tranches pour meteo_rain_height
        features['rain_height_range'] = pd.cut(features['meteo_rain_height'], bins=bins_rain_height, labels=labels_rain_height, right=False)

        # Regrouper par les tranches de rain_height et calculer les moyennes de meteo_pressure_avg
        group_cols_pressure = ['rain_height_range']
        pressure_avg = features.groupby(group_cols_pressure)['meteo_pressure_avg'].mean()

        # Remplir les NaN de meteo_pressure_avg avec les moyennes calculées pour les tranches correspondantes
        features['meteo_pressure_avg'] = features.apply(
            lambda row: pressure_avg.get(row['rain_height_range'], row['meteo_pressure_avg'])
            if pd.isna(row['meteo_pressure_avg']) else row['meteo_pressure_avg'],
            axis=1
        )
        
        # Regrouper par les tranches de radiation et calculer les moyennes
        group_cols_humidity = ['radiation_range']

        # Calcul des moyennes pour chaque variable d'humidité
        humidity_min_avg = features.groupby(group_cols_humidity)['meteo_humidity_min'].mean()
        humidity_duration_below_40_avg = features.groupby(group_cols_humidity)['meteo_humidity_duration_below_40%'].mean()
        humidity_duration_above_80_avg = features.groupby(group_cols_humidity)['meteo_humidity_duration_above_80%'].mean()
        humidity_avg_avg = features.groupby(group_cols_humidity)['meteo_humidity_avg'].mean()

        # Remplir les NaN pour meteo_humidity_min
        features['meteo_humidity_min'] = features.apply(
            lambda row: humidity_min_avg.get(row['radiation_range'], row['meteo_humidity_min'])
            if pd.isna(row['meteo_humidity_min']) else row['meteo_humidity_min'],
            axis=1
        )

        # Remplir les NaN pour meteo_humidity_duration_below_40%
        features['meteo_humidity_duration_below_40%'] = features.apply(
            lambda row: humidity_duration_below_40_avg.get(row['radiation_range'], row['meteo_humidity_duration_below_40%'])
            if pd.isna(row['meteo_humidity_duration_below_40%']) else row['meteo_humidity_duration_below_40%'],
            axis=1
        )

        # Remplir les NaN pour meteo_humidity_duration_above_80%
        features['meteo_humidity_duration_above_80%'] = features.apply(
            lambda row: humidity_duration_above_80_avg.get(row['radiation_range'], row['meteo_humidity_duration_above_80%'])
            if pd.isna(row['meteo_humidity_duration_above_80%']) else row['meteo_humidity_duration_above_80%'],
            axis=1
        )

        # Remplir les NaN pour meteo_humidity_avg
        features['meteo_humidity_avg'] = features.apply(
            lambda row: humidity_avg_avg.get(row['radiation_range'], row['meteo_humidity_avg'])
            if pd.isna(row['meteo_humidity_avg']) else row['meteo_humidity_avg'],
            axis=1
        )
        features.drop(columns=['rain_height_range','radiation_direct_range','radiation_range','frost_range','evapotranspiration_range','temp_range_min','temp_range_max'], inplace = True)
        return features

    def construct_piezo_features(self):
        """
        Constructs piezo-related features from the dataset.
        
        Returns:
            pd.DataFrame: Dataset with selected water-related features and `self.key`.
        """
        # Define the columns to use
        piezo_important_cols = [
            'piezo_station_investigation_depth',
            'piezo_station_altitude',
            'piezo_station_longitude',
            'piezo_station_latitude',
            # 'piezo_measurement_date',
            'piezo_obtention_mode',
        ]
        # Select relevant columns along with the key
        features = self.data[[self.key] + piezo_important_cols]
        return features



    def construct_withdrawal_features(self, obs_latitude=None, obs_longitude=None):
        """
        Constructs relevant and derived features for the 'Prélèvement (Withdrawals)' category.
        
        Parameters:
            obs_latitude (float, optional): Latitude of the observation point for distance calculation.
            obs_longitude (float, optional): Longitude of the observation point for distance calculation.
        
        Returns:
            pd.DataFrame: Dataset with constructed features and `self.key`.
        """
        # Volume-Related Aggregations
        self.data['total_volume'] = (
            self.data['prelev_volume_0'] + 
            self.data['prelev_volume_1'] + 
            self.data['prelev_volume_2'] + 
            self.data['prelev_other_volume_sum']
        )
        self.data['largest_ratio'] = self.data['prelev_volume_0'] / self.data['total_volume']
        self.data['volume_mean'] = (
            self.data['prelev_volume_0'] + 
            self.data['prelev_volume_1'] + 
            self.data['prelev_volume_2']
        ) / 3
        self.data['volume_variance'] = self.data[
            ['prelev_volume_0', 'prelev_volume_1', 'prelev_volume_2']
        ].var(axis=1)
        
        # Categorical Encodings for Usage
        usage_labels = ['prelev_usage_label_0', 'prelev_usage_label_1', 'prelev_usage_label_2']
        usage_dummies = pd.get_dummies(self.data[usage_labels].stack(), prefix="usage").groupby(level=0).sum()
        usage_distribution = pd.DataFrame({
            "drinking_water_count": self.data[usage_labels].apply(lambda row: sum(row == "Eau potable"), axis=1),
            "irrigation_count": self.data[usage_labels].apply(lambda row: sum(row == "Irrigation"), axis=1),
            "industry_count": self.data[usage_labels].apply(lambda row: sum(row == "Industrie et activités économiques"), axis=1),
            "energy_count": self.data[usage_labels].apply(lambda row: sum(row == "Energie"), axis=1),
            "channels_count": self.data[usage_labels].apply(lambda row: sum(row == "Canaux"), axis=1),
            "turbined_water_count": self.data[usage_labels].apply(lambda row: sum(row == "Eau turbinée (barrage)"), axis=1),
        })
        usage_features = pd.concat([usage_distribution, usage_dummies], axis=1)
        
        # Geospatial Features
        self.data['mean_longitude'] = (
            self.data['prelev_longitude_0'] + 
            self.data['prelev_longitude_1'] + 
            self.data['prelev_longitude_2']
        ) / 3
        self.data['mean_latitude'] = (
            self.data['prelev_latitude_0'] + 
            self.data['prelev_latitude_1'] + 
            self.data['prelev_latitude_2']
        ) / 3
        
        if obs_latitude is not None and obs_longitude is not None:
            obs_coords = (obs_latitude, obs_longitude)
            distances = self.data.apply(lambda row: [
                geodesic(obs_coords, (row['prelev_latitude_0'], row['prelev_longitude_0'])).km,
                geodesic(obs_coords, (row['prelev_latitude_1'], row['prelev_longitude_1'])).km,
                geodesic(obs_coords, (row['prelev_latitude_2'], row['prelev_longitude_2'])).km
            ], axis=1)
            self.data['min_distance'] = distances.apply(min)
        
        # Quality of Measurements
        measurement_labels = ['prelev_volume_obtention_mode_label_0', 
                              'prelev_volume_obtention_mode_label_1', 
                              'prelev_volume_obtention_mode_label_2']
        self.data['measurement_type_ratio'] = self.data[measurement_labels].apply(
            lambda row: sum(row == "Specific Type") / 3, axis=1
        )
        
        # Administrative and Socioeconomic Factors
        insee_labels = ['prelev_commune_code_insee_0', 'prelev_commune_code_insee_1', 'prelev_commune_code_insee_2']
        insee_dummies = pd.get_dummies(self.data[insee_labels].stack(), prefix="insee").groupby(level=0).sum()
        
        # Combine all features
        constructed_features = pd.concat([
            self.data[[self.key, 'total_volume', 'largest_ratio', 'volume_mean', 'volume_variance', 
                       'mean_longitude', 'mean_latitude', 'measurement_type_ratio']],
            usage_features,
            # insee_dummies
        ], axis=1)
        
        # Include min_distance if calculated
        if 'min_distance' in self.data.columns:
            constructed_features['min_distance'] = self.data['min_distance']
        
        return constructed_features

    def construct_insee_features(self):
        """
        Constructs economic-related features from the dataset.
        
        Returns:
            pd.DataFrame: Dataset with selected water-related features and `self.key`.
        """
        # Define the columns to use
        insee_important_cols = [
            'insee_%_agri',
            'insee_pop_commune',
            'insee_med_living_level',
            'insee_%_ind',
            'insee_%_const',
        ]
        # Select relevant columns along with the key
        features = self.data[[self.key] + insee_important_cols]
        colonnes_object = [
            'insee_%_agri',
            'insee_med_living_level',
            'insee_%_ind',
            'insee_%_const',
        ]
        # Convertir ces colonnes en float
        for col in colonnes_object:
            features[col] = pd.to_numeric(features[col], errors='coerce')
        
        return features
    
    def construct_distance_features(self):
        """
        Constructs distance-related features from the dataset.
        
        Returns:
            pd.DataFrame: Dataset with selected water-related features and `self.key`.
        """
        distance_important_cols = [
            'distance_piezo_hydro',
            'distance_piezo_meteo'
        ]
        # Select relevant columns along with the key
        features = self.data[[self.key] + distance_important_cols]
        return features
    
    
    def construct_seasonality_features(self):
        """
        Adds a seasonality feature to the data by calculating the cosine of the month
        derived from the `piezo_station_update_date` column.
        
        Assumes `piezo_station_update_date` is a datetime column or can be parsed as datetime.
        """
        features = pd.DataFrame(columns=[self.key,'month'])
        features[self.key] = self.data[self.key]
        seasonal_feat = pd.DataFrame()
          # Convert 'piezo_measurement_date' to datetime
        self.data['piezo_measurement_date'] = pd.to_datetime(self.data['piezo_measurement_date'], format='%Y-%m-%d', errors='coerce')
        
        # Extract the month
        features['month'] = self.data['piezo_measurement_date'].dt.month
        
        # Optionally, create the cosine of the month as a seasonality feature
        features['month_cosine'] = np.cos(2 * np.pi * (features['month'] - 1) / 12)

        return features

    
# Define the key and target column
key = "row_index"
target_column = 'piezo_groundwater_level_category'

# Paths for the raw data and the preprocessed data
raw_file_path = 'X_train_Hi5.csv'
processed_data_path = 'preprocessed_data.csv'  # File to save/load preprocessed data
processed_target_path = 'preprocessed_target.csv'  # File to save/load target variable
model_path = 'trained_model.pkl'  # File to save/load the trained model
label_mapping_path = 'label_mapping.pkl'  # File to save/load label mappings

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