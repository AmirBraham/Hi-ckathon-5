# Import necessary libraries
import pandas as pd
from geopy.distance import geodesic  # For geospatial calculations

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

    


  
