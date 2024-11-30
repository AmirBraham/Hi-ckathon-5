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
            'meteo_amplitude_tn_tx',
            'meteo_temperature_max',
            'meteo_temperature_min',
            'meteo_temperature_avg',
            'meteo_rain_height',
        ]
        # Select relevant columns along with the key
        features = self.data[[self.key] + meteo_important_cols]
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

    


  