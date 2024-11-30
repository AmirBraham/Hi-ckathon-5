from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class Seasons:
    WINTER = [12, 1, 2]
    SPRING = [3, 4, 5]
    SUMMER = [6, 7, 8]
    AUTUMN = [9, 10, 11]

class Filters:
    def __init__(self, data):
        self.data = data

    def filter_on_seasons(self, seasons=Seasons.SUMMER):
        # Ensure 'piezo_station_update_date' is properly handled
        self.data['Month'] = self.data['piezo_station_update_date'].apply(
            lambda x: datetime.strptime(x.split()[1], "%b").month
        )
        # Filter data based on the specified seasons
        filtered_data = self.data[self.data['Month'].isin(seasons)]
        # Drop the temporary 'Month' column
        return filtered_data.drop(columns=['Month'])
