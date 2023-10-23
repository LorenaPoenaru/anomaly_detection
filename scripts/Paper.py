#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 19:15:49 2023

@author: home
"""
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

#'path_to_shapefile' 
world_map = gpd.read_file('/Users/home/Documents/Master/ne_110m_admin_0_countries_lakes')

#'path_to_dataset'
df = pd.read_csv('/Users/home/Documents/Master/DOT.csv')

# Preprocess the dataset to match country names with the shapefile
df['Country Name'] = df['Country Name'].str.strip()  # Remove leading/trailing whitespace
df = df.dropna(subset=['Country Name'])  # Remove rows with missing country names

# Merge the dataset with the shapefile
merged_data = world_map.merge(df, left_on='NAME', right_on='Country Name', how='left')

import matplotlib.pyplot as plt

# Plot the export values on the map
#Modify the code to filter the rows based on the "Indicator name" column containing "Values of Exports":
export_data = merged_data[merged_data['Indicator Name'].str.contains('TXG_FOB_USD', case=False, na=False)]

# Replace missing values with a specific value (e.g., 0) in the '2019' column
export_data['2019'].fillna(0, inplace=True)

# Convert the '2019' column to numeric type
export_data['2019'] = pd.to_numeric(export_data['2019'], errors='coerce')

# Plot the export values on the map
export_data.plot(column='2019', cmap='Blues', linewidth=0.8, edgecolor='0.8', figsize=(12, 8), aspect='equal')

#Visualization
plt.title('Merchandise Exports by Country')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.axis('off')

#Add the legend
sm = plt.cm.ScalarMappable(cmap='Blues')
sm.set_array(export_data['2019'])
plt.colorbar(sm)

#Showing
plt.show()


# Filter the rows based on the "Indicator Name" column containing "TMG_FOB_USD"
import_data = merged_data[merged_data['Indicator Name'].str.contains('TMG_FOB_USD', case=False, na=False)]

# Replace missing values with a specific value (e.g., 0) in the '2019' column
import_data['2019'].fillna(0, inplace=True)

# Convert the '2019' column to numeric type
import_data['2019'] = pd.to_numeric(import_data['2019'], errors='coerce')

# Plot the import values on the map
import_data.plot(column='2019', cmap='Reds', linewidth=0.8, edgecolor='0.8', figsize=(12, 8), aspect='equal')

plt.title('Merchandise Imports by Country')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.axis('off')

sm = plt.cm.ScalarMappable(cmap='Reds')
sm.set_array(export_data['2019'])
plt.colorbar(sm)
plt.show()

print(df.columns)


# Load the dataset
df = pd.read_csv(r'C:/Users/Daniil/Downloads/DOT.csv')





print(df.columns)