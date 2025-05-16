import os
import geopandas as gpd
import folium
import pandas as pd

# Directory containing the Shapefiles
shapefile_dir = 'C:/Users/bentl/Desktop/FINAL/REFACTOR/RDI-Python/src/app/map_tiles/planet_121.431_13.762_8ec95a08.osm.shp/planet_121.431_13.762_8ec95a08-shp/shape'

# List all .shp files in the directory
shapefiles = [f for f in os.listdir(shapefile_dir) if f.endswith('.shp')]

if not shapefiles:
    print("No Shapefiles found in the specified directory.")
    exit(1)

# Initialize an empty GeoDataFrame to hold all data
all_data = gpd.GeoDataFrame()

# Read each Shapefile and append to the combined GeoDataFrame
for shapefile in shapefiles:
    shapefile_path = os.path.join(shapefile_dir, shapefile)
    gdf = gpd.read_file(shapefile_path)
    all_data = gpd.GeoDataFrame(pd.concat([all_data, gdf], ignore_index=True))

# Create a map centered at the mean of the geometries
m = folium.Map(location=[all_data.geometry.centroid.y.mean(), all_data.geometry.centroid.x.mean()], zoom_start=10)

# Add each layer to the map
for idx, row in all_data.iterrows():
    folium.GeoJson(row.geometry).add_to(m)

# Save the map to an HTML file
m.save('C:/Users/bentl/Desktop/FINAL/REFACTOR/RDI-Python/src/app/map_tiles/luzon_map.html')