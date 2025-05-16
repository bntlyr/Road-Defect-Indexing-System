import os
import logging
import sqlite3
import osmium
from tqdm import tqdm

class OSMDataPreloader:
    """Handles one-time preloading of OSM data into a SQLite database"""
    
    def __init__(self, pbf_path: str, cache_dir: str):
        self.pbf_path = pbf_path
        self.cache_dir = cache_dir
        self.db_path = os.path.join(cache_dir, 'osm_data.db')
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def preload_data(self, force=False):
        """Preload OSM data into SQLite database"""
        if os.path.exists(self.db_path):
            if not force:
                self.logger.info("OSM database already exists. Use --force to reload.")
                return False
            else:
                self.logger.info("Forcing reload of OSM database.")
                os.remove(self.db_path)  # Remove existing database

        try:
            self.logger.info("Starting OSM data preloading...")
            
            # Initialize handler
            handler = OSMHandler()
            
            # Load OSM data with progress bar
            self.logger.info(f"Loading OSM data from {self.pbf_path}")
            handler.apply_file(self.pbf_path, locations=True)
            
            # Create and populate database
            self.logger.info("Creating SQLite database...")
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Create tables
            c.execute('''CREATE TABLE nodes
                        (id INTEGER PRIMARY KEY, lat REAL, lon REAL)''')
            c.execute('''CREATE TABLE ways
                        (id INTEGER PRIMARY KEY, type TEXT, name TEXT)''')
            c.execute('''CREATE TABLE way_nodes
                        (way_id INTEGER, node_id INTEGER, sequence INTEGER,
                         FOREIGN KEY(way_id) REFERENCES ways(id),
                         FOREIGN KEY(node_id) REFERENCES nodes(id))''')
            
            # Create indices for better query performance
            c.execute('CREATE INDEX idx_way_nodes_way_id ON way_nodes(way_id)')
            c.execute('CREATE INDEX idx_way_nodes_node_id ON way_nodes(node_id)')
            c.execute('CREATE INDEX idx_nodes_coords ON nodes(lat, lon)')
            
            # Start a transaction
            conn.execute('PRAGMA synchronous = OFF;')  # Disable synchronous mode
            conn.execute('PRAGMA journal_mode = OFF;')  # Disable journaling
            conn.execute('BEGIN TRANSACTION;')  # Start transaction
            
            # Insert nodes with progress bar
            self.logger.info("Inserting nodes...")
            nodes_data = [(id, lat, lon) for id, (lat, lon) in handler.nodes.items()]
            for batch in tqdm([nodes_data[i:i + 50000] for i in range(0, len(nodes_data), 50000)]):
                c.executemany('INSERT INTO nodes VALUES (?, ?, ?)', batch)
            
            # Insert ways and way_nodes with progress bar
            self.logger.info("Inserting ways and way nodes...")
            ways_data = []
            way_nodes_data = []
            
            for way_id, way_data in handler.ways.items():
                ways_data.append((way_id, way_data['type'], way_data['name']))
                for seq, node_id in enumerate(way_data['nodes']):
                    way_nodes_data.append((way_id, node_id, seq))
                
            # Commit all ways and way_nodes at once
            c.executemany('INSERT INTO ways VALUES (?, ?, ?)', ways_data)
            c.executemany('INSERT INTO way_nodes VALUES (?, ?, ?)', way_nodes_data)
            
            # Commit the transaction
            conn.commit()
            self.logger.info("Data inserted successfully.")
            
            # Save bounds information
            if handler.bounds is None:
                self.logger.error("Bounds were not set. Please check the OSM data.")
                return False
            
            bounds_info = {
                'min_lat': handler.bounds[0],
                'min_lon': handler.bounds[1],
                'max_lat': handler.bounds[2],
                'max_lon': handler.bounds[3]
            }
            
            with open(os.path.join(self.cache_dir, 'bounds_info.json'), 'w') as f:
                import json
                json.dump(bounds_info, f)
            
            conn.close()
            self.logger.info("OSM data preloading completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error preloading OSM data: {e}")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            return False

class OSMHandler(osmium.SimpleHandler):
    """Handler for processing OSM data during preloading"""
    def __init__(self):
        super(OSMHandler, self).__init__()
        self.ways = {}
        self.nodes = {}
        self.bounds = None

    def node(self, n):
        """Process a node"""
        self.nodes[n.id] = (n.location.lat, n.location.lon)

    def way(self, w):
        """Process a way"""
        if 'highway' in w.tags:  # Only store roads and paths
            self.ways[w.id] = {
                'nodes': [n.ref for n in w.nodes],
                'type': w.tags.get('highway', ''),
                'name': w.tags.get('name', '')
            }

    def bounds(self, b):
        """Store the bounds of the map"""
        self.bounds = (b.min_lat, b.min_lon, b.max_lat, b.max_lon)
        logging.info(f"Bounds set to: {self.bounds}")

def main():
    # Hardcoded values
    pbf_path = 'C:/Users/bentl/Desktop/FINAL/REFACTOR/RDI-Python/src/app/map_tiles/philippines-latest.osm.pbf'
    cache_dir = 'C:/Users/bentl/Desktop/FINAL/REFACTOR/RDI-Python/src/app/map_tiles'

    preloader = OSMDataPreloader(pbf_path, cache_dir)
    success = preloader.preload_data(force=True)  # Set force to True to overwrite existing database
    if not success:
        print("Failed to preload OSM data. Check the logs for details.")
        exit(1)

if __name__ == "__main__":
    main() 