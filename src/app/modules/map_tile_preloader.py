import os
import math
import logging
import requests
import time
from typing import Tuple, Dict
from dataclasses import dataclass
import json
from PIL import Image
from io import BytesIO
from tqdm import tqdm

@dataclass
class TileBounds:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    zoom: int

class MapTilePreloader:
    """Handles preloading and caching of map tiles for offline use"""
    
    # Philippines/Luzon bounds
    LUZON_BOUNDS = TileBounds(
        min_lat=12.0,    # Southern Luzon
        max_lat=18.5,    # Northern Luzon
        min_lon=119.5,   # Western Luzon
        max_lon=124.5,   # Eastern Luzon
        zoom=17          # Street level only
    )

    def __init__(self, cache_dir: str = None, batch_delay: float = 1.0, use_proxy: bool = False, proxy_url: str = None):
        """Initialize the preloader with cache directory and settings"""
        if cache_dir is None:
            # Default to src/map_tiles
            self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'map_tiles')
        else:
            self.cache_dir = cache_dir
        
        # Create cache directory structure
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize session with proper headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RoadDefectInspector/1.0',
            'Accept': 'image/png',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        
        # Optional proxy support
        if use_proxy and proxy_url:
            self.session.proxies.update({
                'http': proxy_url,
                'https': proxy_url
            })
        
        # Settings
        self.batch_delay = batch_delay
        
        # Statistics
        self.downloaded_tiles = 0
        self.compressed_tiles = 0
        self.original_size = 0
        self.compressed_size = 0
        
        # Save bounds info
        self._save_bounds_info()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.StreamHandler()]
        )

    def _save_bounds_info(self):
        """Save the bounds information to a JSON file"""
        bounds_info = {
            'min_lat': self.LUZON_BOUNDS.min_lat,
            'max_lat': self.LUZON_BOUNDS.max_lat,
            'min_lon': self.LUZON_BOUNDS.min_lon,
            'max_lon': self.LUZON_BOUNDS.max_lon,
            'zoom': self.LUZON_BOUNDS.zoom,
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(self.cache_dir, 'bounds_info.json'), 'w') as f:
            json.dump(bounds_info, f, indent=2)

    def lat_lon_to_tile(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convert latitude/longitude to tile coordinates"""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    def get_tile_range(self) -> Tuple[int, int, int, int]:
        """Calculate the tile range for Luzon at street level"""
        x_start, y_start = self.lat_lon_to_tile(
            self.LUZON_BOUNDS.max_lat,  # Note: max_lat for min_y
            self.LUZON_BOUNDS.min_lon,
            self.LUZON_BOUNDS.zoom
        )
        x_end, y_end = self.lat_lon_to_tile(
            self.LUZON_BOUNDS.min_lat,  # Note: min_lat for max_y
            self.LUZON_BOUNDS.max_lon,
            self.LUZON_BOUNDS.zoom
        )
        return x_start, y_start, x_end, y_end

    def compress_tile(self, image_data: bytes) -> bytes:
        """Compress a tile image using PIL/Pillow"""
        try:
            # Open image from bytes
            img = Image.open(BytesIO(image_data))
            
            # Convert to RGB if needed (some tiles might be RGBA)
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            
            # Save to bytes with compression
            output = BytesIO()
            img.save(output, format='PNG', optimize=True)
            return output.getvalue()
        except Exception as e:
            logging.error(f"Error compressing tile: {e}")
            return image_data  # Return original if compression fails

    def download_tile(self, x: int, y: int, max_retries: int = 3) -> bool:
        """Download a single tile with retry logic and compression"""
        tile_path = os.path.join(self.cache_dir, str(self.LUZON_BOUNDS.zoom), str(x), f"{y}.png")
        
        # Skip if already downloaded
        if os.path.exists(tile_path):
            self.downloaded_tiles += 1
            return True

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(tile_path), exist_ok=True)
        
        # Download tile with retries
        for attempt in range(max_retries):
            try:
                url = f"https://tile.openstreetmap.org/{self.LUZON_BOUNDS.zoom}/{x}/{y}.png"
                
                # Exponential backoff for rate limiting
                if attempt > 0:
                    time.sleep((2 ** attempt) * 0.5)  # 0.5s, 1s, 2s
                else:
                    time.sleep(0.2)  # Base delay
                
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                # Verify we got a valid image
                if not response.headers.get('content-type', '').startswith('image/'):
                    raise ValueError(f"Invalid content type: {response.headers.get('content-type')}")
                
                # Get original size and compress
                original_data = response.content
                self.original_size += len(original_data)
                compressed_data = self.compress_tile(original_data)
                self.compressed_size += len(compressed_data)
                
                # Save compressed tile
                with open(tile_path, 'wb') as f:
                    f.write(compressed_data)
                
                self.downloaded_tiles += 1
                self.compressed_tiles += 1
                return True
                
            except (requests.exceptions.RequestException, ValueError) as e:
                logging.warning(f"[{self.LUZON_BOUNDS.zoom}/{x}/{y}] Attempt {attempt + 1} failed: {e}")
                continue
            except Exception as e:
                logging.error(f"[{self.LUZON_BOUNDS.zoom}/{x}/{y}] Unexpected error: {e}", exc_info=True)
                break
        
        return False

    def preload_tiles(self):
        """Preload street-level tiles for Luzon with improved rate limiting"""
        # Calculate tile range
        x_start, y_start, x_end, y_end = self.get_tile_range()
        total_tiles = (x_end - x_start + 1) * (y_end - y_start + 1)
        
        logging.info(f"Starting download of {total_tiles} street-level tiles...")
        start_time = time.time()
        
        # Create progress bar
        pbar = tqdm(total=total_tiles, desc="Downloading tiles")
        
        # Download tiles with rate limiting
        for x in range(x_start, x_end + 1):
            for y in range(y_start, y_end + 1):
                success = self.download_tile(x, y)
                status = "✓" if success else "✗"
                logging.debug(f"Tile ({x}, {y}) - {status}")
                
                pbar.update(1)
                time.sleep(self.batch_delay)  # Rate-limiting delay
        
        pbar.close()
        
        # Log final statistics
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Download completed in {duration:.2f}s")
        logging.info(f"Downloaded tiles: {self.downloaded_tiles}/{total_tiles}")
        logging.info(f"Compressed tiles: {self.compressed_tiles}")
        logging.info(f"Original size: {self.original_size / (1024*1024):.2f} MB")
        logging.info(f"Compressed size: {self.compressed_size / (1024*1024):.2f} MB")
        
        if self.original_size > 0:
            reduction = 100 - (self.compressed_size / self.original_size * 100)
            logging.info(f"Size reduction: {reduction:.2f}%")

    def get_cache_stats(self) -> Dict:
        """Get statistics about the cached tiles"""
        stats = {
            'total_tiles': 0,
            'cache_size': 0,
            'compressed_size': 0,
            'compression_ratio': 0,
            'last_updated': None
        }
        
        # Read bounds info
        bounds_file = os.path.join(self.cache_dir, 'bounds_info.json')
        if os.path.exists(bounds_file):
            with open(bounds_file, 'r') as f:
                bounds_info = json.load(f)
                stats['last_updated'] = bounds_info.get('last_updated')
        
        # Count tiles and calculate size
        zoom_dir = os.path.join(self.cache_dir, str(self.LUZON_BOUNDS.zoom))
        if os.path.exists(zoom_dir):
            for x_dir in os.listdir(zoom_dir):
                x_path = os.path.join(zoom_dir, x_dir)
                if not os.path.isdir(x_path):
                    continue
                    
                for tile_file in os.listdir(x_path):
                    if tile_file.endswith('.png'):
                        stats['total_tiles'] += 1
                        file_path = os.path.join(x_path, tile_file)
                        stats['cache_size'] += os.path.getsize(file_path)
        
        # Convert sizes to MB
        stats['cache_size'] = round(stats['cache_size'] / (1024 * 1024), 2)
        
        # Calculate compression ratio if we have both sizes
        if self.original_size > 0:
            stats['compressed_size'] = round(self.compressed_size / (1024 * 1024), 2)
            stats['compression_ratio'] = round((1 - (self.compressed_size / self.original_size)) * 100, 1)
        
        return stats

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create preloader
    preloader = MapTilePreloader(
        batch_delay=1.0,  # 1 second delay between tiles
        use_proxy=False   # Set to True and provide proxy_url if needed
    )
    
    # Print initial cache stats
    print("\nInitial cache statistics:")
    print(json.dumps(preloader.get_cache_stats(), indent=2))
    
    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Download street-level tiles")
    print("2. View cache statistics only")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        # Preload tiles
        print("\nStarting tile download...")
        preloader.preload_tiles()
    elif choice == "2":
        pass  # Just show stats
    else:
        print("Invalid choice")
        exit(1)
    
    # Print final cache stats
    print("\nFinal cache statistics:")
    print(json.dumps(preloader.get_cache_stats(), indent=2)) 