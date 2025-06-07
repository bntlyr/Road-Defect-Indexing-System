import os
import sys
import shutil
from pathlib import Path

def convert_icon_to_ico():
    """Convert PNG icon to ICO format"""
    try:
        from PIL import Image
        icon_path = os.path.join('public', 'icons', 'icon.png')
        ico_path = os.path.join('public', 'icons', 'icon.ico')
        
        # Open the PNG image
        img = Image.open(icon_path)
        
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Save as ICO
        img.save(ico_path, format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)])
        print("Icon converted successfully!")
        return ico_path
    except Exception as e:
        print(f"Error converting icon: {e}")
        return None

def build_exe():
    """Build the executable using PyInstaller"""
    try:
        import PyInstaller.__main__
        
        # Convert icon to ICO format
        icon_path = convert_icon_to_ico()
        if not icon_path:
            print("Using default icon...")
            icon_path = ""
        
        # Define PyInstaller arguments
        args = [
            'src/app.py',  # Main script
            '--name=RoadDefectSystem',  # Name of the executable
            '--onefile',  # Create a single executable
            '--windowed',  # Don't show console window
            '--clean',  # Clean PyInstaller cache
            '--add-data=src/models;models',  # Include models directory
            '--add-data=public;public',  # Include public directory
            '--exclude-module=PyQt5',  # Exclude PyQt5
            '--exclude-module=PyQt5.QtCore',
            '--exclude-module=PyQt5.QtGui',
            '--exclude-module=PyQt5.QtWidgets',
            '--hidden-import=cv2',
            '--hidden-import=numpy',
            '--hidden-import=torch',
            '--hidden-import=PIL',
            '--hidden-import=PIL.Image',
            '--hidden-import=PIL.ImageQt',
        ]
        
        # Add icon if available
        if icon_path:
            args.append(f'--icon={icon_path}')
        
        # Run PyInstaller
        PyInstaller.__main__.run(args)
        
        print("\nBuild completed successfully!")
        print("Executable can be found in the 'dist' directory")
        
    except Exception as e:
        print(f"Error building executable: {e}")

if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import PyInstaller
        from PIL import Image
    except ImportError:
        print("Installing required packages...")
        os.system("pip install pyinstaller pillow")
    
    # Create build directory if it doesn't exist
    if not os.path.exists('build'):
        os.makedirs('build')
    
    # Build the executable
    build_exe()
