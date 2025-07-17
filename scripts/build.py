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

def get_hidden_imports():
    """Get all necessary hidden imports"""
    return [
        # Core Python modules
        'os',
        'sys',
        'time',
        'datetime',
        'json',
        'queue',
        'threading',
        'logging',
        'pathlib',
        
        # PyQt5 and related
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt5.sip',
        'PyQt5.QtMultimedia',
        'PyQt5.QtMultimediaWidgets',
        
        # OpenCV and image processing
        'cv2',
        'cv2.cv2',
        'cv2.dnn',
        'cv2.data',
        'cv2.gapi',
        'cv2.utils',
        'cv2.ml',
        'cv2.ocl',
        'cv2.videoio',
        'cv2.videoio_registry',
        'cv2.cuda',
        'numpy',
        'PIL',
        'PIL.Image',
        'PIL.ImageQt',
        'PIL.ImageDraw',
        'PIL.ImageFont',
        
        # Machine Learning
        'torch',
        'torchvision',
        'torchaudio',
        'torch.nn',
        'torch.nn.functional',
        'torch.utils.data',
        'torchvision.transforms',
        
        # GPS and Serial
        'serial',
        'serial.tools',
        'serial.tools.list_ports',
        'pynmea2',
        
        # Additional utilities
        'psutil',
        'requests',
        'urllib3',
        'certifi',
        'chardet',
        'idna',
        
        # Application specific imports
        'src',
        'src.ui',
        'src.modules',
        'src.ui.dashboard',
        'src.ui.video_controls',
        'src.ui.main_controls',
        'src.ui.statistics',
        'src.ui.status_bar',
        'src.ui.settings_manager',
        'src.modules.camera',
        'src.modules.gps_reader',
        'src.modules.detection',
    ]

def get_collect_all_packages():
    """Get all packages that need to be collected"""
    return [
        'torch',
        'torchvision',
        'torchaudio',
        'opencv-python',
        'opencv-contrib-python',
        'PyQt5',
        'PIL',
        'numpy',
        'pynmea2',
        'serial',
    ]

def create_spec_file():
    """Create a custom spec file for better package handling"""
    # Convert icon to ICO format first
    icon_path = convert_icon_to_ico()
    icon_arg = f"icon='{icon_path}'," if icon_path else ""
    
    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src/models', 'models'),
        ('public', 'public'),
        ('src/ui', 'ui'),
        ('src/modules', 'modules'),
    ],
    hiddenimports=[
        'src',
        'src.ui',
        'src.modules',
        'src.ui.dashboard',
        'src.ui.video_controls',
        'src.ui.main_controls',
        'src.ui.statistics',
        'src.ui.status_bar',
        'src.ui.settings_manager',
        'src.modules.camera',
        'src.modules.gps_reader',
        'src.modules.detection',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='RoadDefectSystem',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    {icon_arg}
)
"""
    with open('RoadDefectSystem.spec', 'w') as f:
        f.write(spec_content)

def build_exe():
    """Build the executable using PyInstaller"""
    try:
        import PyInstaller.__main__
        
        # Create custom spec file
        create_spec_file()
        
        # Prepare PyInstaller arguments
        args = [
            'RoadDefectSystem.spec',  # Use our custom spec file
            '--clean',  # Clean PyInstaller cache
            '--noconfirm',  # Replace existing spec file
        ]
        
        print("Starting build with PyInstaller...")
        print("Arguments:", ' '.join(args))
        
        # Run PyInstaller
        PyInstaller.__main__.run(args)
        
        print("\nBuild completed successfully!")
        print("Executable can be found in the 'dist' directory")
        
        # Create a directory for additional resources
        dist_dir = os.path.join('dist', 'RoadDefectSystem')
        if not os.path.exists(dist_dir):
            os.makedirs(dist_dir)
        
        # Copy additional required files
        data_dirs = [
            ('src/models', 'models'),
            ('public', 'public'),
            ('src/ui', 'ui'),
            ('src/modules', 'modules'),
        ]
        
        for src, dst in data_dirs:
            src_path = os.path.join(os.getcwd(), src)
            dst_path = os.path.join(dist_dir, dst)
            if os.path.exists(src_path):
                print(f"Copying {src} to {dst_path}")
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        
        print("\nAdditional resources have been copied to the dist directory")
        
    except Exception as e:
        print(f"Error building executable: {e}")
        raise

def check_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        'pyinstaller',
        'pillow',
        'opencv-python',
        'PyQt5',
        'numpy',
        'torch',
        'torchvision',
        'pynmea2',
        'pyserial',
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"pip install {package}")

if __name__ == "__main__":
    # Check and install dependencies
    check_dependencies()
    
    # Create build directory if it doesn't exist
    if not os.path.exists('build'):
        os.makedirs('build')
    
    # Build the executable
    build_exe() 