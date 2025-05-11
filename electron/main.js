// Electron main.js used for packaging the application into an executable file for the user to download and use

const { app, BrowserWindow, Menu } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const isDev = process.env.NODE_ENV === 'development';

let mainWindow;
let backendProcess;
let frontendProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    frame: true,
    titleBarStyle: 'default',
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  // Remove the menu bar
  Menu.setApplicationMenu(null);

  // Load the frontend URL
  const frontendUrl = isDev ? 'http://localhost:3000' : `file://${path.join(__dirname, '../frontend/out/index.html')}`;
  mainWindow.loadURL(frontendUrl);

  // Only open DevTools if explicitly requested in development
  if (isDev && process.argv.includes('--debug')) {
    mainWindow.webContents.openDevTools();
  }
}

function startBackend() {
  const pythonPath = isDev ? 'python' : path.join(process.resourcesPath, 'python/python.exe');
  const scriptPath = isDev ? '../backend/app.py' : path.join(process.resourcesPath, 'backend/app.py');
  
  backendProcess = spawn(pythonPath, [scriptPath], {
    stdio: 'inherit'
  });

  backendProcess.on('error', (err) => {
    console.error('Failed to start backend:', err);
  });
}

function startFrontend() {
  if (isDev) {
    frontendProcess = spawn('npm', ['run', 'dev'], {
      cwd: path.join(__dirname, '../frontend'),
      shell: true
    });

    frontendProcess.on('error', (err) => {
      console.error('Failed to start frontend:', err);
    });
  }
}

app.whenReady().then(() => {
  startBackend();
  startFrontend();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    if (backendProcess) backendProcess.kill();
    if (frontendProcess) frontendProcess.kill();
    app.quit();
  }
});

app.on('before-quit', () => {
  if (backendProcess) backendProcess.kill();
  if (frontendProcess) frontendProcess.kill();
});


