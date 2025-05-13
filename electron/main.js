const { app, BrowserWindow, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const waitOn = require('wait-on');
const isDev = process.env.NODE_ENV === 'development';
const fs = require('fs');
const axios = require('axios');

let mainWindow;
let backendProcess;
let frontendProcess;
let isBackendReady = false;
let isFrontendReady = false;
let backendStartAttempts = 0;
const MAX_BACKEND_START_ATTEMPTS = 3;
const BACKEND_START_TIMEOUT = 30000; // 30 seconds timeout
const FRONTEND_START_TIMEOUT = 30000; // 30 seconds timeout
const FRONTEND_PORT = 3000;
const BACKEND_PORT = 5000;

function findPythonExecutable() {
  // In production, Python will be in the resources directory
  if (!isDev) {
    const prodPythonPath = path.join(process.resourcesPath, 'backend', 'venv', 'Scripts', 'python.exe');
    if (fs.existsSync(prodPythonPath)) {
      console.log('Found Python in production environment:', prodPythonPath);
      return prodPythonPath;
    }
  }

  // In development, try multiple possible locations
  const possiblePaths = [
    // Try venv in project root (3 levels up from electron)
    path.join(__dirname, '..', '..', 'venv', 'Scripts', 'python.exe'),
    // Try venv in backend directory
    path.join(__dirname, '..', 'backend', 'venv', 'Scripts', 'python.exe'),
    // Try venv in current directory
    path.join(__dirname, 'venv', 'Scripts', 'python.exe')
  ];

  for (const pythonPath of possiblePaths) {
    if (fs.existsSync(pythonPath)) {
      console.log('Found Python in development environment:', pythonPath);
      return pythonPath;
    }
  }

  // If no virtual environment found, try to use system Python
  console.log('Virtual environment Python not found, checking system Python...');
  try {
    // Try to run 'python --version' to check if Python is available
    const { execSync } = require('child_process');
    execSync('python --version');
    console.log('Using system Python');
    return 'python';
  } catch (error) {
    console.error('System Python not found. Please ensure Python is installed and available in PATH');
    throw new Error('Python executable not found. Please ensure Python is installed and a virtual environment exists.');
  }
}

async function startBackend() {
  const backendPath = isDev 
    ? path.join(__dirname, '../backend/main.py')
    : path.join(process.resourcesPath, 'backend/main.py');

  const pythonPath = findPythonExecutable();

  console.log('Starting backend with Python:', pythonPath);
  console.log('Backend path:', backendPath);

  if (!fs.existsSync(backendPath)) {
    throw new Error(`Backend script not found at: ${backendPath}`);
  }

  const env = {
    ...process.env,
    PYTHONUNBUFFERED: '1',
    PYTHONPATH: isDev 
      ? path.join(__dirname, '../backend')
      : path.join(process.resourcesPath, 'backend'),
    FLASK_ENV: 'development',
    FLASK_DEBUG: '0'  // Disable Flask debug mode to prevent reloader
  };

  if (pythonPath.includes('venv')) {
    const venvPath = path.dirname(pythonPath);
    env.PATH = `${venvPath}${path.delimiter}${env.PATH}`;
  }

  // Kill any existing backend process
  if (backendProcess) {
    try {
      backendProcess.kill();
      await new Promise(resolve => setTimeout(resolve, 1000)); // Wait for process to fully terminate
    } catch (error) {
      console.warn('Failed to kill existing backend process:', error);
    }
  }

  return new Promise((resolve, reject) => {
    let startupTimeout;
    let healthCheckInterval;

    const cleanup = () => {
      if (startupTimeout) clearTimeout(startupTimeout);
      if (healthCheckInterval) clearInterval(healthCheckInterval);
    };

    startupTimeout = setTimeout(() => {
      cleanup();
      if (backendProcess) {
        backendProcess.kill();
      }
      reject(new Error('Backend startup timed out'));
    }, BACKEND_START_TIMEOUT);

    backendProcess = spawn(pythonPath, [backendPath], {
      stdio: 'inherit',
      env,
      shell: true
    });

    backendProcess.on('error', (error) => {
      cleanup();
      console.error('Failed to start backend:', error);
      reject(error);
    });

    backendProcess.on('exit', (code) => {
      cleanup();
      if (code !== 0) {
        console.error(`Backend process exited with code ${code}`);
        isBackendReady = false;
        reject(new Error(`Backend process exited with code ${code}`));
      }
    });

    // Check backend health periodically
    healthCheckInterval = setInterval(async () => {
      try {
        const response = await axios.get('http://localhost:5000/health', { timeout: 1000 });
        const { status, services } = response.data;
        
        if (status === 'ok' && services.camera) {
          cleanup();
          console.log('Backend is ready!');
          isBackendReady = true;
          resolve();
        } else if (status === 'error') {
          cleanup();
          reject(new Error('Backend reported error status'));
        }
      } catch (error) {
        // Ignore connection errors during startup
        if (error.code !== 'ECONNREFUSED') {
          cleanup();
          reject(error);
        }
      }
    }, 1000);
  });
}

// Add function to check if ports are in use
async function isPortInUse(port) {
  try {
    const response = await axios.get(`http://localhost:${port}`, { timeout: 1000 });
    return true;
  } catch (error) {
    return error.code !== 'ECONNREFUSED';
  }
}

// Add function to kill process on port
async function killProcessOnPort(port) {
  if (process.platform === 'win32') {
    try {
      const { execSync } = require('child_process');
      execSync(`netstat -ano | findstr :${port}`).toString().split('\n').forEach(line => {
        const parts = line.trim().split(/\s+/);
        if (parts.length > 4) {
          const pid = parts[parts.length - 1];
          try {
            process.kill(parseInt(pid));
          } catch (e) {
            console.log(`Failed to kill process ${pid}:`, e);
          }
        }
      });
    } catch (e) {
      console.log('No process found on port', port);
    }
  }
}

async function startFrontend() {
  if (isDev) {
    const frontendDir = path.join(__dirname, '../frontend');
    console.log('Starting frontend development server in:', frontendDir);
    
    // Check if ports are in use
    if (await isPortInUse(FRONTEND_PORT)) {
      console.log(`Port ${FRONTEND_PORT} is in use, attempting to kill process...`);
      await killProcessOnPort(FRONTEND_PORT);
      // Wait a bit for the port to be released
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    // Ensure we're in the right directory
    try {
      process.chdir(frontendDir);
      console.log('Changed working directory to:', frontendDir);
    } catch (err) {
      console.error('Failed to change directory:', err);
    }

    // Check if frontend directory exists
    if (!fs.existsSync(frontendDir)) {
      const error = new Error(`Frontend directory not found at: ${frontendDir}`);
      console.error(error.message);
      dialog.showErrorBox('Frontend Error', `Frontend directory not found at:\n${frontendDir}\n\nPlease ensure the frontend directory exists and contains a valid Next.js project.`);
      return Promise.reject(error);
    }
    
    // Check if package.json exists and has required scripts
    const packageJsonPath = path.join(frontendDir, 'package.json');
    if (!fs.existsSync(packageJsonPath)) {
      const error = new Error('package.json not found in frontend directory');
      console.error(error.message);
      dialog.showErrorBox('Frontend Error', 'package.json not found in frontend directory.\n\nPlease ensure you have run "npm init" or copied a valid Next.js project into the frontend folder.');
      return Promise.reject(error);
    }

    // Read and validate package.json
    try {
      const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
      if (!packageJson.scripts || !packageJson.scripts.dev) {
        const error = new Error('Invalid package.json: missing "dev" script');
        console.error(error.message);
        dialog.showErrorBox('Frontend Error', 'package.json is missing the required "dev" script.\n\nPlease ensure this is a valid Next.js project.');
        return Promise.reject(error);
      }
    } catch (err) {
      const error = new Error('Failed to parse package.json');
      console.error(error.message, err);
      dialog.showErrorBox('Frontend Error', 'Failed to parse package.json.\n\nPlease ensure it is a valid JSON file.');
      return Promise.reject(error);
    }
    
    return new Promise((resolve, reject) => {
      let startupTimeout;
      
      // Kill any existing frontend process
      if (frontendProcess) {
        try {
          console.log('Killing existing frontend process...');
          frontendProcess.kill();
          // Wait for process to fully terminate
          setTimeout(() => {
            startFrontendDevServer();
          }, 1000);
        } catch (error) {
          console.warn('Failed to kill existing frontend process:', error);
          startFrontendDevServer();
        }
      } else {
        startFrontendDevServer();
      }

      function startFrontendDevServer() {
        console.log('Starting frontend dev server...');
        console.log('Current working directory:', process.cwd());
        console.log('Node version:', process.version);
        console.log('NPM version:', require('child_process').execSync('npm --version').toString().trim());

        // Use npm run dev with explicit shell options
        frontendProcess = spawn('npm', ['run', 'dev'], {
          cwd: frontendDir,
          shell: process.platform === 'win32' ? 'cmd.exe' : '/bin/bash',
          windowsHide: false,
          stdio: ['pipe', 'pipe', 'pipe'],
          env: { 
            ...process.env, 
            PORT: FRONTEND_PORT.toString(),
            NODE_ENV: 'development',
            FORCE_COLOR: '1',
            NEXT_TELEMETRY_DISABLED: '1',
            ELECTRON_RUN_AS_NODE: '1',
            DEBUG: 'electron*'
          }
        });

        // Buffer for collecting output
        let outputBuffer = '';
        let errorBuffer = '';

        // Log process events with more detail
        frontendProcess.stdout?.on('data', (data) => {
          const output = data.toString();
          outputBuffer += output;
          console.log(`[Frontend Output] ${output.trim()}`);
          
          // Check for specific success patterns
          if (output.includes('ready - started server on') || 
              output.includes('compiled successfully')) {
            console.log('Frontend server started successfully!');
            clearTimeout(startupTimeout);
            isFrontendReady = true;
            resolve();
          }
        });
        
        frontendProcess.stderr?.on('data', (data) => {
          const error = data.toString();
          errorBuffer += error;
          console.error(`[Frontend Error] ${error.trim()}`);
          
          // Check for specific error patterns
          if (error.includes('EADDRINUSE')) {
            console.error('Port 3000 is already in use');
            clearTimeout(startupTimeout);
            reject(new Error('Port 3000 is already in use'));
          }
        });

        frontendProcess.on('spawn', () => {
          console.log('Frontend process spawned successfully');
        });

        startupTimeout = setTimeout(() => {
          console.error('Frontend startup timed out after', FRONTEND_START_TIMEOUT, 'ms');
          console.error('Collected output:', outputBuffer);
          console.error('Collected errors:', errorBuffer);
          if (frontendProcess) {
            console.log('Killing frontend process due to timeout...');
            frontendProcess.kill();
          }
          reject(new Error('Frontend startup timed out'));
        }, FRONTEND_START_TIMEOUT);

        frontendProcess.on('error', (error) => {
          clearTimeout(startupTimeout);
          console.error('Failed to start frontend:', error);
          console.error('Process error details:', {
            code: error.code,
            message: error.message,
            stack: error.stack
          });
          
          let errorMessage = 'Could not start frontend server.\n\n';
          if (error.code === 'ENOENT') {
            errorMessage += 'Please ensure:\n' +
              '1. Node.js is installed and in your PATH\n' +
              '2. You have run "npm install" in the frontend directory\n' +
              '3. The frontend directory contains a valid Next.js project\n\n' +
              'Try running these commands in the frontend directory:\n' +
              'npm install\n' +
              'npm run dev\n\n' +
              'Error details: ' + error.message;
          } else {
            errorMessage += `Error: ${error.message}\n` +
              `Error code: ${error.code}\n` +
              'Please check the console for more details.';
          }
          
          dialog.showErrorBox('Frontend Error', errorMessage);
          reject(error);
        });

        frontendProcess.on('exit', (code, signal) => {
          clearTimeout(startupTimeout);
          if (code !== 0) {
            console.error(`Frontend process exited with code ${code} and signal ${signal}`);
            console.error('Final output buffer:', outputBuffer);
            console.error('Final error buffer:', errorBuffer);
            
            let errorMessage = `Frontend server exited with code ${code}.\n\n`;
            switch (code) {
              case 1:
                errorMessage += 'Compilation error detected.\n\n' +
                  'Please try these steps:\n' +
                  '1. Delete the .next folder in the frontend directory\n' +
                  '2. Run "npm install" in the frontend directory\n' +
                  '3. Run "npm run dev" manually to see the error\n\n' +
                  'Last output:\n' + outputBuffer + '\n\n' +
                  'Last errors:\n' + errorBuffer;
                break;
              case 2:
                errorMessage += 'Port 3000 is already in use.\n' +
                  'Please close any other applications using this port or restart your computer.';
                break;
              default:
                errorMessage += 'Unexpected error occurred.\n\n' +
                  'Last output:\n' + outputBuffer + '\n\n' +
                  'Last errors:\n' + errorBuffer;
            }
            dialog.showErrorBox('Frontend Error', errorMessage);
            reject(new Error(`Frontend process exited with code ${code}`));
          }
        });
      }
    });
  } else {
    // In production, frontend is built and served statically
    isFrontendReady = true;
    return Promise.resolve();
  }
}

async function createWindow() {
  try {
    // Check if ports are in use before creating window
    if (await isPortInUse(FRONTEND_PORT) || await isPortInUse(BACKEND_PORT)) {
      await dialog.showMessageBox({
        type: 'warning',
        title: 'Ports in Use',
        message: 'Some required ports are in use. Would you like to attempt to free them?',
        buttons: ['Yes', 'No'],
        defaultId: 0
      }).then(async (result) => {
        if (result.response === 0) {
          await killProcessOnPort(FRONTEND_PORT);
          await killProcessOnPort(BACKEND_PORT);
          // Wait for ports to be released
          await new Promise(resolve => setTimeout(resolve, 2000));
        } else {
          app.quit();
          return;
        }
      });
    }

    mainWindow = new BrowserWindow({
      width: 1280,
      height: 800,
      webPreferences: {
        nodeIntegration: true,
        contextIsolation: false,
        webSecurity: !isDev
      },
      icon: path.join(__dirname, '../frontend/public/icon.ico'),
      autoHideMenuBar: true,
      frame: true,
      show: false
    });

    mainWindow.setMenu(null);

    const frontendUrl = isDev 
      ? `http://localhost:${FRONTEND_PORT}` 
      : `file://${path.join(__dirname, '../frontend/dist/index.html')}`;

    console.log('Loading frontend from:', frontendUrl);

    // Show window when both services are ready
    const showWindowWhenReady = () => {
      if (isBackendReady && (isDev ? isFrontendReady : true)) {
        console.log('Both services ready, showing window...');
        mainWindow.show();
        mainWindow.focus();
      }
    };

    mainWindow.once('ready-to-show', showWindowWhenReady);

    // Handle failed loads
    mainWindow.webContents.on('did-fail-load', async (event, errorCode, errorDescription) => {
      console.error('Failed to load frontend:', errorDescription, 'Error code:', errorCode);
      
      if (backendStartAttempts < MAX_BACKEND_START_ATTEMPTS) {
        backendStartAttempts++;
        console.log(`Retrying backend start (attempt ${backendStartAttempts}/${MAX_BACKEND_START_ATTEMPTS})...`);
        
        await dialog.showMessageBox(mainWindow, {
          type: 'info',
          title: 'Retrying Connection',
          message: `Attempting to connect to backend (${backendStartAttempts}/${MAX_BACKEND_START_ATTEMPTS})...`,
          buttons: ['OK']
        });

        // Retry loading after a delay
        setTimeout(() => {
          console.log('Retrying frontend load...');
          mainWindow.loadURL(frontendUrl);
        }, 2000);
      } else {
        await dialog.showErrorBox(
          'Connection Failed',
          'Could not establish connection to the backend after multiple attempts.\n\nPlease restart the application.'
        );
        app.quit();
      }
    });

    // Add console logging
    mainWindow.webContents.on('console-message', (event, level, message, line, sourceId) => {
      console.log(`[Renderer ${level}] ${message}`);
    });

    // Handle window close
    mainWindow.on('closed', () => {
      mainWindow = null;
    });

    // Load the frontend URL
    console.log('Loading frontend URL...');
    await mainWindow.loadURL(frontendUrl);
    
    // Force show window after a timeout if it hasn't shown
    setTimeout(() => {
      if (!mainWindow.isVisible()) {
        console.log('Forcing window show after timeout...');
        mainWindow.show();
        mainWindow.focus();
      }
    }, 5000);

  } catch (error) {
    console.error('Error creating window:', error);
    throw error;
  }
}

app.whenReady().then(async () => {
  try {
    console.log('Starting application...');
    console.log('Environment:', isDev ? 'Development' : 'Production');
    
    // Start both services
    if (isDev) {
      console.log('Starting development services...');
      await Promise.all([
        startFrontend().catch(error => {
          console.error('Frontend startup failed:', error);
          dialog.showErrorBox(
            'Frontend Error',
            'Failed to start the frontend development server. Please check the logs for more details.'
          );
          throw error;
        }),
        startBackend().catch(error => {
          console.error('Backend startup failed:', error);
          dialog.showErrorBox(
            'Backend Error',
            'Failed to start the backend server. Please check the logs for more details.'
          );
          throw error;
        })
      ]);
    } else {
      console.log('Starting production backend...');
      await startBackend();
    }
    
    console.log('Creating main window...');
    await createWindow();
  } catch (error) {
    console.error('Failed to start application:', error);
    dialog.showErrorBox(
      'Application Startup Failed',
      `Failed to start the application: ${error.message}\n\nPlease check the logs for more details.`
    );
    app.quit();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    if (backendProcess) {
      backendProcess.kill();
    }
    if (frontendProcess) {
      frontendProcess.kill();
    }
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught exception:', error);
  dialog.showErrorBox(
    'Application Error',
    `An unexpected error occurred: ${error.message}\n\nThe application will now close.`
  );
  app.quit();
}); 