/** @type {import('next').NextConfig} */
const nextConfig = {
  // Remove static export for development
  // output: 'export',  // Only use this for production build
  images: {
    unoptimized: true,
    domains: ['localhost'],
  },
  // Add basePath and assetPrefix for Electron
  basePath: process.env.NODE_ENV === 'development' ? '' : '',
  assetPrefix: process.env.NODE_ENV === 'development' ? '' : '',
  // Configure webpack
  webpack: (config, { isServer }) => {
    // Add canvas to externals
    config.externals = [...(config.externals || []), { canvas: "canvas" }];
    
    // Add fallbacks for node modules
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
        child_process: false,
      };
    }
    
    return config;
  },
}

module.exports = nextConfig; 