/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',  // Enable static exports
  images: {
    unoptimized: true, // Required for static export
  },
  // Remove experimental.appDir as it's not needed
  webpack: (config: { externals: any[]; }) => {
    config.externals = [...(config.externals || []), { canvas: "canvas" }];
    return config;
  },
}

export default nextConfig;
