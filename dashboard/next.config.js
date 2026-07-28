/** @type {import('next').NextConfig} */
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const nextConfig = {
  reactStrictMode: true,
  // Proxy /api/* to the FastAPI backend so the browser never hits CORS/mixed-content.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/:path*` }];
  },
};
module.exports = nextConfig;
