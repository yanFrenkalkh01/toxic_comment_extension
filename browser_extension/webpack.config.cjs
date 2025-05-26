const path = require('path');
const BundleAnalyzerPlugin = require('webpack-bundle-analyzer').BundleAnalyzerPlugin;

module.exports = {
  mode: 'production', // or 'development' for easier debugging
  entry: {
    content_script: './src/youtube.js', 
    // background: './src/background.js', // If you have a background script
  },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: '[name].bundle.js', // Output bundled files
    chunkFilename: '[name].[contenthash].chunk.js',
    publicPath: ''
  },
  module: {
    rules: [
      {
        test: /\.m?js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: { // These are Babel's options
            presets: [
              [
                "@babel/preset-env",
                {
                  "modules": false // Correctly nested here
                }
              ]
            ]
          }
        }
      }
    ]
  },
  resolve: {
    // In case 'natural' or other libs have issues with certain Node built-ins
    fallback: {
      "fs": false,
      "path": false,
      "crypto": false,
      "os": false,
      "stream": false,
      "buffer": false,
      "util": false,
      "assert": false,
      "vm": false,
      "constants": false,
      "zlib": false,
      "http": false,
      "https": false,
      "url": false,
      "net": false,      // Mock or provide a polyfill if available
      "tls": false,      // Mock or provide a polyfill if available
      "dns": false,      // Mock or provide a polyfill if available
      "webworker-threads": false, // No direct browser equivalent, likely mock
      "pg-native": false, // Mock, browser won't use native bindings
    }
  },
};