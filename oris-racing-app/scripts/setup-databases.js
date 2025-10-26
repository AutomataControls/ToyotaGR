#!/usr/bin/env node

import { execSync } from 'child_process';
import { mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = join(__dirname, '..');

console.log('🏁 Setting up ORIS databases...\n');

// Create data directory
const dataDir = join(projectRoot, 'data');
if (!existsSync(dataDir)) {
  mkdirSync(dataDir, { recursive: true });
  console.log('✅ Created data directory');
}

// Setup InfluxDB
console.log('\n📊 Setting up InfluxDB...');
console.log('Please ensure InfluxDB 3.0 OSS is installed and running.');
console.log('Download from: https://www.influxdata.com/downloads/');
console.log('\nDefault configuration:');
console.log('  - URL: http://localhost:8086');
console.log('  - Create a bucket named "telemetry"');
console.log('  - Create an API token with read/write permissions');
console.log('  - Update .env with your token');

// SQLite will be created automatically when the app starts
console.log('\n💾 SQLite database will be created automatically on first run.');

// Install dependencies if needed
console.log('\n📦 Installing database dependencies...');
try {
  execSync('npm install', { stdio: 'inherit', cwd: projectRoot });
  console.log('✅ Dependencies installed');
} catch (error) {
  console.error('❌ Failed to install dependencies:', error.message);
}

console.log('\n🏁 Database setup complete!');
console.log('\nNext steps:');
console.log('1. Copy .env.example to .env');
console.log('2. Update .env with your InfluxDB credentials');
console.log('3. Start InfluxDB service');
console.log('4. Run npm run dev to start the application');