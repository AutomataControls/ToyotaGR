# ORIS - OLYMPUS Racing Intelligence System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![TypeScript](https://img.shields.io/badge/typescript-5.9.3-blue.svg)
![React](https://img.shields.io/badge/react-19.1.1-blue.svg)
![Vite](https://img.shields.io/badge/vite-7.1.7-purple.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-green.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.2%2B-orange.svg)
![FastAPI](https://img.shields.io/badge/fastapi-0.104%2B-green.svg)
![InfluxDB](https://img.shields.io/badge/influxdb-3.0-blue.svg)

**Advanced AI-powered racing intelligence platform for Toyota GR "Hack the Track" Hackathon**

ORIS is a complete real-time racing analytics system that combines 5 specialized AI models with actual Toyota GR Cup telemetry data to provide strategic racing insights. Built for the Real-Time Analytics category with full hackathon compliance.

## Hackathon Compliance

**Real-Time Analytics Category**  
**Uses Toyota GR Cup Race Data** from COTA, Road America, Sebring, VIR, Barber, Sonoma  
**Live Processing** of telemetry streams with AI predictions  
**Interactive Dashboard** with strategy recommendations  
**Professional Racing Interface** built for Toyota drivers  

## Overview

ORIS combines five specialized AI models with **real Toyota GR Cup telemetry data** to deliver comprehensive racing intelligence:

- **Real-time telemetry analysis** from actual Toyota races (60Hz data processing)
- **AI-powered strategy recommendations** using 5 specialist models
- **Live database integration** with InfluxDB 3 Core for data correlation
- **Real-time 2D track visualization** of racing lines and track analysis
- **Complete settings management** with user access control

## Features

### AI Specialists (5+ Million Parameters)

- **MINERVA** (~800K params): Strategic racing intelligence with multi-head attention for pit strategy and race tactics
- **ATLAS** (~1.2M params): Spatial track intelligence with position-aware attention and track memory systems  
- **IRIS** (~1M params): Vehicle dynamics intelligence with specialized throttle/brake/balance analysis
- **CHRONOS** (~1.3M params): Timing intelligence with LSTM networks and temporal pattern recognition
- **PROMETHEUS** (~1.4M params): Predictive analytics with multi-horizon forecasting and race outcome modeling

### Racing Intelligence

- **Live Strategy Advisor**: Real-time Standard/Push/Conserve mode recommendations
- **Toyota Data Integration**: Processing actual GR Cup race telemetry files  
- **Live Data Feed**: Real-time telemetry streaming from field cars with toggle control
- **Database Management**: InfluxDB 3 Core with user access control and live connection monitoring
- **Interactive Settings**: Complete configuration interface for database, AI models, alerts
- **Responsive Design**: Optimized for all screen sizes and racing environments

### Supported Tracks

1. Barber Motorsports Park
2. Circuit of the Americas (COTA)
3. Road America
4. Sebring International Raceway
5. Sonoma Raceway
6. Virginia International Raceway (VIR)

## Technology Stack

### Frontend
- **React 19.1** with TypeScript 5.9
- **Vite 7.1** for lightning-fast development
- **Custom 2D Canvas** for track visualization
- **Recharts** for real-time telemetry graphs
- **Lucide React** for professional racing icons
- **Responsive CSS** with mobile optimization

### Backend & AI
- **FastAPI** REST API serving AI model predictions
- **PyTorch 2.2+** for neural network inference
- **5 Specialized AI Models** (MINERVA, ATLAS, IRIS, CHRONOS, PROMETHEUS)
- **Concurrent server startup** with automatic dependency management
- **Flexible Training**: Models trainable locally, Google Colab, or cloud platforms

### Database & Data
- **InfluxDB 3 Core OSS** for telemetry time-series data
- **Toyota GR Cup Data** from 6 professional racing circuits
- **CSV/JSON processing** for historical race analysis
- **Real-time data correlation** and vector storage

## Quick Start

### Prerequisites
- **Node.js 18+** and npm
- **Python 3.12+** 
- **Git** for cloning the repository

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/AutomataControls/ToyotaGR.git
cd ToyotaGR
```

2. **Install all dependencies (Node.js + Python):**
```bash
npm install
```

3. **Start the application (auto-starts both servers):**
```bash
npm run dev
```

That's it! The command will:
- Create Python virtual environment automatically
- Install PyTorch, FastAPI, and all Python dependencies  
- Start AI API server on port 8000
- Start React app on port 5173
- Run both servers concurrently

### Manual Commands

```bash
# Install Python dependencies only
npm run setup:venv

# Start AI API server only  
npm run start:api

# Start React frontend only
npm run start:frontend

# Production build
npm run build
```

### InfluxDB Setup (Optional)

For full database functionality:

1. **Install InfluxDB 3 Core:**
```bash
# Download from InfluxDB website
# Or use Docker: docker run -p 8181:8181 influxdb:3.0
```

2. **Start InfluxDB server:**
```bash
influxdb3 serve --node-id host01 --object-store file --data-dir ~/.influxdb3
```

3. **Configure in ORIS Settings:**
   - URL: `http://localhost:8181`
   - Username: `your_username`  
   - Token: `your_token`
   - Database: `toyota_gr_telemetry`

## Project Structure

```
ToyotaGR/
├── src/
│   ├── components/         # React components
│   │   ├── layout/        # Header, Sidebar, Layout
│   │   ├── common/        # Reusable UI components  
│   │   ├── dashboard/     # Main dashboard widgets
│   │   ├── strategy/      # Strategy advisor components
│   │   └── telemetry/     # Telemetry visualization
│   ├── pages/             # Main application pages
│   │   ├── Dashboard.tsx  # Main racing dashboard
│   │   ├── Strategy.tsx   # Strategy recommendations
│   │   ├── Telemetry.tsx  # Live telemetry graphs
│   │   ├── Database.tsx   # Live data feed management
│   │   ├── Help.tsx       # Documentation and AI model details
│   │   └── Settings.tsx   # System configuration
│   ├── services/          # API clients and data services
│   │   ├── aiModelAPI.py      # FastAPI server for AI models
│   │   ├── aiModelClient.ts   # TypeScript AI client
│   │   ├── liveDataService.ts # Live data feed management
│   │   └── toyotaDataLoader.ts # Toyota data processing
│   ├── models/            # AI model implementations  
│   │   ├── minerva/       # Strategic analysis model
│   │   ├── atlas/         # Spatial reasoning model
│   │   ├── iris/          # Vehicle dynamics model
│   │   ├── chronos/       # Timing analysis model
│   │   └── prometheus/    # Predictive modeling
│   └── data/              # Toyota GR Cup race data
│       └── tracks/        # COTA, Road America, etc.
├── venv/                  # Python virtual environment (auto-created)
├── dist/                  # Production build output
├── requirements.txt       # Python dependencies
└── package.json          # Node.js dependencies & scripts
```

## API Reference

### AI Model Endpoints

```bash
# Get strategy recommendations from MINERVA
POST http://localhost:8000/predict/minerva

# Get spatial analysis from ATLAS  
POST http://localhost:8000/predict/atlas

# Get vehicle dynamics from IRIS
POST http://localhost:8000/predict/iris

# Get timing predictions from CHRONOS
POST http://localhost:8000/predict/chronos

# Get future predictions from PROMETHEUS
POST http://localhost:8000/predict/prometheus

# Get all model status
GET http://localhost:8000/models/status
```

### Live Data Endpoints

```bash
# Get live data status
GET http://localhost:8000/live-data/status

# Toggle live data feed
POST http://localhost:8000/live-data/toggle

# Get connected field cars
GET http://localhost:8000/live-data/cars

# Get recent telemetry packets
GET http://localhost:8000/live-data/recent?limit=20

# Field car data connection endpoint
POST http://localhost:8000/live-data/connect
```

### Key Features

- **5 AI Specialists**: Each model provides specialized racing insights
- **Real Toyota Data**: Processing actual GR Cup telemetry from 6 tracks
- **Live Data Feed**: Real-time telemetry streaming from field cars with toggle control
- **Live Updates**: Real-time strategy recommendations and connection monitoring
- **Database Integration**: InfluxDB 3 Core for time-series data with live feed management
- **Full Settings**: User management, database config, model status, live data configuration
- **Comprehensive Help**: Detailed AI model architecture documentation
- **Responsive UI**: Works on all devices and screen sizes
- **Auto-Deploy**: One command starts everything

## Performance

- **Startup Time**: < 30 seconds for full system (both servers + AI models)
- **AI Inference**: < 100ms per model, < 500ms for all 5 models  
- **React Build**: ~30 seconds optimized production bundle
- **Memory Usage**: ~1GB total (React + Python + AI models)
- **Real-time Processing**: 60Hz telemetry data streams supported

## Development

```bash
# Development with hot reload
npm run dev

# Build production bundle  
npm run build

# Preview production build
npm run preview

# Python environment setup
npm run setup:venv

# Run linting
npm run lint
```

## Hackathon Categories

🏆 **COMPETING IN: Real-Time Analytics Category**  

**Why ORIS Wins:**
- **Real AI Implementation**: 5+ million parameter PyTorch neural networks (not mock data)
- **Live Race Engineer Tool**: Professional real-time decision support system
- **Toyota Integration**: Processes actual GR Cup telemetry with exact parameter specifications
- **Production Ready**: Complete full-stack implementation with live data capabilities

**Toyota Requirements Met**:
- Real Toyota GR Cup data integration  
- Professional racing interface  
- Real-time analytics processing  
- Strategic insights for drivers  
- Scalable architecture  

## License

MIT License  

**🏁 Built for Toyota GR "Hack the Track" Hackathon 2024**  
**Category**: Real-Time Analytics  
**Team**: AutomataControls

## Acknowledgments

- **Toyota Racing Development** for GR Cup telemetry data access
- **InfluxDB** for time-series database technology  
- **FastAPI** for high-performance Python web framework
- **React/Vite** for modern frontend development

---

**Ready to race? Run `npm run dev` and experience the future of racing intelligence!**