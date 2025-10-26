# ORIS - OLYMPUS Racing Intelligence System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![TypeScript](https://img.shields.io/badge/typescript-5.6.2-blue.svg)
![React](https://img.shields.io/badge/react-18.3.1-blue.svg)
![Vite](https://img.shields.io/badge/vite-6.0.1-purple.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-green.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)

Advanced AI-powered racing intelligence platform for Toyota GR Cup drivers. ORIS leverages cutting-edge pattern recognition models adapted from ARC-AGI-2 competition winners to provide real-time strategic insights during races.

## Overview

ORIS combines five specialized AI models (MINERVA, ATLAS, IRIS, CHRONOS, PROMETHEUS) to deliver comprehensive racing intelligence:

- **Real-time telemetry analysis** with sub-second latency
- **Strategic pit stop optimization** based on tire degradation and fuel consumption
- **Track position visualization** with 3D track models
- **Predictive race modeling** for competitor behavior and weather conditions
- **Lap time analysis** with sector-by-sector breakdowns

## Features

### AI Specialists

- **MINERVA**: Strategic pattern recognition for race tactics
- **ATLAS**: Spatial reasoning for optimal racing lines
- **IRIS**: Visual pattern analysis for track conditions
- **CHRONOS**: Temporal pattern recognition for timing optimization
- **PROMETHEUS**: Predictive synthesis for race outcome modeling

### Racing Intelligence

- **Pit Strategy Advisor**: AI-driven recommendations for optimal pit windows
- **Tire Management**: Real-time tire degradation analysis and compound selection
- **Fuel Optimization**: Consumption predictions and lift-and-coast recommendations
- **Weather Adaptation**: Dynamic strategy adjustments based on weather forecasts
- **Competitor Analysis**: Behavioral pattern recognition of other drivers

### Supported Tracks

1. Barber Motorsports Park
2. Circuit of the Americas (COTA)
3. Road America
4. Sebring International Raceway
5. Sonoma Raceway
6. Virginia International Raceway (VIR)

## Technology Stack

### Frontend
- React 18.3 with TypeScript
- Vite for blazing-fast development
- Three.js for 3D track visualization
- Recharts for real-time data visualization
- Lucide React for professional icons

### AI/ML
- PyTorch 2.0+ for model training
- Custom grid-based encoding (3x3 to 30x30)
- Ensemble learning with confidence weighting
- Transfer learning from ARC-AGI-2 models

### Data Pipeline
- WebSocket for real-time telemetry streaming
- CSV parsing for historical race data
- 60Hz telemetry processing capability

## Installation

### Prerequisites
- Node.js 18+ and npm
- Python 3.10+
- CUDA-capable GPU (recommended for AI inference)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/oris-racing-app.git
cd oris-racing-app
```

2. Install frontend dependencies:
```bash
npm install
```

3. Install Python dependencies:
```bash
pip install torch numpy pandas scikit-learn
```

4. Download pre-trained models from Google Drive and place in `src/models/`:
   - minerva_racing.pt
   - atlas_racing.pt
   - iris_racing.pt
   - chronos_racing.pt
   - prometheus_racing.pt

5. Start the development server:
```bash
npm run dev
```

## Usage

### Development Mode
```bash
npm run dev
```

### Production Build
```bash
npm run build
npm run preview
```

### Training Models
```bash
python src/models/OLYMPUSRacing_Training.ipynb
```

## Project Structure

```
oris-racing-app/
├── src/
│   ├── components/         # React components
│   │   ├── layout/        # Layout components
│   │   ├── dashboard/     # Dashboard widgets
│   │   └── predictions/   # AI prediction displays
│   ├── models/            # AI model implementations
│   │   ├── minerva/       # Strategic analysis
│   │   ├── atlas/         # Spatial reasoning
│   │   ├── iris/          # Visual patterns
│   │   ├── chronos/       # Temporal analysis
│   │   └── prometheus/    # Predictive modeling
│   ├── data/              # Race data and track info
│   │   └── tracks/        # Track-specific datasets
│   └── services/          # API and WebSocket services
├── public/
│   └── track-models/      # 3D GLB track models
└── package.json
```

## API Reference

### WebSocket Events

```typescript
// Subscribe to telemetry updates
ws.on('telemetry', (data: TelemetryData) => {
  // Handle real-time telemetry
});

// Subscribe to AI predictions
ws.on('prediction', (data: PredictionData) => {
  // Handle AI recommendations
});
```

### AI Model Interface

```typescript
interface RaceState {
  telemetry: TelemetryData;
  timing: TimingData;
  weather: WeatherData;
  standings: StandingsData;
}

interface AIRecommendation {
  pitStrategy: PitWindow[];
  tireChoice: TireCompound;
  fuelTarget: number;
  confidence: number;
}
```

## Performance

- **Telemetry Processing**: < 16ms latency (60 FPS)
- **AI Inference**: < 100ms for full ensemble prediction
- **Memory Usage**: ~2GB with all models loaded
- **GPU Acceleration**: 10x speedup with CUDA

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Toyota Racing Development for track data and telemetry access
- ARC-AGI-2 competition for inspiring the pattern recognition approach
- Three.js community for 3D visualization capabilities

## Support

For issues, questions, or contributions, please open an issue on GitHub.