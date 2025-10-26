# OLYMPUS AI Models

This directory contains the five specialist AI models that form the OLYMPUS ensemble for racing intelligence.

## Model Structure

Each model directory contains:
- `model.py` - The trained model implementation
- `train.py` - Training script for the model
- `config.json` - Model configuration and hyperparameters
- `utils.py` - Helper functions specific to each model
- `README.md` - Model-specific documentation

## The OLYMPUS Ensemble

### 1. MINERVA (Strategic Analysis)
- Pit stop optimization
- Fuel strategy calculation
- Tire degradation modeling
- Risk/reward analysis

### 2. ATLAS (Track Positioning) 
- Overtaking opportunity detection
- Defensive positioning recommendations
- Track limit optimization
- Racing line analysis

### 3. IRIS (Vehicle Dynamics)
- Real-time telemetry analysis
- Setup optimization suggestions
- Tire temperature management
- Brake and engine monitoring

### 4. CHRONOS (Timing Analysis)
- Lap time prediction
- Sector performance analysis
- Weather impact calculations
- Pace degradation modeling

### 5. PROMETHEUS (Predictive Strategy)
- Race outcome predictions
- Safety car probability
- Weather change forecasting
- Competitor behavior modeling

### Ensemble Coordinator
The `ensemble/` directory contains the meta-model that combines insights from all five specialists to provide unified recommendations.

## Training Data

Models are trained on:
- Historical race data from all 6 tracks
- Real-time telemetry streams (10Hz)
- Weather conditions
- Driver performance metrics
- Team strategy patterns

## Integration

Models integrate with the ORIS frontend through:
- WebSocket connections for real-time predictions
- REST API for historical analysis
- Event-driven updates for critical decisions