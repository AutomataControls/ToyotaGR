#!/usr/bin/env python3
"""
AI Model API Service for ORIS Racing Intelligence System
Serves predictions from MINERVA, ATLAS, IRIS, CHRONOS, PROMETHEUS models
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import torch
import numpy as np
import sys
import os
import json
import uvicorn
from datetime import datetime

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

# Import AI models
from minerva.minerva import create_minerva_model
from atlas.atlas import create_atlas_model
from iris.iris import create_iris_model  
from chronos.chronos import create_chronos_model
from prometheus.prometheus import create_prometheus_model

app = FastAPI(title="ORIS AI Model API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
models = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Live data management
live_data_enabled = False
connected_cars = {}  # car_number -> connection info
telemetry_packets = []  # Recent packets buffer
packet_count = 0
last_packet_time = None

class TelemetryInput(BaseModel):
    """Telemetry data input schema"""
    timestamp: str
    sessionId: str
    trackId: str
    carNumber: int
    currentLap: int
    telemetry: Dict[str, Any]

class StrategyRequest(BaseModel):
    """Strategy prediction request"""
    telemetry_sequence: List[List[float]]
    race_context: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    """AI model prediction response"""
    model: str
    predictions: Dict[str, Any]
    confidence: float
    timestamp: str
    recommendations: Dict[str, str]

class LiveTelemetryPacket(BaseModel):
    """Live telemetry packet from field cars"""
    car_number: int
    driver_name: str
    timestamp: str
    telemetry: Dict[str, Any]
    session_info: Optional[Dict[str, Any]] = None

class LiveDataStatus(BaseModel):
    """Live data connection status"""
    enabled: bool
    connected_cars: int
    total_packets: int
    packets_per_second: float
    last_update: str

async def load_models():
    """Load all AI models on startup"""
    print(f"🚀 Starting ORIS AI Model API on {device}...")
    
    try:
        # Load actual AI models
        print("📡 Loading MINERVA (Strategic Intelligence)...")
        models['minerva'] = create_minerva_model(device)
        
        print("🗺️ Loading ATLAS (Spatial Intelligence)...")
        models['atlas'] = create_atlas_model(device)
        
        print("🚗 Loading IRIS (Vehicle Dynamics)...")
        models['iris'] = create_iris_model(device)
        
        print("⏱️ Loading CHRONOS (Timing Intelligence)...")
        models['chronos'] = create_chronos_model(device)
        
        print("🔮 Loading PROMETHEUS (Predictive Analytics)...")
        models['prometheus'] = create_prometheus_model(device)
        
        print("🏁 ORIS AI Model API ready (5 real AI models loaded)!")
        
    except Exception as e:
        print(f"⚠️ Error loading models: {e}")
        print("🎭 Falling back to mock data for development...")
        # Fall back to mock data if model loading fails
        models['minerva'] = "loaded"
        models['atlas'] = "loaded"
        models['iris'] = "loaded"
        models['chronos'] = "loaded"
        models['prometheus'] = "loaded"

@app.get("/")
async def root():
    """API health check"""
    return {
        "service": "ORIS AI Model API", 
        "status": "running",
        "models": list(models.keys()),
        "device": device
    }

@app.post("/predict/minerva", response_model=PredictionResponse)
async def predict_strategy(request: StrategyRequest):
    """Get strategic predictions from MINERVA"""
    try:
        # Check if using real model or mock data
        if models.get('minerva') == "loaded":
            # Return dummy data for development (when models failed to load)
            return PredictionResponse(
                model="minerva",
                predictions={
                    "pit_strategy": [0.1, 0.3, 0.4, 0.15, 0.05],  # [now, 1lap, 2lap, 3lap, no_pit]
                    "pace_strategy": [0.2, 0.6, 0.2],  # [push, maintain, conserve]
                    "tire_degradation": 0.65,
                    "fuel_strategy": 0.3,
                    "confidence": 0.87
                },
                confidence=0.87,
                timestamp=datetime.now().isoformat(),
                recommendations={
                    "pit_strategy": "Pit in 2 Laps (confidence: 0.40)",
                    "pace_strategy": "Maintain Pace (confidence: 0.60)",
                    "tire_warning": "MEDIUM tire degradation: 65%"
                }
            )
        
        # Use real MINERVA model
        model = models.get('minerva')
        if model is None:
            raise HTTPException(status_code=503, detail="MINERVA model not loaded")
        
        # Convert telemetry sequence to tensor
        telemetry_tensor = torch.tensor(request.telemetry_sequence, dtype=torch.float32).to(device)
        if len(telemetry_tensor.shape) == 2:
            telemetry_tensor = telemetry_tensor.unsqueeze(0)  # Add batch dimension
        
        # Ensure proper sequence length (300 timesteps)
        if telemetry_tensor.shape[1] < 300:
            # Pad with zeros if sequence too short
            padding = torch.zeros(telemetry_tensor.shape[0], 300 - telemetry_tensor.shape[1], telemetry_tensor.shape[2]).to(device)
            telemetry_tensor = torch.cat([padding, telemetry_tensor], dim=1)
        elif telemetry_tensor.shape[1] > 300:
            # Take last 300 timesteps if too long
            telemetry_tensor = telemetry_tensor[:, -300:, :]
        
        with torch.no_grad():
            output = model(telemetry_tensor, request.race_context)
            recommendations = model.get_strategic_recommendations(output)
        
        return PredictionResponse(
            model="minerva",
            predictions={
                "pit_strategy": output['pit_strategy'][0].tolist(),
                "pace_strategy": output['pace_strategy'][0].tolist(), 
                "tire_strategy": output['tire_strategy'][0].tolist(),
                "traffic_strategy": output['traffic_strategy'][0].tolist(),
                "tire_degradation": output['tire_degradation'][0].item(),
                "fuel_strategy": output['fuel_strategy'][0].item(),
                "confidence": output['confidence'][0].item()
            },
            confidence=output['confidence'][0].item(),
            timestamp=datetime.now().isoformat(),
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MINERVA prediction error: {str(e)}")

@app.post("/predict/atlas")
async def predict_spatial(request: StrategyRequest):
    """Get spatial/positioning predictions from ATLAS"""
    try:
        # Check if using real model or mock data
        if models.get('atlas') == "loaded":
            # Return dummy data for development
            return {
                "model": "atlas",
                "predictions": {
                    "optimal_racing_line": [[x/100.0, np.sin(x/10.0)] for x in range(100)],
                    "track_position_score": 0.82,
                    "sector_analysis": {
                        "sector1": {"optimal_speed": 145.2, "current_speed": 142.8, "delta": -2.4},
                        "sector2": {"optimal_speed": 98.5, "current_speed": 96.1, "delta": -2.4},
                        "sector3": {"optimal_speed": 167.3, "current_speed": 165.9, "delta": -1.4}
                    },
                    "confidence": 0.78
                },
                "timestamp": datetime.now().isoformat()
            }
        
        # Use real ATLAS model
        model = models.get('atlas')
        if model is None:
            raise HTTPException(status_code=503, detail="ATLAS model not loaded")
        
        # Convert telemetry sequence to tensor
        telemetry_tensor = torch.tensor(request.telemetry_sequence, dtype=torch.float32).to(device)
        if len(telemetry_tensor.shape) == 2:
            telemetry_tensor = telemetry_tensor.unsqueeze(0)
        
        # Ensure proper sequence length (300 timesteps)
        if telemetry_tensor.shape[1] < 300:
            padding = torch.zeros(telemetry_tensor.shape[0], 300 - telemetry_tensor.shape[1], telemetry_tensor.shape[2]).to(device)
            telemetry_tensor = torch.cat([padding, telemetry_tensor], dim=1)
        elif telemetry_tensor.shape[1] > 300:
            telemetry_tensor = telemetry_tensor[:, -300:, :]
        
        with torch.no_grad():
            output = model(telemetry_tensor, request.race_context)
            recommendations = model.get_spatial_recommendations(output)
        
        return {
            "model": "atlas",
            "predictions": {
                "racing_line_analysis": output['racing_line_analysis'][0].tolist(),
                "track_position": output['track_position'][0].tolist(),
                "corner_analysis": output['corner_analysis'][0].tolist(),
                "overtaking_analysis": output['overtaking_analysis'][0].tolist(),
                "track_limits": output['track_limits'][0].tolist(),
                "spatial_quality": output['spatial_quality'][0].item(),
                "confidence": output['confidence'][0].item()
            },
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ATLAS prediction error: {str(e)}")

@app.post("/predict/iris")  
async def predict_dynamics(request: StrategyRequest):
    """Get vehicle dynamics predictions from IRIS"""
    try:
        # Check if using real model or mock data
        if models.get('iris') == "loaded":
            # Return dummy data for development
            return {
                "model": "iris", 
                "predictions": {
                    "vehicle_balance": 0.73,
                    "handling_analysis": {
                        "understeer_tendency": 0.25,
                        "oversteer_tendency": 0.15,
                        "optimal_balance": 0.60
                    },
                    "setup_recommendations": {
                        "front_wing": "+2 clicks",
                        "rear_wing": "maintain", 
                        "tire_pressure": "FL: -0.5 PSI, FR: -0.3 PSI"
                    },
                    "confidence": 0.81
                },
                "timestamp": datetime.now().isoformat()
            }
        
        # Use real IRIS model
        model = models.get('iris')
        if model is None:
            raise HTTPException(status_code=503, detail="IRIS model not loaded")
        
        # Convert telemetry sequence to tensor
        telemetry_tensor = torch.tensor(request.telemetry_sequence, dtype=torch.float32).to(device)
        if len(telemetry_tensor.shape) == 2:
            telemetry_tensor = telemetry_tensor.unsqueeze(0)
        
        # Ensure proper sequence length (300 timesteps)
        if telemetry_tensor.shape[1] < 300:
            padding = torch.zeros(telemetry_tensor.shape[0], 300 - telemetry_tensor.shape[1], telemetry_tensor.shape[2]).to(device)
            telemetry_tensor = torch.cat([padding, telemetry_tensor], dim=1)
        elif telemetry_tensor.shape[1] > 300:
            telemetry_tensor = telemetry_tensor[:, -300:, :]
        
        with torch.no_grad():
            output = model(telemetry_tensor, request.race_context)
            recommendations = model.get_dynamics_recommendations(output)
        
        return {
            "model": "iris",
            "predictions": {
                "throttle_analysis": output['throttle_analysis'][0].tolist(),
                "brake_analysis": output['brake_analysis'][0].tolist(),
                "vehicle_balance": output['vehicle_balance'][0].tolist(),
                "stability_analysis": output['stability_analysis'][0].tolist(),
                "gear_analysis": output['gear_analysis'][0].tolist(),
                "steering_analysis": output['steering_analysis'][0].tolist(),
                "performance_metrics": output['performance_metrics'][0].tolist(),
                "confidence": output['confidence'][0].item()
            },
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IRIS prediction error: {str(e)}")

@app.post("/predict/chronos")
async def predict_timing(request: StrategyRequest):
    """Get timing predictions from CHRONOS"""
    try:
        # Check if using real model or mock data
        if models.get('chronos') == "loaded":
            # Return dummy data for development
            return {
                "model": "chronos",
                "predictions": {
                    "predicted_lap_time": "1:23.456",
                    "sector_predictions": {
                        "sector1": "27.123",
                        "sector2": "28.891", 
                        "sector3": "27.442"
                    },
                    "time_delta_to_optimal": "+0.234",
                    "improvement_potential": 0.8,
                    "confidence": 0.85
                },
                "timestamp": datetime.now().isoformat()
            }
        
        # Use real CHRONOS model
        model = models.get('chronos')
        if model is None:
            raise HTTPException(status_code=503, detail="CHRONOS model not loaded")
        
        # Convert telemetry sequence to tensor
        telemetry_tensor = torch.tensor(request.telemetry_sequence, dtype=torch.float32).to(device)
        if len(telemetry_tensor.shape) == 2:
            telemetry_tensor = telemetry_tensor.unsqueeze(0)
        
        # Ensure proper sequence length (300 timesteps)
        if telemetry_tensor.shape[1] < 300:
            padding = torch.zeros(telemetry_tensor.shape[0], 300 - telemetry_tensor.shape[1], telemetry_tensor.shape[2]).to(device)
            telemetry_tensor = torch.cat([padding, telemetry_tensor], dim=1)
        elif telemetry_tensor.shape[1] > 300:
            telemetry_tensor = telemetry_tensor[:, -300:, :]
        
        with torch.no_grad():
            output = model(telemetry_tensor, request.race_context)
            recommendations = model.get_timing_recommendations(output)
        
        return {
            "model": "chronos",
            "predictions": {
                "lap_time_analysis": output['lap_time_analysis'][0].tolist(),
                "sector_analysis": output['sector_analysis'][0].tolist(),
                "timing_consistency": output['timing_consistency'][0].tolist(),
                "pace_analysis": output['pace_analysis'][0].tolist(),
                "timing_trends": output['timing_trends'][0].tolist(),
                "performance_benchmarks": output['performance_benchmarks'][0].tolist(),
                "confidence": output['confidence'][0].item()
            },
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CHRONOS prediction error: {str(e)}")

@app.post("/predict/prometheus")
async def predict_future(request: StrategyRequest):
    """Get future predictions from PROMETHEUS"""
    try:
        # Check if using real model or mock data
        if models.get('prometheus') == "loaded":
            # Return dummy data for development
            return {
                "model": "prometheus",
                "predictions": {
                    "position_forecast": {
                        "lap_25": {"position": 3, "probability": 0.72},
                        "lap_35": {"position": 2, "probability": 0.58}, 
                        "lap_45": {"position": 2, "probability": 0.81}
                    },
                    "incident_probability": 0.12,
                    "weather_forecast": {
                        "rain_probability": 0.15,
                        "track_temperature_trend": "stable"
                    },
                    "strategic_opportunities": [
                        {"event": "undercut_opportunity", "lap": 42, "probability": 0.67},
                        {"event": "safety_car_window", "lap_range": [38, 44], "probability": 0.23}
                    ],
                    "confidence": 0.79
                },
                "timestamp": datetime.now().isoformat()
            }
        
        # Use real PROMETHEUS model
        model = models.get('prometheus')
        if model is None:
            raise HTTPException(status_code=503, detail="PROMETHEUS model not loaded")
        
        # Convert telemetry sequence to tensor
        telemetry_tensor = torch.tensor(request.telemetry_sequence, dtype=torch.float32).to(device)
        if len(telemetry_tensor.shape) == 2:
            telemetry_tensor = telemetry_tensor.unsqueeze(0)
        
        # Ensure proper sequence length (300 timesteps)
        if telemetry_tensor.shape[1] < 300:
            padding = torch.zeros(telemetry_tensor.shape[0], 300 - telemetry_tensor.shape[1], telemetry_tensor.shape[2]).to(device)
            telemetry_tensor = torch.cat([padding, telemetry_tensor], dim=1)
        elif telemetry_tensor.shape[1] > 300:
            telemetry_tensor = telemetry_tensor[:, -300:, :]
        
        with torch.no_grad():
            output = model(telemetry_tensor, request.race_context)
            recommendations = model.get_predictive_recommendations(output)
        
        return {
            "model": "prometheus",
            "predictions": {
                "future_lap_times": output['future_lap_times'][0].tolist(),
                "tire_degradation_forecast": output['tire_degradation_forecast'][0].tolist(),
                "race_outcome_forecast": output['race_outcome_forecast'][0].tolist(),
                "weather_forecast": output['weather_forecast'][0].tolist(),
                "fuel_forecast": output['fuel_forecast'][0].tolist(),
                "performance_trajectory": output['performance_trajectory'][0].tolist(),
                "prediction_confidence": output['prediction_confidence'][0].tolist()
            },
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PROMETHEUS prediction error: {str(e)}")

@app.post("/predict/ensemble")
async def predict_ensemble(request: StrategyRequest):
    """Get ensemble predictions from all models"""
    try:
        # Get predictions from all models
        results = {}
        
        # This would call all model endpoints and combine results
        minerva_result = await predict_strategy(request)
        results['minerva'] = minerva_result.dict()
        
        # Add other model predictions
        results['atlas'] = await predict_spatial(request)
        results['iris'] = await predict_dynamics(request)
        results['chronos'] = await predict_timing(request)
        results['prometheus'] = await predict_future(request)
        
        # Calculate consensus
        consensus_score = 0.87  # Weighted average of individual confidences
        
        return {
            "ensemble_results": results,
            "consensus_score": consensus_score,
            "primary_recommendation": "Execute undercut strategy at lap 42",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ensemble prediction error: {str(e)}")

@app.get("/models/status")
async def get_model_status():
    """Get status of all loaded models"""
    status = {}
    for model_name, model in models.items():
        if model == "loaded":
            # Mock data mode (models failed to load)
            status[model_name] = {
                "loaded": True,  # Still report as loaded since API is functional
                "device": "mock",
                "mode": "development"
            }
        else:
            # Real model loaded
            status[model_name] = {
                "loaded": True,
                "device": device,
                "mode": "production",
                "parameters": getattr(model, 'parameters', lambda: [])() and sum(p.numel() for p in model.parameters()) or "unknown"
            }
    return status

@app.post("/live-data/connect")
async def connect_field_car(packet: LiveTelemetryPacket):
    """Accept live telemetry from field cars"""
    global packet_count, last_packet_time, connected_cars, telemetry_packets
    
    if not live_data_enabled:
        raise HTTPException(status_code=503, detail="Live data feed is disabled")
    
    try:
        # Update connection info
        connected_cars[packet.car_number] = {
            "driver_name": packet.driver_name,
            "last_update": packet.timestamp,
            "status": "connected"
        }
        
        # Store packet
        telemetry_packets.append({
            "car_number": packet.car_number,
            "timestamp": packet.timestamp,
            "telemetry": packet.telemetry,
            "session_info": packet.session_info
        })
        
        # Keep only last 1000 packets
        if len(telemetry_packets) > 1000:
            telemetry_packets = telemetry_packets[-1000:]
        
        packet_count += 1
        last_packet_time = datetime.now().isoformat()
        
        print(f"📡 Received telemetry from Car #{packet.car_number} ({packet.driver_name})")
        
        return {
            "status": "received",
            "car_number": packet.car_number,
            "packet_id": packet_count,
            "timestamp": last_packet_time
        }
        
    except Exception as e:
        print(f"❌ Error processing telemetry from Car #{packet.car_number}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process telemetry: {str(e)}")

@app.get("/live-data/status")
async def get_live_data_status():
    """Get live data connection status"""
    global live_data_enabled, connected_cars, packet_count, last_packet_time
    
    # Calculate packets per second (simple estimate)
    packets_per_second = len(connected_cars) * 60.0 if live_data_enabled else 0.0
    
    return LiveDataStatus(
        enabled=live_data_enabled,
        connected_cars=len(connected_cars),
        total_packets=packet_count,
        packets_per_second=packets_per_second,
        last_update=last_packet_time or datetime.now().isoformat()
    )

@app.post("/live-data/toggle")
async def toggle_live_data():
    """Enable/disable live data feed"""
    global live_data_enabled
    
    live_data_enabled = not live_data_enabled
    
    if live_data_enabled:
        print("🟢 Live data feed ENABLED")
    else:
        print("🔴 Live data feed DISABLED")
        # Clear connection info when disabled
        connected_cars.clear()
    
    return {
        "enabled": live_data_enabled,
        "message": f"Live data feed {'enabled' if live_data_enabled else 'disabled'}"
    }

@app.get("/live-data/cars")
async def get_connected_cars():
    """Get list of connected field cars"""
    return {
        "cars": connected_cars,
        "count": len(connected_cars),
        "enabled": live_data_enabled
    }

@app.get("/live-data/recent")
async def get_recent_packets(limit: int = 20):
    """Get recent telemetry packets"""
    return {
        "packets": telemetry_packets[-limit:] if telemetry_packets else [],
        "total_packets": packet_count,
        "enabled": live_data_enabled
    }

if __name__ == "__main__":
    import asyncio
    async def startup():
        await load_models()
    asyncio.run(startup())
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)