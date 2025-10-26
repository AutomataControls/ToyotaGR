#!/usr/bin/env python3
"""
Train all OLYMPUS racing models with one command
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all models and adapters
from minerva.minerva import MinervaV6Enhanced
from minerva.train_racing import MinervaRacingAdapter
from atlas.atlas import AtlasV5Enhanced
from atlas.train_racing import AtlasRacingAdapter
from iris.iris import IrisV6Enhanced
from iris.train_racing import IrisRacingAdapter
from chronos.chronos import ChronosV4Enhanced
from chronos.train_racing import ChronosRacingAdapter
from prometheus.prometheus import PrometheusV6Enhanced
from prometheus.train_racing import PrometheusRacingAdapter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🏁 Using device: {device}")

# Create directories
os.makedirs('racing_models/checkpoints', exist_ok=True)

# Train each model
print("\n" + "="*60)
print("🧠 Training MINERVA...")
minerva = MinervaV6Enhanced().to(device)
minerva_racing = MinervaRacingAdapter(minerva).to(device)
optimizer = torch.optim.AdamW(minerva_racing.parameters(), lr=1e-4)

for epoch in range(5):
    # Mock data
    data = {
        'track_position': 0.3,
        'speed': 150.0,
        'tire_degradation': {'fl': 0.7, 'fr': 0.72, 'rl': 0.68, 'rr': 0.69},
        'fuel_percentage': 0.6,
        'track_id': 0,
        'competitors': [{'position': 0.35}, {'position': 0.25}]
    }
    
    predictions = minerva_racing(data)
    loss = sum(pred.mean() for pred in predictions.values())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  Epoch {epoch+1}/5 - Loss: {loss.item():.4f}")

torch.save({
    'model_state_dict': minerva_racing.state_dict(),
}, 'racing_models/checkpoints/minerva_racing.pt')

print("\n" + "="*60)
print("🗺️ Training ATLAS...")
atlas = AtlasV5Enhanced().to(device)
atlas_racing = AtlasRacingAdapter(atlas).to(device)
optimizer = torch.optim.AdamW(atlas_racing.parameters(), lr=1e-4)

for epoch in range(5):
    data = {
        'position_on_track': 0.5,
        'lateral_position': 0.4,
        'racing_line': [{'lateral_position': 0.5} for _ in range(30)],
        'nearby_cars': [{'position': 0.48, 'lateral': 0.6}]
    }
    
    predictions = atlas_racing(data)
    loss = sum(pred.mean() for pred in predictions.values())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  Epoch {epoch+1}/5 - Loss: {loss.item():.4f}")

torch.save({
    'model_state_dict': atlas_racing.state_dict(),
}, 'racing_models/checkpoints/atlas_racing.pt')

print("\n" + "="*60)
print("👁️ Training IRIS...")
iris = IrisV6Enhanced().to(device)
iris_racing = IrisRacingAdapter(iris).to(device)
optimizer = torch.optim.AdamW(iris_racing.parameters(), lr=1e-4)

for epoch in range(5):
    data = {
        'tire_temps': {'fl': 90, 'fr': 92, 'rl': 88, 'rr': 89},
        'brake_temps': {'fl': 350, 'fr': 360, 'rl': 320, 'rr': 330},
        'g_force_history': {
            'lateral': [0.5, -0.3, 0.8, -0.6] * 7,
            'longitudinal': [-0.9, 0.2, -0.7, 0.4] * 7
        }
    }
    
    predictions = iris_racing(data)
    loss = sum(pred.mean() for pred in predictions.values())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  Epoch {epoch+1}/5 - Loss: {loss.item():.4f}")

torch.save({
    'model_state_dict': iris_racing.state_dict(),
}, 'racing_models/checkpoints/iris_racing.pt')

print("\n" + "="*60)
print("⏱️ Training CHRONOS...")
chronos = ChronosV4Enhanced().to(device)
chronos_racing = ChronosRacingAdapter(chronos).to(device)
optimizer = torch.optim.AdamW(chronos_racing.parameters(), lr=1e-4)

for epoch in range(5):
    import numpy as np
    data = {
        'lap_times': [83.5 + np.random.randn() * 0.5 for _ in range(30)],
        'sector_times': [[27.1, 28.3, 28.1] for _ in range(30)],
        'sector_best': [26.8, 28.0, 27.8],
        'tire_age': list(range(30)),
        'track_temp': [35 + np.random.randn() * 2 for _ in range(30)]
    }
    
    predictions = chronos_racing(data)
    loss = sum(pred.mean() for pred in predictions.values())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  Epoch {epoch+1}/5 - Loss: {loss.item():.4f}")

torch.save({
    'model_state_dict': chronos_racing.state_dict(),
}, 'racing_models/checkpoints/chronos_racing.pt')

print("\n" + "="*60)
print("🔥 Training PROMETHEUS...")
prometheus = PrometheusV6Enhanced().to(device)
prometheus_racing = PrometheusRacingAdapter(prometheus).to(device)
optimizer = torch.optim.AdamW(prometheus_racing.parameters(), lr=1e-4)

for epoch in range(5):
    data = {
        'standings': [
            {'gap_to_leader': i * 2.5, 'lap_trend': [0.1] * 10, 'pit_stops': 1, 'tire_age': 15}
            for i in range(20)
        ],
        'track_characteristics': {'elevation_profile': [50 + i*2 for i in range(30)]},
        'weather': {
            'temp_history': [28 + np.random.randn() for _ in range(30)],
            'rain_forecast': [0.1 + np.random.rand() * 0.2 for _ in range(30)]
        },
        'incident_history': [0] * 25 + [1, 0, 0, 1, 0]
    }
    
    predictions = prometheus_racing(data)
    loss = sum(pred.mean() for pred in predictions.values())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  Epoch {epoch+1}/5 - Loss: {loss.item():.4f}")

torch.save({
    'model_state_dict': prometheus_racing.state_dict(),
}, 'racing_models/checkpoints/prometheus_racing.pt')

print("\n" + "="*60)
print("🏛️ OLYMPUS RACING INTELLIGENCE TRAINING COMPLETE!")
print("✅ All models saved to racing_models/checkpoints/")