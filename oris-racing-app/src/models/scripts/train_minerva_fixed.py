#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn

# Fix the path to ensure imports work
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import directly
from minerva.minerva import MinervaV6Enhanced
from minerva.train_racing import MinervaRacingAdapter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🧠 Training MINERVA on {device}...")

# Create models
minerva = MinervaV6Enhanced()
minerva_racing = MinervaRacingAdapter(minerva)

# Move to device AFTER creating the adapter
minerva_racing = minerva_racing.to(device)

# Create optimizer
optimizer = torch.optim.AdamW(minerva_racing.parameters(), lr=1e-4)

os.makedirs('racing_models/checkpoints', exist_ok=True)

# Training loop with proper device handling
for epoch in range(5):
    data = {
        'track_position': 0.3,
        'speed': 150.0,
        'tire_degradation': {'FL': 0.7, 'FR': 0.72, 'RL': 0.68, 'RR': 0.69},
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

torch.save({'model_state_dict': minerva_racing.state_dict()}, 'racing_models/checkpoints/minerva_racing.pt')
print("✅ MINERVA training complete!")