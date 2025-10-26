#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn

# Fix the path to ensure imports work
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🧠 Training MINERVA (simplified) on {device}...")

# Create a simple racing adapter that works
class SimpleMinervaRacing(nn.Module):
    def __init__(self):
        super().__init__()
        self.pit_strategy_head = nn.Linear(256, 4)
        self.tire_strategy_head = nn.Linear(256, 3)
        self.fuel_optimization_head = nn.Linear(256, 1)
        self.feature_extractor = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
    
    def forward(self, race_data):
        # Create simple features from race data
        features = torch.tensor([
            race_data['track_position'],
            race_data['speed'] / 200.0,
            race_data['tire_degradation']['fl'],
            race_data['tire_degradation']['fr'],
            race_data['tire_degradation']['rl'],
            race_data['tire_degradation']['rr'],
            race_data['fuel_percentage'],
            race_data['track_id'] / 10.0,
            len(race_data['competitors']) / 20.0,
            0.5  # dummy feature
        ], device=device).unsqueeze(0)
        
        # Extract features
        features = self.feature_extractor(features)
        
        # Generate predictions
        return {
            'pit_strategy': self.pit_strategy_head(features),
            'tire_choice': self.tire_strategy_head(features),
            'fuel_save': torch.sigmoid(self.fuel_optimization_head(features))
        }

# Create and train model
model = SimpleMinervaRacing().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

os.makedirs('racing_models/checkpoints', exist_ok=True)

for epoch in range(5):
    data = {
        'track_position': 0.3,
        'speed': 150.0,
        'tire_degradation': {'fl': 0.7, 'fr': 0.72, 'rl': 0.68, 'rr': 0.69},
        'fuel_percentage': 0.6,
        'track_id': 0,
        'competitors': [{'position': 0.35}, {'position': 0.25}]
    }
    
    predictions = model(data)
    loss = sum(pred.mean() for pred in predictions.values())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  Epoch {epoch+1}/5 - Loss: {loss.item():.4f}")

torch.save({'model_state_dict': model.state_dict()}, 'racing_models/checkpoints/minerva_racing.pt')
print("✅ MINERVA training complete!")