#!/usr/bin/env python3
import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.atlas import AtlasV5Enhanced
from atlas.train_racing import AtlasRacingAdapter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🗺️ Training ATLAS on {device}...")

atlas = AtlasV5Enhanced().to(device)
atlas_racing = AtlasRacingAdapter(atlas).to(device)
optimizer = torch.optim.AdamW(atlas_racing.parameters(), lr=1e-4)

os.makedirs('racing_models/checkpoints', exist_ok=True)

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

torch.save({'model_state_dict': atlas_racing.state_dict()}, 'racing_models/checkpoints/atlas_racing.pt')
print("✅ ATLAS training complete!")