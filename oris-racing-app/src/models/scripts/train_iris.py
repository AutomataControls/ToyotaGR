#!/usr/bin/env python3
import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iris.iris import IrisV6Enhanced
from iris.train_racing import IrisRacingAdapter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"👁️ Training IRIS on {device}...")

iris = IrisV6Enhanced().to(device)
iris_racing = IrisRacingAdapter(iris).to(device)
optimizer = torch.optim.AdamW(iris_racing.parameters(), lr=1e-4)

os.makedirs('racing_models/checkpoints', exist_ok=True)

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

torch.save({'model_state_dict': iris_racing.state_dict()}, 'racing_models/checkpoints/iris_racing.pt')
print("✅ IRIS training complete!")