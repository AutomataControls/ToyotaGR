#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos.chronos import ChronosV4Enhanced
from chronos.train_racing import ChronosRacingAdapter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"⏱️ Training CHRONOS on {device}...")

chronos = ChronosV4Enhanced().to(device)
chronos_racing = ChronosRacingAdapter(chronos).to(device)
optimizer = torch.optim.AdamW(chronos_racing.parameters(), lr=1e-4)

os.makedirs('racing_models/checkpoints', exist_ok=True)

for epoch in range(5):
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

torch.save({'model_state_dict': chronos_racing.state_dict()}, 'racing_models/checkpoints/chronos_racing.pt')
print("✅ CHRONOS training complete!")