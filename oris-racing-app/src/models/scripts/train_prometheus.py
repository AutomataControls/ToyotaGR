#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prometheus.prometheus import PrometheusV6Enhanced
from prometheus.train_racing import PrometheusRacingAdapter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Training PROMETHEUS on {device}...")

prometheus = PrometheusV6Enhanced().to(device)
prometheus_racing = PrometheusRacingAdapter(prometheus).to(device)
optimizer = torch.optim.AdamW(prometheus_racing.parameters(), lr=1e-4)

os.makedirs('racing_models/checkpoints', exist_ok=True)

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

torch.save({'model_state_dict': prometheus_racing.state_dict()}, 'racing_models/checkpoints/prometheus_racing.pt')
print("✅ PROMETHEUS training complete!")