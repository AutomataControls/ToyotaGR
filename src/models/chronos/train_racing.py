"""
CHRONOS Racing Training Script  
Adapts CHRONOS's temporal pattern mastery for lap time prediction and pace analysis
Uses sequence understanding to predict performance over time
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from chronos import ChronosV4Enhanced

class ChronosRacingAdapter(nn.Module):
    """Adapts CHRONOS's temporal reasoning for racing time analysis"""
    
    def __init__(self, base_model: ChronosV4Enhanced):
        super().__init__()
        self.chronos = base_model
        self.d_model = 256
        
        # Freeze base model initially
        for param in self.chronos.parameters():
            param.requires_grad = False
            
        # Lap time predictor
        self.lap_time_predictor = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # Next lap, +5 laps, +10 laps
        )
        
        # Sector performance analyzer
        self.sector_analyzer = nn.Sequential(
            nn.Linear(self.d_model * 3, 256),  # 3 sectors
            nn.ReLU(),
            nn.Linear(256, 9)  # 3 sectors x (time, optimal_time, degradation)
        )
        
        # Pace degradation model
        self.degradation_predictor = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20)  # Degradation curve for next 20 laps
        )
        
        # Weather impact analyzer
        self.weather_impact = nn.Sequential(
            nn.Linear(self.d_model * 2, 256),  # Current + forecast
            nn.ReLU(),
            nn.Linear(256, 5)  # Impact on pace over next 5 stints
        )
        
    def encode_lap_sequence_as_grid(self, timing_data: Dict) -> torch.Tensor:
        """
        Convert lap time sequences to temporal grids
        CHRONOS will recognize patterns in:
        - Lap time progressions
        - Sector splits evolution  
        - Degradation patterns
        - Stint performance curves
        """
        grid = torch.zeros(30, 30)
        
        # Recent lap times as horizontal progression
        lap_times = timing_data['lap_times'][-30:]  # Last 30 laps
        baseline = min(lap_times) if lap_times else 90.0
        
        for i, lap_time in enumerate(lap_times):
            # Normalize lap time to grid height
            normalized_time = (lap_time - baseline) / 5.0  # 5 second range
            height = int((1.0 - normalized_time) * 29)
            grid[max(0, height), i] = 1.0
            
            # Add intensity gradient for recency
            for j in range(max(0, height-2), min(30, height+3)):
                grid[j, i] = 0.3 * (1.0 - abs(j - height) / 3.0)
        
        # Sector times as separate bands
        if 'sector_times' in timing_data:
            sectors = timing_data['sector_times'][-30:]
            for i, sector_data in enumerate(sectors):
                for s, sector_time in enumerate(sector_data):
                    row = 10 + s * 5  # Separate bands for each sector
                    normalized = sector_time / timing_data['sector_best'][s]
                    grid[row:row+3, i] = normalized
                    
        # Tire age indicator
        if 'tire_age' in timing_data:
            tire_ages = timing_data['tire_age'][-30:]
            for i, age in enumerate(tire_ages):
                grid[28, i] = min(age / 50.0, 1.0)  # Normalize to 50 lap stint
                
        # Weather conditions
        if 'track_temp' in timing_data:
            temps = timing_data['track_temp'][-30:]
            for i, temp in enumerate(temps):
                grid[29, i] = (temp - 20) / 40.0  # Normalize 20-60°C range
                
        return grid
    
    def forward(self, timing_data: Dict) -> Dict[str, torch.Tensor]:
        # Encode timing sequences as temporal grid
        timing_grid = self.encode_lap_sequence_as_grid(timing_data)
        timing_grid = timing_grid.unsqueeze(0).unsqueeze(0)
        
        # Process through CHRONOS temporal transformer
        temporal_features = self.chronos.temporal_transformer(
            self.chronos.pattern_encoder(timing_grid)
        )
        
        # Global temporal understanding
        global_temporal = temporal_features.mean(dim=[2, 3])
        
        predictions = {
            'lap_times': self.lap_time_predictor(global_temporal),
            'pace_degradation': self.degradation_predictor(global_temporal)
        }
        
        # Sector analysis if available
        if 'sector_features' in timing_data:
            sector_features = []
            h = temporal_features.shape[2] // 3
            for i in range(3):
                sector_feat = temporal_features[:, :, i*h:(i+1)*h, :].mean(dim=[2, 3])
                sector_features.append(sector_feat)
            combined_sectors = torch.cat(sector_features, dim=1)
            predictions['sector_analysis'] = self.sector_analyzer(combined_sectors)
            
        # Weather impact if forecast available
        if 'weather_forecast' in timing_data:
            weather_features = torch.cat([
                global_temporal,
                timing_data['weather_forecast']
            ], dim=1)
            predictions['weather_impact'] = self.weather_impact(weather_features)
            
        return predictions


class TimingDataset(Dataset):
    """Dataset for lap timing and performance data"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.samples = []
        
        # Load all timing data files
        for track_dir in os.listdir(data_path):
            track_path = os.path.join(data_path, track_dir)
            if os.path.isdir(track_path):
                for csv_file in os.listdir(track_path):
                    if csv_file.endswith('.csv'):
                        self.samples.append(os.path.join(track_path, csv_file))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load timing data
        csv_path = self.samples[idx]
        df = pd.read_csv(csv_path)
        
        # Generate synthetic timing data
        base_time = 85.0 + np.random.rand() * 10  # Base lap time 85-95s
        
        timing_data = {
            'lap_times': [base_time + np.random.randn() * 2 + i * 0.1 for i in range(30)],
            'sector_times': [[28.5 + np.random.randn() * 0.5, 
                              29.0 + np.random.randn() * 0.5,
                              27.5 + np.random.randn() * 0.5] for _ in range(30)],
            'sector_best': [28.0, 28.5, 27.0],
            'tire_age': list(range(1, 31)),
            'track_temp': [25 + np.random.randn() * 5 for _ in range(30)],
            'weather_forecast': torch.randn(1, 256)  # Weather features
        }
        
        # Generate labels
        labels = {
            'next_lap_times': np.array([base_time + 1, base_time + 5, base_time + 10]),
            'sector_performance': np.random.rand(9),
            'degradation_curve': np.array([i * 0.05 for i in range(20)]),
            'weather_impact': np.random.rand(5)
        }
        
        return timing_data, labels


def train_chronos_racing(
    model_path: str = "chronos_v4_enhanced.pt",
    data_path: str = "../../data/tracks",  
    epochs: int = 80,  # Increased for better convergence
    batch_size: int = 512,  # A100 optimized batch size
    learning_rate: float = 1e-4
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load base CHRONOS model
    base_model = ChronosV4Enhanced()
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        base_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create racing adapter
    model = ChronosRacingAdapter(base_model).to(device)
    
    # Create dataset and dataloader
    dataset = TimingDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Train timing heads
    optimizer = torch.optim.AdamW([
        {'params': model.lap_time_predictor.parameters()},
        {'params': model.sector_analyzer.parameters()},
        {'params': model.degradation_predictor.parameters()},
        {'params': model.weather_impact.parameters()}
    ], lr=learning_rate)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Loss functions
    mse_loss = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, (timing_batch, labels_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Process batch through model with mixed precision
            with autocast():
                # Process each item in batch
                batch_predictions = []
                for i in range(len(timing_batch['lap_times'])):
                    timing_data = {
                        'lap_times': timing_batch['lap_times'][i],
                        'sector_times': timing_batch['sector_times'][i],
                        'sector_best': timing_batch['sector_best'][i],
                        'tire_age': timing_batch['tire_age'][i],
                        'track_temp': timing_batch['track_temp'][i],
                        'weather_forecast': timing_batch['weather_forecast'][i].to(device),
                        'sector_features': True  # Flag to enable sector analysis
                    }
                    
                    predictions = model(timing_data)
                    batch_predictions.append(predictions)
                
                # Compute losses
                lap_time_loss = mse_loss(
                    torch.cat([p['lap_times'] for p in batch_predictions]),
                    torch.tensor(labels_batch['next_lap_times']).float().to(device)
                )
                degradation_loss = mse_loss(
                    torch.cat([p['pace_degradation'] for p in batch_predictions]),
                    torch.tensor(labels_batch['degradation_curve']).float().to(device)
                )
                
                # Sector loss if available
                if 'sector_analysis' in batch_predictions[0]:
                    sector_loss = mse_loss(
                        torch.cat([p['sector_analysis'] for p in batch_predictions]),
                        torch.tensor(labels_batch['sector_performance']).float().to(device)
                    )
                else:
                    sector_loss = 0
                
                # Weather impact loss
                if 'weather_impact' in batch_predictions[0]:
                    weather_loss = mse_loss(
                        torch.cat([p['weather_impact'] for p in batch_predictions]),
                        torch.tensor(labels_batch['weather_impact']).float().to(device)
                    )
                else:
                    weather_loss = 0
                
                loss = lap_time_loss + degradation_loss + sector_loss + weather_loss
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
        
        # Fine-tune base model
        if epoch == epochs // 2:
            print("Unfreezing base model for fine-tuning...")
            for param in model.chronos.parameters():
                param.requires_grad = True
            optimizer.add_param_group({
                'params': model.chronos.parameters(),
                'lr': learning_rate * 0.1
            })
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'chronos_racing.pt')
    
    print("CHRONOS racing adapter training complete!")

if __name__ == "__main__":
    train_chronos_racing()