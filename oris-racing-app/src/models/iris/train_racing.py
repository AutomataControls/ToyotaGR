"""
IRIS Racing Training Script
Adapts IRIS's object understanding for vehicle dynamics and telemetry analysis
Leverages pattern detection for tire/brake/engine optimization
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from iris import IrisV6Enhanced

class IrisRacingAdapter(nn.Module):
    """Adapts IRIS's object detection for vehicle telemetry analysis"""
    
    def __init__(self, base_model: IrisV6Enhanced):
        super().__init__()
        self.iris = base_model
        self.d_model = 256
        
        # Freeze base model initially
        for param in self.iris.parameters():
            param.requires_grad = False
            
        # Tire degradation analyzer
        self.tire_analyzer = nn.Sequential(
            nn.Linear(self.d_model * 4, 512),  # 4 tires
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 16)  # 4 tires x (temp, wear, pressure, grip)
        )
        
        # Brake system monitor
        self.brake_analyzer = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 8)  # 4 brakes x (temp, wear)
        )
        
        # Engine performance optimizer
        self.engine_analyzer = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # Oil temp, water temp, fuel mix, ers mode, power
        )
        
        # Setup recommendation head
        self.setup_optimizer = nn.Sequential(
            nn.Linear(self.d_model * 3, 512),  # Combined telemetry features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # Wing, suspension, diff, brake bias, etc.
        )
        
    def encode_telemetry_as_grid(self, telemetry: Dict) -> torch.Tensor:
        """
        Convert telemetry streams to 30x30 grid patterns
        IRIS will recognize patterns in:
        - Tire temperature distributions
        - Brake heat patterns
        - G-force traces
        - Suspension travel patterns
        """
        grid = torch.zeros(30, 30)
        
        # Tire temperature heatmap (corners of grid)
        tire_temps = telemetry['tire_temps']  # FL, FR, RL, RR
        # Front left - top left quadrant
        grid[0:15, 0:15] = tire_temps['fl'] / 150.0
        # Front right - top right quadrant
        grid[0:15, 15:30] = tire_temps['fr'] / 150.0
        # Rear left - bottom left quadrant
        grid[15:30, 0:15] = tire_temps['rl'] / 150.0
        # Rear right - bottom right quadrant
        grid[15:30, 15:30] = tire_temps['rr'] / 150.0
        
        # Brake temperatures as intensity gradients
        brake_temps = telemetry['brake_temps']
        for i, (brake, temp) in enumerate(brake_temps.items()):
            row = 7 + (i // 2) * 15  # Place in middle of each quadrant
            col = 7 + (i % 2) * 15
            grid[row-2:row+3, col-2:col+3] = temp / 1000.0
            
        # G-force trace in center
        if 'g_force_history' in telemetry:
            g_lat = telemetry['g_force_history']['lateral'][-30:]
            g_lon = telemetry['g_force_history']['longitudinal'][-30:]
            for i, (lat, lon) in enumerate(zip(g_lat, g_lon)):
                x = 15 + int(lat * 5)  # Scale to grid
                y = 15 + int(lon * 5)
                if 0 <= x < 30 and 0 <= y < 30:
                    grid[y, x] = 0.7
                    
        return grid
    
    def forward(self, telemetry: Dict) -> Dict[str, torch.Tensor]:
        # Encode telemetry as visual patterns
        telemetry_grid = self.encode_telemetry_as_grid(telemetry)
        telemetry_grid = telemetry_grid.unsqueeze(0).unsqueeze(0)
        
        # Process through IRIS's object detection
        # IRIS will identify "objects" which are actually telemetry patterns
        object_features = self.iris.high_res_object_detector(
            self.iris.feature_extractor(telemetry_grid)
        )
        
        # Extract features for different systems
        global_features = object_features.mean(dim=[2, 3])
        
        # Separate features for each tire quadrant
        tire_features = []
        h, w = object_features.shape[2:]
        for i in range(2):
            for j in range(2):
                quadrant = object_features[:, :, 
                    i*h//2:(i+1)*h//2,
                    j*w//2:(j+1)*w//2
                ].mean(dim=[2, 3])
                tire_features.append(quadrant)
        
        combined_tire_features = torch.cat(tire_features, dim=1)
        
        predictions = {
            'tire_analysis': self.tire_analyzer(combined_tire_features),
            'brake_status': self.brake_analyzer(global_features),
            'engine_params': self.engine_analyzer(global_features)
        }
        
        # Setup optimization using all telemetry
        all_features = torch.cat([
            combined_tire_features,
            global_features,
            predictions['engine_params']
        ], dim=1)
        predictions['setup_recommendations'] = self.setup_optimizer(all_features)
        
        return predictions


class VehicleTelemetryDataset(Dataset):
    """Dataset for vehicle telemetry and dynamics data"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.samples = []
        
        # Load all telemetry files
        for track_dir in os.listdir(data_path):
            track_path = os.path.join(data_path, track_dir)
            if os.path.isdir(track_path):
                for csv_file in os.listdir(track_path):
                    if csv_file.endswith('.csv'):
                        self.samples.append(os.path.join(track_path, csv_file))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load telemetry data
        csv_path = self.samples[idx]
        df = pd.read_csv(csv_path)
        
        # Extract vehicle telemetry
        telemetry = {
            'tire_temps': {
                'fl': np.random.randint(80, 130),
                'fr': np.random.randint(80, 130),
                'rl': np.random.randint(80, 130),
                'rr': np.random.randint(80, 130)
            },
            'brake_temps': {
                'fl': np.random.randint(200, 800),
                'fr': np.random.randint(200, 800),
                'rl': np.random.randint(200, 800),
                'rr': np.random.randint(200, 800)
            },
            'g_force_history': {
                'lateral': np.random.randn(30) * 2,
                'longitudinal': np.random.randn(30) * 2
            },
            'speed': np.random.randint(100, 350),
            'rpm': np.random.randint(8000, 15000),
            'gear': np.random.randint(1, 8)
        }
        
        # Generate target values
        labels = {
            'tire_status': np.random.rand(16),  # 4 tires x 4 params
            'brake_status': np.random.rand(8),   # 4 brakes x 2 params
            'engine_status': np.random.rand(5),  # 5 engine params
            'setup_params': np.random.rand(10)   # 10 setup parameters
        }
        
        return telemetry, labels


def train_iris_racing(
    model_path: str = "iris_v6_enhanced.pt",
    data_path: str = "../../data/tracks",
    epochs: int = 90,  # Increased for better convergence
    batch_size: int = 512,  # A100 optimized batch size
    learning_rate: float = 1e-4
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load base IRIS model
    base_model = IrisV6Enhanced()
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        base_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create racing adapter
    model = IrisRacingAdapter(base_model).to(device)
    
    # Create dataset and dataloader
    dataset = VehicleTelemetryDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Train vehicle dynamics heads
    optimizer = torch.optim.AdamW([
        {'params': model.tire_analyzer.parameters()},
        {'params': model.brake_analyzer.parameters()},
        {'params': model.engine_analyzer.parameters()},
        {'params': model.setup_optimizer.parameters()}
    ], lr=learning_rate)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Loss function for continuous predictions
    mse_loss = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, (telemetry_batch, labels_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Process batch through model with mixed precision
            with autocast():
                # Process each item in batch
                batch_predictions = []
                for i in range(len(telemetry_batch['speed'])):
                    telemetry = {
                        'tire_temps': {
                            'fl': telemetry_batch['tire_temps']['fl'][i].item(),
                            'fr': telemetry_batch['tire_temps']['fr'][i].item(),
                            'rl': telemetry_batch['tire_temps']['rl'][i].item(),
                            'rr': telemetry_batch['tire_temps']['rr'][i].item()
                        },
                        'brake_temps': {
                            'fl': telemetry_batch['brake_temps']['fl'][i].item(),
                            'fr': telemetry_batch['brake_temps']['fr'][i].item(),
                            'rl': telemetry_batch['brake_temps']['rl'][i].item(),
                            'rr': telemetry_batch['brake_temps']['rr'][i].item()
                        },
                        'g_force_history': {
                            'lateral': telemetry_batch['g_force_history']['lateral'][i].numpy(),
                            'longitudinal': telemetry_batch['g_force_history']['longitudinal'][i].numpy()
                        }
                    }
                    
                    predictions = model(telemetry)
                    batch_predictions.append(predictions)
                
                # Compute losses
                tire_loss = mse_loss(
                    torch.cat([p['tire_analysis'] for p in batch_predictions]),
                    torch.tensor(labels_batch['tire_status']).float().to(device)
                )
                brake_loss = mse_loss(
                    torch.cat([p['brake_status'] for p in batch_predictions]),
                    torch.tensor(labels_batch['brake_status']).float().to(device)
                )
                engine_loss = mse_loss(
                    torch.cat([p['engine_params'] for p in batch_predictions]),
                    torch.tensor(labels_batch['engine_status']).float().to(device)
                )
                setup_loss = mse_loss(
                    torch.cat([p['setup_recommendations'] for p in batch_predictions]),
                    torch.tensor(labels_batch['setup_params']).float().to(device)
                )
                
                loss = tire_loss + brake_loss + engine_loss + setup_loss
            
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
        
        # Fine-tune base model later
        if epoch == epochs // 2:
            print("Unfreezing base model for fine-tuning...")
            for param in model.iris.parameters():
                param.requires_grad = True
            optimizer.add_param_group({
                'params': model.iris.parameters(),
                'lr': learning_rate * 0.1
            })
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'iris_racing.pt')
    
    print("IRIS racing adapter training complete!")

if __name__ == "__main__":
    train_iris_racing()