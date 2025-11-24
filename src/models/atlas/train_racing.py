"""
ATLAS Racing Training Script
Adapts ATLAS's spatial pattern recognition for track positioning and overtaking
Uses grid mastery to analyze racing lines and positioning strategies
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from .atlas import AtlasV5Enhanced

class AtlasRacingAdapter(nn.Module):
    """Adapts ATLAS's spatial intelligence for track positioning"""
    
    def __init__(self, base_model: AtlasV5Enhanced):
        super().__init__()
        self.atlas = base_model
        self.d_model = 256
        
        # Freeze base model initially
        for param in self.atlas.parameters():
            param.requires_grad = False
            
        # Racing line prediction head
        self.racing_line_head = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 30)  # 30 points defining optimal line
        )
        
        # Overtaking opportunity detector
        self.overtaking_head = nn.Sequential(
            nn.Linear(self.d_model * 2, 256),  # Current + target car features
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # Inside, outside, late brake, slipstream, wait
        )
        
        # Defensive positioning predictor
        self.defense_head = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # Block inside, outside, weave, normal
        )
        
        # Track section analyzer (for corner types)
        self.corner_analyzer = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # Hairpin, chicane, fast, medium, slow, straight
        )
        
    def encode_track_section_as_grid(self, telemetry: Dict) -> torch.Tensor:
        """
        Convert track section to 30x30 grid for ATLAS
        - Racing line visualization
        - Car positions and trajectories
        - Track limits and boundaries
        - Grip levels/track conditions
        """
        grid = torch.zeros(30, 30)
        
        # Track boundaries (edges of grid)
        grid[0, :] = 1.0  # Left boundary
        grid[29, :] = 1.0  # Right boundary
        
        # Current racing line
        if 'racing_line' in telemetry:
            line_points = telemetry['racing_line']
            for i, point in enumerate(line_points[:30]):
                y_pos = int(point['lateral_position'] * 29)
                grid[y_pos, i] = 0.5
                
        # Current car position
        car_x = int(telemetry['position_on_track'] * 29)
        car_y = int(telemetry['lateral_position'] * 29)
        grid[max(0, car_y-1):min(30, car_y+2), 
             max(0, car_x-1):min(30, car_x+2)] = 0.8
        
        # Other cars in proximity
        for car in telemetry.get('nearby_cars', []):
            other_x = int(car['position'] * 29)
            other_y = int(car['lateral'] * 29)
            grid[other_y, other_x] = 1.0
            
        # Track condition/grip levels
        if 'grip_map' in telemetry:
            grip_layer = torch.tensor(telemetry['grip_map'][:30, :30])
            grid = grid + grip_layer * 0.3
            
        return grid
    
    def forward(self, telemetry: Dict) -> Dict[str, torch.Tensor]:
        # Encode current track section
        track_grid = self.encode_track_section_as_grid(telemetry)
        track_grid = track_grid.unsqueeze(0).unsqueeze(0)
        
        # Process through ATLAS spatial transformer
        spatial_features = self.atlas.fast_spatial_transformer(
            self.atlas.object_encoder(track_grid)
        )
        
        # Global features for current position
        global_features = spatial_features.mean(dim=[1, 2])
        
        predictions = {
            'racing_line': self.racing_line_head(global_features),
            'corner_type': self.corner_analyzer(global_features),
            'defensive_move': self.defense_head(global_features)
        }
        
        # Overtaking analysis if another car is nearby
        if 'target_car_features' in telemetry:
            combined_features = torch.cat([
                global_features,
                telemetry['target_car_features']
            ], dim=-1)
            predictions['overtaking_opportunity'] = self.overtaking_head(combined_features)
            
        return predictions


class TelemetryDataset(Dataset):
    """Dataset for track telemetry and positioning data"""
    
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
        
        # Extract telemetry state
        telemetry = {
            'position_on_track': np.random.rand(),
            'lateral_position': np.random.rand(),
            'racing_line': [{'lateral_position': np.random.rand()} for _ in range(30)],
            'nearby_cars': [{
                'position': np.random.rand(),
                'lateral': np.random.rand()
            } for _ in range(3)],
            'grip_map': np.random.rand(30, 30),
            'speed': np.random.randint(150, 350),
            'gear': np.random.randint(1, 8)
        }
        
        # Generate labels
        labels = {
            'optimal_line': np.random.rand(30),
            'corner_type': np.random.randint(0, 6),
            'defensive_move': np.random.randint(0, 4),
            'overtaking_decision': np.random.randint(0, 5)
        }
        
        return telemetry, labels


def train_atlas_racing(
    model_path: str = "atlas_v5_enhanced.pt",
    data_path: str = "../../data/tracks",
    epochs: int = 80,  # Increased for better convergence
    batch_size: int = 512,  # A100 optimized batch size
    learning_rate: float = 1e-4
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load base ATLAS model
    base_model = AtlasV5Enhanced()
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        base_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create racing adapter
    model = AtlasRacingAdapter(base_model).to(device)
    
    # Create dataset and dataloader
    dataset = TelemetryDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Train new heads first
    optimizer = torch.optim.AdamW([
        {'params': model.racing_line_head.parameters()},
        {'params': model.overtaking_head.parameters()},
        {'params': model.defense_head.parameters()},
        {'params': model.corner_analyzer.parameters()}
    ], lr=learning_rate)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Loss functions
    line_loss_fn = nn.MSELoss()  # For racing line prediction
    class_loss_fn = nn.CrossEntropyLoss()  # For classifications
    
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
                for i in range(len(telemetry_batch['position_on_track'])):
                    telemetry = {
                        'position_on_track': telemetry_batch['position_on_track'][i].item(),
                        'lateral_position': telemetry_batch['lateral_position'][i].item(),
                        'racing_line': [{'lateral_position': pos.item()} for pos in telemetry_batch['racing_line'][i]],
                        'nearby_cars': [
                            {
                                'position': telemetry_batch['nearby_cars'][i][j]['position'].item(),
                                'lateral': telemetry_batch['nearby_cars'][i][j]['lateral'].item()
                            } for j in range(len(telemetry_batch['nearby_cars'][i]))
                        ],
                        'grip_map': telemetry_batch['grip_map'][i].numpy()
                    }
                    
                    # Add target car features for overtaking analysis
                    if np.random.rand() > 0.5:  # 50% chance of overtaking scenario
                        telemetry['target_car_features'] = torch.randn(1, 256).to(device)
                    
                    predictions = model(telemetry)
                    batch_predictions.append(predictions)
                
                # Compute losses
                line_loss = line_loss_fn(
                    torch.cat([p['racing_line'] for p in batch_predictions]),
                    torch.tensor(labels_batch['optimal_line']).float().to(device)
                )
                corner_loss = class_loss_fn(
                    torch.cat([p['corner_type'] for p in batch_predictions]),
                    torch.tensor(labels_batch['corner_type']).to(device)
                )
                defense_loss = class_loss_fn(
                    torch.cat([p['defensive_move'] for p in batch_predictions]),
                    torch.tensor(labels_batch['defensive_move']).to(device)
                )
                
                # Add overtaking loss if applicable
                overtaking_preds = [p.get('overtaking_opportunity') for p in batch_predictions if 'overtaking_opportunity' in p]
                if overtaking_preds:
                    overtaking_loss = class_loss_fn(
                        torch.cat(overtaking_preds),
                        torch.tensor(labels_batch['overtaking_decision'][:len(overtaking_preds)]).to(device)
                    )
                else:
                    overtaking_loss = 0
                
                loss = line_loss + corner_loss + defense_loss + overtaking_loss
            
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
        
        # Unfreeze base model halfway through
        if epoch == epochs // 2:
            print("Unfreezing base model for fine-tuning...")
            for param in model.atlas.parameters():
                param.requires_grad = True
            optimizer.add_param_group({
                'params': model.atlas.parameters(),
                'lr': learning_rate * 0.1
            })
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'atlas_racing.pt')
    
    print("ATLAS racing adapter training complete!")

if __name__ == "__main__":
    train_atlas_racing()