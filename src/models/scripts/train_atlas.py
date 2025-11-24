"""
ATLAS Training Script - Spatial Track Intelligence
Toyota GR Cup Series Real-Time Analytics

Trains the ATLAS model for spatial track intelligence and racing line optimization.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import StandardScaler

# Models are imported via exec() in Colab - no need for additional imports
try:
    from atlas.atlas import AtlasRacingModel, create_atlas_model
except ImportError:
    # Models already loaded via exec() in Colab
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToyotaGRSpatialDataset(Dataset):
    """Dataset for Toyota GR Cup spatial track intelligence"""
    
    def __init__(self, telemetry_data: pd.DataFrame, sequence_length: int = 300):
        self.data = telemetry_data
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
        # Required telemetry columns for spatial analysis
        self.feature_columns = ['speed', 'ath', 'pbrake_f', 'pbrake_r', 'gear', 
                              'Steering_Angle', 'accx_can', 'accy_can']
        
        # Validate columns exist
        missing_cols = [col for col in self.feature_columns if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Prepare features
        self.features = self.scaler.fit_transform(
            self.data[self.feature_columns].fillna(0)
        )
        
        # Create spatial targets
        self.spatial_targets = self._create_spatial_targets()
        
    def _create_spatial_targets(self) -> np.ndarray:
        """Create spatial intelligence targets from telemetry"""
        targets = []
        
        for i in range(len(self.features)):
            # Racing line quality (0-1)
            speed = self.features[i, 0]
            steering = abs(self.features[i, 5])
            racing_line_quality = np.clip(speed * 0.8 - steering * 0.2, 0, 1)
            
            # Track position score (0-1)
            lateral_g = abs(self.features[i, 6])
            track_position_score = np.clip(1.0 - lateral_g * 0.3, 0, 1)
            
            # Corner performance (0-1)
            brake_pressure = (self.features[i, 2] + self.features[i, 3]) / 2
            corner_performance = np.clip(brake_pressure + 0.3, 0, 1)
            
            # Overtaking opportunity (0-1)
            overtaking_opportunity = np.clip(0.5 + np.random.normal(0, 0.1), 0, 1)
            
            # Track limits safety (0-1, higher = safer)
            track_limits_safety = np.clip(1.0 - (steering * 0.1 + lateral_g * 0.2), 0, 1)
            
            # Spatial efficiency (0-1)
            spatial_efficiency = (racing_line_quality + track_position_score) / 2
            
            targets.append([
                racing_line_quality, track_position_score, corner_performance,
                overtaking_opportunity, track_limits_safety, spatial_efficiency
            ])
            
        return np.array(targets, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.features[idx:idx + self.sequence_length]
        target = self.spatial_targets[idx + self.sequence_length - 1]
        
        return torch.FloatTensor(sequence), torch.FloatTensor(target)


class AtlasTrainer:
    """Trainer for ATLAS spatial racing model"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda', lr=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        
        for sequences, targets in tqdm(self.train_loader, desc="Training ATLAS"):
            sequences, targets = sequences.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            
            # Calculate loss based on model outputs
            loss = 0.0
            loss_count = 0
            
            if 'racing_line_quality' in outputs:
                loss += self.criterion(outputs['racing_line_quality'].squeeze(), targets[:, 0])
                loss_count += 1
            
            if 'overtaking_opportunity' in outputs:
                loss += self.criterion(outputs['overtaking_opportunity'].squeeze(), targets[:, 3])
                loss_count += 1
                
            if 'track_limits_warning' in outputs:
                # Convert safety to warning (inverse)
                warning_targets = 1.0 - targets[:, 4]
                loss += self.criterion(outputs['track_limits_warning'].squeeze(), warning_targets)
                loss_count += 1
                
            if 'spatial_advantage' in outputs:
                loss += self.criterion(outputs['spatial_advantage'].squeeze(), targets[:, 5])
                loss_count += 1
                
            if loss_count > 0:
                loss = loss / loss_count
            else:
                # Fallback to global spatial state
                if 'global_spatial_state' in outputs:
                    spatial_mean = torch.mean(outputs['global_spatial_state'], dim=1)
                    target_mean = torch.mean(targets, dim=1)
                    loss = self.criterion(spatial_mean, target_mean)
                else:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for sequences, targets in self.val_loader:
                sequences, targets = sequences.to(self.device), targets.to(self.device)
                outputs = self.model(sequences)
                
                # Same loss calculation as training
                loss = 0.0
                loss_count = 0
                
                if 'racing_line_quality' in outputs:
                    loss += self.criterion(outputs['racing_line_quality'].squeeze(), targets[:, 0])
                    loss_count += 1
                
                if 'overtaking_opportunity' in outputs:
                    loss += self.criterion(outputs['overtaking_opportunity'].squeeze(), targets[:, 3])
                    loss_count += 1
                    
                if 'track_limits_warning' in outputs:
                    warning_targets = 1.0 - targets[:, 4]
                    loss += self.criterion(outputs['track_limits_warning'].squeeze(), warning_targets)
                    loss_count += 1
                    
                if 'spatial_advantage' in outputs:
                    loss += self.criterion(outputs['spatial_advantage'].squeeze(), targets[:, 5])
                    loss_count += 1
                    
                if loss_count > 0:
                    loss = loss / loss_count
                elif 'global_spatial_state' in outputs:
                    spatial_mean = torch.mean(outputs['global_spatial_state'], dim=1)
                    target_mean = torch.mean(targets, dim=1)
                    loss = self.criterion(spatial_mean, target_mean)
                else:
                    continue
                
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs: int = 100, save_dir: str = "./BestModels"):
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, "atlas_best.pt")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step()
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss
                }, best_model_path)
                
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train={train_loss:.6f}, Val={val_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_model_path': best_model_path
        }


def load_telemetry_data(data_path: str) -> pd.DataFrame:
    """Load Toyota GR telemetry data"""
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.parquet'):
        return pd.read_parquet(data_path)
    else:
        raise ValueError("Unsupported format. Use CSV or Parquet.")


# Note: Training functions are defined above
# The notebook handles data loading and model training directly using the classes