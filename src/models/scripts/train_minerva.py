"""
MINERVA Training Script - Strategic Racing Decisions
Toyota GR Cup Series Real-Time Analytics

Trains the MINERVA model for strategic racing decisions and tactical planning.
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
    from minerva.minerva import MinervaRacingModel, create_minerva_model
except ImportError:
    # Models already loaded via exec() in Colab
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToyotaGRStrategicDataset(Dataset):
    """Dataset for Toyota GR Cup strategic racing decisions"""
    
    def __init__(self, telemetry_data: pd.DataFrame, sequence_length: int = 300):
        self.data = telemetry_data
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
        # Required telemetry columns for strategic analysis
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
        
        # Create strategic targets
        self.strategic_targets = self._create_strategic_targets()
        
    def _create_strategic_targets(self) -> np.ndarray:
        """Create strategic decision targets from telemetry"""
        targets = []
        
        for i in range(len(self.features)):
            # Pit strategy recommendation (0-1, higher = more urgent pit need)
            lap_progress = (i % 1000) / 1000.0  # Simulate lap progress
            tire_wear_factor = min(lap_progress * 2, 1.0)
            fuel_level = max(1.0 - lap_progress * 1.5, 0)
            pit_strategy_recommendation = np.clip(tire_wear_factor * 0.6 + (1-fuel_level) * 0.4, 0, 1)
            
            # Overtaking strategy (0-1, higher = more aggressive)
            speed_advantage = self.features[i, 0]
            position_factor = np.random.uniform(0.3, 0.8)
            overtaking_strategy = np.clip(speed_advantage * 0.01 + position_factor * 0.5, 0, 1)
            
            # Defensive driving (0-1, higher = more defensive needed)
            proximity_pressure = np.random.uniform(0.2, 0.7)
            track_conditions = np.random.uniform(0.8, 1.0)
            defensive_driving = np.clip(proximity_pressure / track_conditions, 0, 1)
            
            # Tire management (0-1, higher = more conservation needed)
            current_pace = self.features[i, 0]
            stint_progress = min((i % 800) / 800.0, 1.0)
            tire_management = np.clip(stint_progress * 0.7 + (1 - current_pace * 0.01) * 0.3, 0, 1)
            
            # Fuel saving strategy (0-1, higher = more fuel saving needed)
            remaining_distance = max(1.0 - (i % 1200) / 1200.0, 0)
            throttle_usage = self.features[i, 1]
            fuel_saving_strategy = np.clip((1-remaining_distance) * 0.6 + throttle_usage * 0.4, 0, 1)
            
            # Track position priority (0-1, higher = prioritize track position)
            gap_management = np.random.uniform(0.4, 0.9)
            race_phase = min(i / 5000.0, 1.0)  # Early race = 0, late race = 1
            track_position_priority = np.clip(gap_management * 0.6 + race_phase * 0.4, 0, 1)
            
            targets.append([
                pit_strategy_recommendation, overtaking_strategy, defensive_driving,
                tire_management, fuel_saving_strategy, track_position_priority
            ])
            
        return np.array(targets, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.features[idx:idx + self.sequence_length]
        target = self.strategic_targets[idx + self.sequence_length - 1]
        
        return torch.FloatTensor(sequence), torch.FloatTensor(target)


class MinervaTrainer:
    """Trainer for MINERVA strategic racing model"""
    
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
        
        for sequences, targets in tqdm(self.train_loader, desc="Training MINERVA"):
            sequences, targets = sequences.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            
            # Calculate loss based on model outputs
            loss = 0.0
            loss_count = 0
            
            if 'pit_strategy_recommendation' in outputs:
                loss += self.criterion(outputs['pit_strategy_recommendation'].squeeze(), targets[:, 0])
                loss_count += 1
            
            if 'overtaking_strategy' in outputs:
                loss += self.criterion(outputs['overtaking_strategy'].squeeze(), targets[:, 1])
                loss_count += 1
                
            if 'defensive_driving' in outputs:
                loss += self.criterion(outputs['defensive_driving'].squeeze(), targets[:, 2])
                loss_count += 1
                
            if 'tire_management' in outputs:
                loss += self.criterion(outputs['tire_management'].squeeze(), targets[:, 3])
                loss_count += 1
                
            if 'fuel_saving_strategy' in outputs:
                loss += self.criterion(outputs['fuel_saving_strategy'].squeeze(), targets[:, 4])
                loss_count += 1
                
            if 'track_position_priority' in outputs:
                loss += self.criterion(outputs['track_position_priority'].squeeze(), targets[:, 5])
                loss_count += 1
                
            if loss_count > 0:
                loss = loss / loss_count
            else:
                # Fallback to global strategic state
                if 'global_strategic_state' in outputs:
                    strategic_mean = torch.mean(outputs['global_strategic_state'], dim=1)
                    target_mean = torch.mean(targets, dim=1)
                    loss = self.criterion(strategic_mean, target_mean)
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
                
                if 'pit_strategy_recommendation' in outputs:
                    loss += self.criterion(outputs['pit_strategy_recommendation'].squeeze(), targets[:, 0])
                    loss_count += 1
                
                if 'overtaking_strategy' in outputs:
                    loss += self.criterion(outputs['overtaking_strategy'].squeeze(), targets[:, 1])
                    loss_count += 1
                    
                if 'defensive_driving' in outputs:
                    loss += self.criterion(outputs['defensive_driving'].squeeze(), targets[:, 2])
                    loss_count += 1
                    
                if 'tire_management' in outputs:
                    loss += self.criterion(outputs['tire_management'].squeeze(), targets[:, 3])
                    loss_count += 1
                    
                if 'fuel_saving_strategy' in outputs:
                    loss += self.criterion(outputs['fuel_saving_strategy'].squeeze(), targets[:, 4])
                    loss_count += 1
                    
                if 'track_position_priority' in outputs:
                    loss += self.criterion(outputs['track_position_priority'].squeeze(), targets[:, 5])
                    loss_count += 1
                    
                if loss_count > 0:
                    loss = loss / loss_count
                elif 'global_strategic_state' in outputs:
                    strategic_mean = torch.mean(outputs['global_strategic_state'], dim=1)
                    target_mean = torch.mean(targets, dim=1)
                    loss = self.criterion(strategic_mean, target_mean)
                else:
                    continue
                
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs: int = 100, save_dir: str = "./BestModels"):
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, "minerva_best.pt")
        
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