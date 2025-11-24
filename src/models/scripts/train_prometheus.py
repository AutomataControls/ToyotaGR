"""
PROMETHEUS Training Script - Predictive Analytics
Toyota GR Cup Series Real-Time Analytics

Trains the PROMETHEUS model for predictive analytics and race outcome forecasting.
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
    from prometheus.prometheus import PrometheusRacingModel, create_prometheus_model
except ImportError:
    # Models already loaded via exec() in Colab
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToyotaGRPredictiveDataset(Dataset):
    """Dataset for Toyota GR Cup predictive analytics"""
    
    def __init__(self, telemetry_data: pd.DataFrame, sequence_length: int = 300):
        self.data = telemetry_data
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
        # Required telemetry columns for predictive analysis
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
        
        # Create predictive targets
        self.predictive_targets = self._create_predictive_targets()
        
    def _create_predictive_targets(self) -> np.ndarray:
        """Create predictive analytics targets from telemetry"""
        targets = []
        
        for i in range(len(self.features)):
            # Race position prediction (0-1, higher = better position)
            speed_performance = self.features[i, 0]
            consistency_factor = 1.0 - abs(np.random.normal(0, 0.1))
            race_position_prediction = np.clip(speed_performance * 0.01 + consistency_factor * 0.5, 0, 1)
            
            # Incident probability (0-1, higher = more likely)
            steering_aggression = abs(self.features[i, 5])
            brake_aggression = (self.features[i, 2] + self.features[i, 3]) / 2
            incident_probability = np.clip((steering_aggression * 0.02 + brake_aggression * 0.3) * 0.5, 0, 1)
            
            # Fuel consumption prediction (0-1, normalized)
            throttle = self.features[i, 1]
            speed = self.features[i, 0]
            fuel_consumption_prediction = np.clip(throttle * 0.7 + speed * 0.003, 0, 1)
            
            # Weather impact score (0-1)
            weather_impact_score = np.clip(0.5 + np.random.normal(0, 0.15), 0, 1)
            
            # Mechanical failure risk (0-1)
            stress_indicators = abs(self.features[i, 6]) + abs(self.features[i, 7])
            mechanical_failure_risk = np.clip(stress_indicators * 0.1 + 0.1, 0, 1)
            
            # Performance trend (0-1, 0.5 = stable)
            performance_trend = np.clip(0.5 + np.random.normal(0, 0.2), 0, 1)
            
            targets.append([
                race_position_prediction, incident_probability, fuel_consumption_prediction,
                weather_impact_score, mechanical_failure_risk, performance_trend
            ])
            
        return np.array(targets, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.features[idx:idx + self.sequence_length]
        target = self.predictive_targets[idx + self.sequence_length - 1]
        
        return torch.FloatTensor(sequence), torch.FloatTensor(target)


class PrometheusTrainer:
    """Trainer for PROMETHEUS predictive model"""
    
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
        
        for sequences, targets in tqdm(self.train_loader, desc="Training PROMETHEUS"):
            sequences, targets = sequences.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            
            # Calculate loss based on model outputs
            loss = 0.0
            loss_count = 0
            
            if 'race_position_prediction' in outputs:
                loss += self.criterion(outputs['race_position_prediction'].squeeze(), targets[:, 0])
                loss_count += 1
            
            if 'incident_probability' in outputs:
                loss += self.criterion(outputs['incident_probability'].squeeze(), targets[:, 1])
                loss_count += 1
                
            if 'fuel_consumption_prediction' in outputs:
                loss += self.criterion(outputs['fuel_consumption_prediction'].squeeze(), targets[:, 2])
                loss_count += 1
                
            if 'weather_impact_score' in outputs:
                loss += self.criterion(outputs['weather_impact_score'].squeeze(), targets[:, 3])
                loss_count += 1
                
            if 'mechanical_failure_risk' in outputs:
                loss += self.criterion(outputs['mechanical_failure_risk'].squeeze(), targets[:, 4])
                loss_count += 1
                
            if 'performance_trend' in outputs:
                loss += self.criterion(outputs['performance_trend'].squeeze(), targets[:, 5])
                loss_count += 1
                
            if loss_count > 0:
                loss = loss / loss_count
            else:
                # Fallback to global predictive state
                if 'global_predictive_state' in outputs:
                    predictive_mean = torch.mean(outputs['global_predictive_state'], dim=1)
                    target_mean = torch.mean(targets, dim=1)
                    loss = self.criterion(predictive_mean, target_mean)
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
                
                if 'race_position_prediction' in outputs:
                    loss += self.criterion(outputs['race_position_prediction'].squeeze(), targets[:, 0])
                    loss_count += 1
                
                if 'incident_probability' in outputs:
                    loss += self.criterion(outputs['incident_probability'].squeeze(), targets[:, 1])
                    loss_count += 1
                    
                if 'fuel_consumption_prediction' in outputs:
                    loss += self.criterion(outputs['fuel_consumption_prediction'].squeeze(), targets[:, 2])
                    loss_count += 1
                    
                if 'weather_impact_score' in outputs:
                    loss += self.criterion(outputs['weather_impact_score'].squeeze(), targets[:, 3])
                    loss_count += 1
                    
                if 'mechanical_failure_risk' in outputs:
                    loss += self.criterion(outputs['mechanical_failure_risk'].squeeze(), targets[:, 4])
                    loss_count += 1
                    
                if 'performance_trend' in outputs:
                    loss += self.criterion(outputs['performance_trend'].squeeze(), targets[:, 5])
                    loss_count += 1
                    
                if loss_count > 0:
                    loss = loss / loss_count
                elif 'global_predictive_state' in outputs:
                    predictive_mean = torch.mean(outputs['global_predictive_state'], dim=1)
                    target_mean = torch.mean(targets, dim=1)
                    loss = self.criterion(predictive_mean, target_mean)
                else:
                    continue
                
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs: int = 100, save_dir: str = "./BestModels"):
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, "prometheus_best.pt")
        
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