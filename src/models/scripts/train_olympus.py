"""
OLYMPUS Training Script - Ensemble Racing Intelligence
Toyota GR Cup Series Real-Time Analytics

Trains the OLYMPUS ensemble model that combines all specialist racing models.
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
    from ensemble.olympus_ensemble import OlympusEnsembleModel, create_olympus_ensemble
except ImportError:
    # Models already loaded via exec() in Colab
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToyotaGROlympusDataset(Dataset):
    """Dataset for Toyota GR Cup OLYMPUS ensemble training"""
    
    def __init__(self, telemetry_data: pd.DataFrame, sequence_length: int = 300):
        self.data = telemetry_data
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
        # Required telemetry columns for ensemble analysis
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
        
        # Create comprehensive targets for ensemble training
        self.ensemble_targets = self._create_ensemble_targets()
        
    def _create_ensemble_targets(self) -> np.ndarray:
        """Create comprehensive targets combining all specialist model outputs"""
        targets = []
        
        for i in range(len(self.features)):
            # ATLAS targets (spatial intelligence)
            speed = self.features[i, 0]
            steering = abs(self.features[i, 5])
            racing_line_quality = np.clip(speed * 0.8 - steering * 0.2, 0, 1)
            
            # IRIS targets (vehicle dynamics)
            lateral_g = abs(self.features[i, 6])
            suspension_performance = np.clip(1.0 - lateral_g * 0.3, 0, 1)
            
            # CHRONOS targets (timing)
            throttle = self.features[i, 1]
            sector_time_prediction = np.clip(1.0 - (speed * throttle) * 0.01, 0, 1)
            
            # PROMETHEUS targets (predictive)
            consistency_factor = 1.0 - abs(np.random.normal(0, 0.1))
            race_position_prediction = np.clip(speed * 0.01 + consistency_factor * 0.5, 0, 1)
            
            # MINERVA targets (strategic)
            lap_progress = (i % 1000) / 1000.0
            pit_strategy_recommendation = np.clip(min(lap_progress * 2, 1.0) * 0.6, 0, 1)
            
            # Overall performance score (ensemble target)
            performance_score = (racing_line_quality + suspension_performance + 
                               sector_time_prediction + race_position_prediction) / 4
            
            # Risk assessment (ensemble meta-target)
            brake_aggression = (self.features[i, 2] + self.features[i, 3]) / 2
            risk_assessment = np.clip((steering * 0.02 + brake_aggression * 0.3) * 0.5, 0, 1)
            
            # Confidence score (how confident the ensemble should be)
            confidence_score = np.clip(1.0 - np.random.normal(0.1, 0.05), 0.5, 1.0)
            
            targets.append([
                racing_line_quality,      # ATLAS
                suspension_performance,   # IRIS  
                sector_time_prediction,   # CHRONOS
                race_position_prediction, # PROMETHEUS
                pit_strategy_recommendation, # MINERVA
                performance_score,        # Overall performance
                risk_assessment,         # Risk level
                confidence_score         # Ensemble confidence
            ])
            
        return np.array(targets, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.features[idx:idx + self.sequence_length]
        target = self.ensemble_targets[idx + self.sequence_length - 1]
        
        return torch.FloatTensor(sequence), torch.FloatTensor(target)


class OlympusTrainer:
    """Trainer for OLYMPUS ensemble model"""
    
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
        
        for sequences, targets in tqdm(self.train_loader, desc="Training OLYMPUS"):
            sequences, targets = sequences.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            
            # Calculate comprehensive ensemble loss
            total_ensemble_loss = 0.0
            loss_components = 0
            
            # Individual specialist model losses
            if 'atlas_output' in outputs:
                atlas_loss = self.criterion(outputs['atlas_output'], targets[:, 0:1])
                total_ensemble_loss += atlas_loss
                loss_components += 1
                
            if 'iris_output' in outputs:
                iris_loss = self.criterion(outputs['iris_output'], targets[:, 1:2])
                total_ensemble_loss += iris_loss
                loss_components += 1
                
            if 'chronos_output' in outputs:
                chronos_loss = self.criterion(outputs['chronos_output'], targets[:, 2:3])
                total_ensemble_loss += chronos_loss
                loss_components += 1
                
            if 'prometheus_output' in outputs:
                prometheus_loss = self.criterion(outputs['prometheus_output'], targets[:, 3:4])
                total_ensemble_loss += prometheus_loss
                loss_components += 1
                
            if 'minerva_output' in outputs:
                minerva_loss = self.criterion(outputs['minerva_output'], targets[:, 4:5])
                total_ensemble_loss += minerva_loss
                loss_components += 1
            
            # Ensemble-specific outputs
            if 'overall_performance_score' in outputs:
                performance_loss = self.criterion(outputs['overall_performance_score'], targets[:, 5:6])
                total_ensemble_loss += performance_loss * 2  # Weight ensemble outputs more
                loss_components += 2
                
            if 'risk_assessment' in outputs:
                risk_loss = self.criterion(outputs['risk_assessment'], targets[:, 6:7])
                total_ensemble_loss += risk_loss * 1.5
                loss_components += 1.5
                
            if 'ensemble_confidence' in outputs:
                confidence_loss = self.criterion(outputs['ensemble_confidence'], targets[:, 7:8])
                total_ensemble_loss += confidence_loss
                loss_components += 1
            
            # Fallback to ensemble prediction if specific outputs not available
            if loss_components == 0 and 'ensemble_prediction' in outputs:
                # Use mean of all targets for ensemble prediction
                ensemble_target = torch.mean(targets, dim=1, keepdim=True)
                total_ensemble_loss = self.criterion(outputs['ensemble_prediction'], ensemble_target)
                loss_components = 1
            
            # Final loss calculation
            if loss_components > 0:
                final_loss = total_ensemble_loss / loss_components
            else:
                final_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += final_loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for sequences, targets in self.val_loader:
                sequences, targets = sequences.to(self.device), targets.to(self.device)
                outputs = self.model(sequences)
                
                # Same loss calculation as training
                total_ensemble_loss = 0.0
                loss_components = 0
                
                if 'atlas_output' in outputs:
                    total_ensemble_loss += self.criterion(outputs['atlas_output'], targets[:, 0:1])
                    loss_components += 1
                    
                if 'iris_output' in outputs:
                    total_ensemble_loss += self.criterion(outputs['iris_output'], targets[:, 1:2])
                    loss_components += 1
                    
                if 'chronos_output' in outputs:
                    total_ensemble_loss += self.criterion(outputs['chronos_output'], targets[:, 2:3])
                    loss_components += 1
                    
                if 'prometheus_output' in outputs:
                    total_ensemble_loss += self.criterion(outputs['prometheus_output'], targets[:, 3:4])
                    loss_components += 1
                    
                if 'minerva_output' in outputs:
                    total_ensemble_loss += self.criterion(outputs['minerva_output'], targets[:, 4:5])
                    loss_components += 1
                
                if 'overall_performance_score' in outputs:
                    total_ensemble_loss += self.criterion(outputs['overall_performance_score'], targets[:, 5:6]) * 2
                    loss_components += 2
                    
                if 'risk_assessment' in outputs:
                    total_ensemble_loss += self.criterion(outputs['risk_assessment'], targets[:, 6:7]) * 1.5
                    loss_components += 1.5
                    
                if 'ensemble_confidence' in outputs:
                    total_ensemble_loss += self.criterion(outputs['ensemble_confidence'], targets[:, 7:8])
                    loss_components += 1
                
                if loss_components == 0 and 'ensemble_prediction' in outputs:
                    ensemble_target = torch.mean(targets, dim=1, keepdim=True)
                    total_ensemble_loss = self.criterion(outputs['ensemble_prediction'], ensemble_target)
                    loss_components = 1
                
                if loss_components > 0:
                    final_loss = total_ensemble_loss / loss_components
                    total_loss += final_loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs: int = 100, save_dir: str = "./BestModels"):
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, "olympus_best.pt")
        
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