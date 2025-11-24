"""
OLYMPUS Ensemble Racing Training Script
Combines all 5 specialists for unified racing intelligence
Orchestrates MINERVA + ATLAS + IRIS + CHRONOS + PROMETHEUS
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import sys
import gc
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minerva.train_racing import MinervaRacingAdapter
from atlas.train_racing import AtlasRacingAdapter
from iris.train_racing import IrisRacingAdapter
from chronos.train_racing import ChronosRacingAdapter
from prometheus.train_racing import PrometheusRacingAdapter

class OLYMPUSRacingEnsemble(nn.Module):
    """Complete OLYMPUS ensemble for racing intelligence"""
    
    def __init__(self, specialist_models: Dict[str, nn.Module]):
        super().__init__()
        
        # Load all specialist adapters
        self.minerva = specialist_models['minerva']
        self.atlas = specialist_models['atlas']
        self.iris = specialist_models['iris']
        self.chronos = specialist_models['chronos']
        self.prometheus = specialist_models['prometheus']
        
        # Ensemble fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(256 * 5, 1024),  # 5 specialists * 256 features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Confidence estimators for each specialist
        self.confidence_heads = nn.ModuleDict({
            'minerva': nn.Linear(256, 1),
            'atlas': nn.Linear(256, 1),
            'iris': nn.Linear(256, 1),
            'chronos': nn.Linear(256, 1),
            'prometheus': nn.Linear(512, 1)  # Prometheus has larger features
        })
        
        # Final decision heads
        self.decision_network = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Unified output heads
        self.pit_decision = nn.Linear(256, 4)  # Now, +1, +2, +3 laps
        self.strategy_confidence = nn.Linear(256, 1)
        self.risk_assessment = nn.Linear(256, 5)  # Risk levels 1-5
        
    def forward(self, race_data: Dict) -> Dict[str, torch.Tensor]:
        # Each specialist processes the data in their domain
        
        # MINERVA - Strategic analysis
        minerva_out = self.minerva(race_data)
        minerva_features = self.extract_features(minerva_out)
        minerva_conf = torch.sigmoid(self.confidence_heads['minerva'](minerva_features))
        
        # ATLAS - Track positioning
        atlas_out = self.atlas(race_data['telemetry'])
        atlas_features = self.extract_features(atlas_out)
        atlas_conf = torch.sigmoid(self.confidence_heads['atlas'](atlas_features))
        
        # IRIS - Vehicle dynamics
        iris_out = self.iris(race_data['telemetry'])
        iris_features = self.extract_features(iris_out)
        iris_conf = torch.sigmoid(self.confidence_heads['iris'](iris_features))
        
        # CHRONOS - Timing analysis
        chronos_out = self.chronos(race_data['timing'])
        chronos_features = self.extract_features(chronos_out)
        chronos_conf = torch.sigmoid(self.confidence_heads['chronos'](chronos_features))
        
        # PROMETHEUS - Predictive modeling
        prometheus_out = self.prometheus(race_data)
        prometheus_features = self.extract_features(prometheus_out, size=512)
        prometheus_conf = torch.sigmoid(self.confidence_heads['prometheus'](prometheus_features))
        
        # Weighted fusion based on confidence
        all_features = torch.cat([
            minerva_features * minerva_conf,
            atlas_features * atlas_conf,
            iris_features * iris_conf,
            chronos_features * chronos_conf,
            prometheus_features[:, :256] * prometheus_conf  # Align dimensions
        ], dim=1)
        
        # Fuse all specialist insights
        fused_features = self.fusion_network(all_features)
        
        # Generate final decisions
        decision_features = self.decision_network(fused_features)
        
        # Compile all outputs
        ensemble_output = {
            # Individual specialist outputs
            'minerva': minerva_out,
            'atlas': atlas_out,
            'iris': iris_out,
            'chronos': chronos_out,
            'prometheus': prometheus_out,
            
            # Confidence scores
            'specialist_confidence': {
                'minerva': minerva_conf,
                'atlas': atlas_conf,
                'iris': iris_conf,
                'chronos': chronos_conf,
                'prometheus': prometheus_conf
            },
            
            # Ensemble decisions
            'pit_strategy': self.pit_decision(decision_features),
            'strategy_confidence': torch.sigmoid(self.strategy_confidence(decision_features)),
            'risk_assessment': torch.softmax(self.risk_assessment(decision_features), dim=-1),
            
            # Consensus features for downstream use
            'ensemble_features': fused_features
        }
        
        return ensemble_output
    
    def extract_features(self, specialist_output: Dict, size: int = 256) -> torch.Tensor:
        """Extract feature representation from specialist output"""
        # Simple extraction - can be made more sophisticated
        if isinstance(specialist_output, dict):
            # Average all tensor outputs
            features = []
            for v in specialist_output.values():
                if isinstance(v, torch.Tensor) and v.numel() > 0:
                    features.append(v.view(v.size(0), -1).mean(dim=1, keepdim=True))
            if features:
                return torch.cat(features, dim=1).mean(dim=1, keepdim=True).expand(-1, size)
        return torch.zeros(1, size)

class EnsembleRacingDataset(Dataset):
    """Dataset for ensemble racing training"""
    def __init__(self, data_path: str, num_samples: int = 10000):
        self.data_path = data_path
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Generate synthetic race data for training
        # In production, load real race data from CSV files
        race_data = {
            'track_position': torch.rand(1) * 1.0,
            'speed': torch.rand(1) * 200 + 100,  # 100-300 km/h
            'tire_degradation': {
                'fl': torch.rand(1) * 0.5 + 0.5,  # 50-100%
                'fr': torch.rand(1) * 0.5 + 0.5,
                'rl': torch.rand(1) * 0.5 + 0.5,
                'rr': torch.rand(1) * 0.5 + 0.5
            },
            'fuel_percentage': torch.rand(1) * 0.8 + 0.2,  # 20-100%
            'track_id': torch.randint(0, 6, (1,)),
            'competitors': [{'position': torch.rand(1)} for _ in range(5)],
            'telemetry': {
                'position_on_track': torch.rand(1),
                'lateral_position': torch.rand(1) * 0.8 + 0.1,
                'tire_temps': {
                    'fl': torch.rand(1) * 40 + 70,  # 70-110°C
                    'fr': torch.rand(1) * 40 + 70,
                    'rl': torch.rand(1) * 40 + 70,
                    'rr': torch.rand(1) * 40 + 70
                },
                'brake_temps': {
                    'fl': torch.rand(1) * 200 + 300,  # 300-500°C
                    'fr': torch.rand(1) * 200 + 300,
                    'rl': torch.rand(1) * 200 + 300,
                    'rr': torch.rand(1) * 200 + 300
                },
                'g_force_history': {
                    'lateral': torch.randn(30) * 0.5,
                    'longitudinal': torch.randn(30) * 0.7
                }
            },
            'timing': {
                'lap_times': torch.randn(30) * 2 + 85,  # ~85s lap times
                'sector_times': torch.randn(30, 3) * 1 + torch.tensor([27.0, 28.0, 30.0]),
                'sector_best': torch.tensor([26.8, 27.8, 29.8]),
                'tire_age': torch.arange(30, dtype=torch.float32),
                'track_temp': torch.randn(30) * 5 + 35  # ~35°C
            },
            'standings': [
                {
                    'gap_to_leader': i * 2.5,
                    'lap_trend': torch.randn(10) * 0.2,
                    'pit_stops': torch.randint(0, 3, (1,)).item(),
                    'tire_age': torch.randint(0, 30, (1,)).item()
                }
                for i in range(20)
            ]
        }
        return race_data

def train_olympus_ensemble(
    epochs: int = 60,  # Increased for A100
    batch_size: int = 256,  # Optimized for ensemble
    learning_rate: float = 5e-5
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load all specialist racing adapters
    print("Loading specialist models...")
    
    specialists = {}
    
    # Enable mixed precision for A100
    scaler = GradScaler()
    
    # Load each trained racing specialist
    for model_name in ['minerva', 'atlas', 'iris', 'chronos', 'prometheus']:
        model_path = f"../{model_name}/{model_name}_racing.pt"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            # Initialize the appropriate adapter class
            if model_name == 'minerva':
                from minerva.minerva import MinervaV6Enhanced
                base = MinervaV6Enhanced()
                model = MinervaRacingAdapter(base)
            elif model_name == 'atlas':
                from atlas.atlas import AtlasV5Enhanced
                base = AtlasV5Enhanced()
                model = AtlasRacingAdapter(base)
            elif model_name == 'iris':
                from iris.iris import IrisV6Enhanced
                base = IrisV6Enhanced()
                model = IrisRacingAdapter(base)
            elif model_name == 'chronos':
                from chronos.chronos import ChronosV4Enhanced
                base = ChronosV4Enhanced()
                model = ChronosRacingAdapter(base)
            elif model_name == 'prometheus':
                from prometheus.prometheus import PrometheusV6Enhanced
                base = PrometheusV6Enhanced()
                model = PrometheusRacingAdapter(base)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            specialists[model_name] = model
            print(f"Loaded {model_name} racing adapter")
    
    # Create ensemble
    ensemble = OLYMPUSRacingEnsemble(specialists).to(device)
    
    # Only train fusion and decision networks
    fusion_params = list(ensemble.fusion_network.parameters()) + \
                   list(ensemble.confidence_heads.parameters()) + \
                   list(ensemble.decision_network.parameters()) + \
                   list(ensemble.pit_decision.parameters()) + \
                   list(ensemble.strategy_confidence.parameters()) + \
                   list(ensemble.risk_assessment.parameters())
    
    optimizer = optim.AdamW(fusion_params, lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss functions
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    # Create dataset and dataloader
    dataset = EnsembleRacingDataset(data_path, num_samples=50000)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Training OLYMPUS ensemble fusion on {len(dataset)} samples...")
    print(f"Batch size: {batch_size}, Total batches: {len(dataloader)}")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        ensemble.train()
        epoch_loss = 0.0
        batch_count = 0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch_idx, race_data_batch in enumerate(pbar):
                # Move batch data to device
                # Note: In production, implement proper batch collation
                
                with autocast():
                    # Forward pass through ensemble
                    ensemble_output = ensemble(race_data_batch)
                    
                    # Calculate multi-task loss
                    pit_loss = ce_loss(
                        ensemble_output['pit_strategy'],
                        torch.randint(0, 4, (batch_size,)).to(device)
                    )
                    
                    confidence_loss = bce_loss(
                        ensemble_output['strategy_confidence'].squeeze(),
                        torch.rand(batch_size).to(device)
                    )
                    
                    risk_loss = ce_loss(
                        ensemble_output['risk_assessment'],
                        torch.randint(0, 5, (batch_size,)).to(device)
                    )
                    
                    # Specialist agreement loss
                    specialist_confs = list(ensemble_output['specialist_confidence'].values())
                    agreement_loss = sum(
                        mse_loss(conf, specialist_confs[0]) 
                        for conf in specialist_confs[1:]
                    ) / (len(specialist_confs) - 1)
                    
                    total_loss = pit_loss + confidence_loss + risk_loss + 0.1 * agreement_loss
                
                # Backward pass with mixed precision
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fusion_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += total_loss.item()
                batch_count += 1
                
                # Update progress bar
                if batch_idx % 10 == 0:
                    avg_loss = epoch_loss / batch_count
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'pit': f'{pit_loss.item():.4f}',
                        'conf': f'{confidence_loss.item():.4f}',
                        'risk': f'{risk_loss.item():.4f}'
                    })
        
        # Step scheduler
        scheduler.step()
        
        # Print epoch summary
        avg_epoch_loss = epoch_loss / batch_count
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'model_state_dict': ensemble.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_loss': best_loss
            }, 'olympus_racing_ensemble_best.pt')
            print(f"  -> New best model saved!")
        
        # Memory cleanup
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
    # Save final ensemble
    torch.save({
        'model_state_dict': ensemble.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epochs_trained': epochs,
        'final_loss': best_loss
    }, 'olympus_racing_ensemble_final.pt')
    
    print(f"\nOLYMPUS ensemble training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Models saved: olympus_racing_ensemble_best.pt, olympus_racing_ensemble_final.pt")

if __name__ == "__main__":
    train_olympus_ensemble()