#!/usr/bin/env python3
"""
ATLAS V5 Enhanced Racing Training Script
Adapted from the MINERVA V6 training harness for robust training.
Focuses on training ATLAS for spatial reasoning in racing scenarios.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
from typing import Dict, Optional, Any
import math

# Fix the path to ensure imports work from sibling directories
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the ATLAS model
from atlas.atlas import AtlasV5Enhanced

# IMPORTANT: We reuse the Dataset and Collate Function from the MINERVA training script
try:
    from train_minerva import MinervaRacingDataset, TrainingConfig, custom_collate_fn
except ImportError as e:
    print("Error: Could not import from train_minerva.py.")
    print("Please ensure train_minerva.py is in the same directory as train_atlas.py.")
    print(f"Original error: {e}")
    sys.exit(1)


# =====================================
# ATLAS Racing Adapter
# =====================================

class AtlasRacingAdapter(nn.Module):
    """
    Adapter to connect the spatial reasoning of ATLAS V5
    to specific racing decision-making tasks.
    """
    def __init__(self, base_model: AtlasV5Enhanced):
        super().__init__()
        self.atlas = base_model
        
        # ATLAS d_model is 128
        feature_dim = self.atlas.d_model
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Define prediction heads similar to MINERVA for consistent training targets
        # Pit stop decision (binary)
        self.pit_decision_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, 2)  # 0: No Pit, 1: Pit
        )
        
        # Tire choice (3 compounds)
        self.tire_choice_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, 3)  # 0: Soft, 1: Medium, 2: Hard
        )
        
        # Overtake decision (binary)
        self.overtake_decision_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, 2)  # 0: Don't Overtake, 1: Overtake
        )
        
        # Fuel mode (3 modes)
        self.fuel_mode_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, 3)  # 0: Save, 1: Normal, 2: Push
        )
        
        # ERS deployment (binary)
        self.ers_deploy_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, 2)  # 0: No Deploy, 1: Deploy
        )
        
        # DRS usage (binary)
        self.drs_use_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, 2)  # 0: No DRS, 1: Use DRS
        )
        
        # Defense mode (binary)
        self.defense_mode_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, 2)  # 0: No Defense, 1: Defend
        )
        
        # Push level (3 levels)
        self.push_level_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, 3)  # 0: Conservative, 1: Normal, 2: Aggressive
        )
        
        # Tire temperature management (binary)
        self.tire_temp_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, 2)  # 0: Cool tires, 1: Push tires
        )
        
        # Strategy call (binary)
        self.strategy_call_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, 2)  # 0: Stick to plan, 1: Change strategy
        )

    def forward(self, grid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Processes a grid representation of the race state.
        """
        # Get ATLAS's spatial analysis of the grid
        # Ensure correct input shape for ATLAS (B, C, H, W)
        if grid.dim() == 3:
            grid = grid.unsqueeze(1)  # Add channel dimension
            
        atlas_output = self.atlas(grid)
        
        # Extract global spatial features by averaging over the grid dimensions
        # atlas_output['spatial_features'] has shape (B, H, W, D)
        spatial_features = atlas_output['spatial_features']
        
        # Average pooling over spatial dimensions to get global features
        global_features = spatial_features.mean(dim=[1, 2])  # (B, D)
        
        # Make predictions using the task-specific heads
        predictions = {
            'pit_decision': self.pit_decision_head(global_features),
            'tire_choice': self.tire_choice_head(global_features),
            'overtake': self.overtake_decision_head(global_features),
            'fuel_mode': self.fuel_mode_head(global_features),
            'ers_deploy': self.ers_deploy_head(global_features),
            'drs_use': self.drs_use_head(global_features),
            'defense_mode': self.defense_mode_head(global_features),
            'push_level': self.push_level_head(global_features),
            'tire_temp': self.tire_temp_head(global_features),
            'strategy_call': self.strategy_call_head(global_features)
        }
        
        return predictions


# =====================================
# ATLAS Trainer
# =====================================

class AtlasTrainer:
    """A robust trainer for the ATLAS model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🗺️ Initializing ATLAS V5 Enhanced Racing Trainer on {self.device}...")
        
        # Check for GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU: {torch.cuda.get_device_name(0)} ({total_memory:.1f} GB)")

        # 1. Initialize Model
        self.atlas_base = AtlasV5Enhanced().to(self.device)
        self.model = AtlasRacingAdapter(self.atlas_base).to(self.device)
        
        # Load checkpoint if exists
        checkpoint_path = os.path.join(config.checkpoint_dir, 'atlas_racing_best.pt')
        if os.path.exists(checkpoint_path):
            print(f"Loading existing checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("✓ Loaded model weights")
            if 'best_accuracy' in checkpoint:
                self.best_val_accuracy = checkpoint['best_accuracy']
                print(f"✓ Resuming from best accuracy: {self.best_val_accuracy:.2f}%")
            else:
                self.best_val_accuracy = 0.0
        else:
            self.best_val_accuracy = 0.0
        
        # 2. Initialize Datasets and DataLoaders
        self._initialize_datasets()
        
        # 3. Initialize Optimizer and Loss functions
        # Use higher learning rates for ATLAS (5x MINERVA's rates)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-5)
        
        # Loss functions for different output types
        self.criterion_binary = nn.CrossEntropyLoss()
        self.criterion_ternary = nn.CrossEntropyLoss()
        
        # Training parameters
        self.epochs = 30  # ATLAS typically converges faster than MINERVA
        self.early_stopping_patience = 5
        self.epochs_without_improvement = 0
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

    def _initialize_datasets(self):
        """Initialize training and validation datasets using the shared class."""
        print("Loading racing datasets for ATLAS...")
        self.train_dataset = MinervaRacingDataset(self.config, stage="train", augment=True, augment_factor=5)
        self.val_dataset = MinervaRacingDataset(self.config, stage="val", augment=False)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=custom_collate_fn
        )
        
        if len(self.val_dataset) > 0:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size * 2,
                shuffle=False,
                collate_fn=custom_collate_fn
            )
        else:
            self.val_loader = None
        
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")

    def _compute_loss(self, predictions: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes a weighted loss across all available prediction heads."""
        total_loss = 0.0
        loss_count = 0
        
        # Binary classifications
        binary_heads = ['pit_decision', 'overtake', 'ers_deploy', 'drs_use', 
                       'defense_mode', 'tire_temp', 'strategy_call']
        
        for head in binary_heads:
            if head in predictions and head in labels:
                # Ensure targets are long type
                targets = labels[head].long()
                total_loss += self.criterion_binary(predictions[head], targets)
                loss_count += 1
        
        # Ternary classifications
        ternary_heads = ['tire_choice', 'fuel_mode', 'push_level']
        
        for head in ternary_heads:
            if head in predictions and head in labels:
                # Ensure targets are long type
                targets = labels[head].long()
                total_loss += self.criterion_ternary(predictions[head], targets)
                loss_count += 1
        
        # Average the losses
        if loss_count > 0:
            total_loss = total_loss / loss_count
        
        return total_loss

    def train(self):
        """Main training loop."""
        print("\n🚀 Starting ATLAS training...")
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            
            for batch_idx, batch in enumerate(pbar):
                grids = batch['grid'].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
                
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(grids)
                
                # Compute loss
                loss = self._compute_loss(predictions, labels)
                
                # Skip batch if loss is NaN
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected at batch {batch_idx}. Skipping batch.")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                
                # Calculate accuracy for binary predictions
                for head in ['pit_decision', 'overtake']:
                    if head in predictions and head in labels:
                        preds = predictions[head].argmax(dim=1)
                        targets = labels[head].long()
                        train_correct += (preds == targets).sum().item()
                        train_total += targets.size(0)
                
                # Update progress bar
                if train_total > 0:
                    train_acc = (train_correct / train_total) * 100
                    pbar.set_postfix({
                        'loss': f"{train_loss / (batch_idx + 1):.4f}",
                        'acc': f"{train_acc:.1f}%"
                    })
            
            # Validation phase
            val_accuracy = self.validate()
            avg_train_loss = train_loss / len(self.train_loader)
            
            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%")
            
            # Save best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.epochs_without_improvement = 0
                print(f"  📈 New best validation accuracy: {self.best_val_accuracy:.2f}%. Saving model...")
                
                save_path = os.path.join(self.config.checkpoint_dir, 'atlas_racing_best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_accuracy': self.best_val_accuracy,
                    'config': self.config
                }, save_path)
            else:
                self.epochs_without_improvement += 1
                
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"⏰ Early stopping triggered after {epoch+1} epochs")
                print(f"   No improvement for {self.early_stopping_patience} epochs")
                break
            
            # Save latest checkpoint
            save_path = os.path.join(self.config.checkpoint_dir, 'atlas_racing_latest.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_accuracy': self.best_val_accuracy,
                'config': self.config
            }, save_path)

        print(f"\n✅ ATLAS training complete!")
        print(f"   Best validation accuracy: {self.best_val_accuracy:.2f}%")

    def validate(self) -> float:
        """Validation loop, returns average accuracy."""
        if not self.val_loader:
            return 0.0
        
        self.model.eval()
        val_correct = {}
        val_total = {}
        scenario_accuracies = {}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="           [Val]", leave=False)
            
            for batch in pbar:
                grids = batch['grid'].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
                scenario_types = batch.get('scenario_type', ['unknown'] * grids.size(0))
                
                # Forward pass
                predictions = self.model(grids)
                
                # Track accuracy for each prediction head and scenario
                for head in predictions.keys():
                    if head in labels:
                        preds = predictions[head].argmax(dim=1)
                        targets = labels[head].long()
                        
                        # Overall accuracy
                        if head not in val_correct:
                            val_correct[head] = 0
                            val_total[head] = 0
                        
                        val_correct[head] += (preds == targets).sum().item()
                        val_total[head] += targets.size(0)
                        
                        # Per-scenario accuracy
                        for i, scenario in enumerate(scenario_types):
                            if scenario not in scenario_accuracies:
                                scenario_accuracies[scenario] = {'correct': 0, 'total': 0}
                            
                            if preds[i] == targets[i]:
                                scenario_accuracies[scenario]['correct'] += 1
                            scenario_accuracies[scenario]['total'] += 1
        
        # Calculate overall accuracy
        total_correct = sum(val_correct.values())
        total_samples = sum(val_total.values())
        
        if total_samples == 0:
            return 0.0
        
        overall_accuracy = (total_correct / total_samples) * 100
        
        # Print per-scenario accuracies
        print("\n  Scenario accuracies:")
        for scenario, stats in scenario_accuracies.items():
            if stats['total'] > 0:
                acc = (stats['correct'] / stats['total']) * 100
                print(f"    {scenario}: {acc:.1f}% ({stats['correct']}/{stats['total']})")
        
        return overall_accuracy


# =====================================
# Main Training Entry Point
# =====================================

def main():
    """Main function to run the training."""
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Use deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Use the same config as MINERVA for data paths, etc.
    config = TrainingConfig()
    
    # Modify config for ATLAS training
    config.model_name = "ATLAS V5 Enhanced"
    
    # Create trainer and start training
    trainer = AtlasTrainer(config)
    trainer.train()
    
    print(f"\n🏁 Training complete!")
    print(f"Find best model at: {config.checkpoint_dir}/atlas_racing_best.pt")
    print(f"Find latest model at: {config.checkpoint_dir}/atlas_racing_latest.pt")


if __name__ == "__main__":
    main()