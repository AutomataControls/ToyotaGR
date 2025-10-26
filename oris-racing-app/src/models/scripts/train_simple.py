#!/usr/bin/env python3
"""
Simple MINERVA Racing Training Script
Focuses on getting basic training working without complex features
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from minerva.train_racing import MinervaRacingAdapter
from minerva.minerva import MinervaV6Enhanced
from scripts.train_minerva import RacingTelemetryDataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model - just the racing adapter
    base_model = MinervaV6Enhanced()
    model = MinervaRacingAdapter(base_model).to(device)
    
    # Freeze base model completely
    for param in model.minerva.parameters():
        param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create datasets
    train_dataset = RacingTelemetryDataset(
        data_dir="/content/ToyotaGR/oris-racing-app/src/data/tracks",
        split='train',
        max_samples_per_track=20
    )
    
    val_dataset = RacingTelemetryDataset(
        data_dir="/content/ToyotaGR/oris-racing-app/src/data/tracks",
        split='val',
        max_samples_per_track=5
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Simple optimizer - only train the heads
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    
    # Loss functions
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    # Training loop
    num_epochs = 20
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            race_data_list, labels, indices, scenarios = batch
            
            optimizer.zero_grad()
            
            # Process each sample
            batch_loss = 0.0
            for i in range(len(indices)):
                race_data = {
                    'track_position': race_data_list['track_position'][i].item(),
                    'speed': race_data_list['speed'][i].item(),
                    'rpm': race_data_list['rpm'][i].item(),
                    'gear': race_data_list['gear'][i].item(),
                    'throttle': race_data_list['throttle'][i].item(),
                    'brake': race_data_list['brake'][i].item(),
                    'steer_angle': race_data_list['steer_angle'][i].item(),
                    'lat_g': race_data_list['lat_g'][i].item(),
                    'long_g': race_data_list['long_g'][i].item(),
                    'tire_degradation': {
                        'FL': race_data_list['tire_fl'][i].item(),
                        'FR': race_data_list['tire_fr'][i].item(),
                        'RL': race_data_list['tire_rl'][i].item(),
                        'RR': race_data_list['tire_rr'][i].item()
                    },
                    'fuel_percentage': race_data_list['fuel'][i].item(),
                    'weather_grid': np.random.rand(5, 30),
                    'competitors': [],
                    'track_id': race_data_list['track_id'][i].item()
                }
                
                # Get predictions
                with torch.cuda.amp.autocast(enabled=False):  # Disable AMP for stability
                    predictions = model(race_data)
                
                # Compute loss for pit strategy only (most stable)
                if 'pit_decision' in labels:
                    target = labels['pit_decision'][i].long().to(device)
                    loss = ce_loss(predictions['pit_strategy'].unsqueeze(0), target.unsqueeze(0))
                    batch_loss += loss
                    
                    # Track accuracy
                    pred_class = predictions['pit_strategy'].argmax().item()
                    if pred_class == target.item():
                        train_correct += 1
                    train_total += 1
            
            # Average loss
            if train_total > 0:
                batch_loss = batch_loss / len(indices)
                batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                train_loss += batch_loss.item()
            
            pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                race_data_list, labels, indices, scenarios = batch
                
                for i in range(len(indices)):
                    race_data = {
                        'track_position': race_data_list['track_position'][i].item(),
                        'speed': race_data_list['speed'][i].item(),
                        'rpm': race_data_list['rpm'][i].item(),
                        'gear': race_data_list['gear'][i].item(),
                        'throttle': race_data_list['throttle'][i].item(),
                        'brake': race_data_list['brake'][i].item(),
                        'steer_angle': race_data_list['steer_angle'][i].item(),
                        'lat_g': race_data_list['lat_g'][i].item(),
                        'long_g': race_data_list['long_g'][i].item(),
                        'tire_degradation': {
                            'FL': race_data_list['tire_fl'][i].item(),
                            'FR': race_data_list['tire_fr'][i].item(),
                            'RL': race_data_list['tire_rl'][i].item(),
                            'RR': race_data_list['tire_rr'][i].item()
                        },
                        'fuel_percentage': race_data_list['fuel'][i].item(),
                        'weather_grid': np.random.rand(5, 30),
                        'competitors': [],
                        'track_id': race_data_list['track_id'][i].item()
                    }
                    
                    predictions = model(race_data)
                    
                    if 'pit_decision' in labels:
                        target = labels['pit_decision'][i].long()
                        pred_class = predictions['pit_strategy'].argmax().item()
                        if pred_class == target.item():
                            val_correct += 1
                        val_total += 1
        
        # Calculate accuracies
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        print(f"\nEpoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'minerva_racing_simple_best.pt')
            print(f"Saved new best model with {val_acc:.2f}% accuracy")
    
    print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()