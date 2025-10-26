# TODO: Fix Model Training Issues

Based on the issues encountered with MINERVA training, the following fixes need to be applied to the other models:

## 1. Fix Module Imports (ALL MODELS)
- [ ] **ATLAS**: Remove `from src.models.atlas_model import EnhancedAtlasNet` and define class inline
- [ ] **IRIS**: Remove `from src.models.iris_model import EnhancedIrisNet` and define class inline  
- [ ] **CHRONOS**: Remove `from src.models.chronos_v4_enhanced import ChronosV4Enhanced` and define class inline
- [ ] **PROMETHEUS**: Remove `from src.models.prometheus_model import EnhancedPrometheusNet` and define class inline

## 2. Fix Training Scripts Import Paths
- [ ] Update `scripts/train_atlas.py` to use the same import pattern as fixed minerva script
- [ ] Update `scripts/train_iris.py` to use the same import pattern
- [ ] Update `scripts/train_chronos.py` to use the same import pattern  
- [ ] Update `scripts/train_prometheus.py` to use the same import pattern

## 3. Fix Device Mismatches in train_racing.py files
Each model's train_racing.py likely has tensor device issues. Need to ensure all tensors are on the same device:
- [ ] **ATLAS**: Fix any `torch.tensor()` calls to include `.to(next(self.parameters()).device)`
- [ ] **IRIS**: Fix tensor device placement
- [ ] **CHRONOS**: Fix tensor device placement
- [ ] **PROMETHEUS**: Fix tensor device placement

## 4. Fix Model Architecture Mismatches
Each model's training adapter is likely calling attributes that don't exist:
- [ ] **ATLAS**: Check that train_racing.py only uses attributes that actually exist in AtlasV5Enhanced
- [ ] **IRIS**: Check that train_racing.py matches IrisV6Enhanced structure
- [ ] **CHRONOS**: Check that train_racing.py matches ChronosV4Enhanced structure
- [ ] **PROMETHEUS**: Check that train_racing.py matches PrometheusV6Enhanced structure

## 5. Fix Tensor Shape Mismatches
Models may expect different tensor shapes than what's being passed:
- [ ] **ATLAS**: Ensure input tensors match expected dimensions (likely 4D for conv layers)
- [ ] **IRIS**: Fix tensor dimensions for model forward passes
- [ ] **CHRONOS**: Fix temporal sequence dimensions
- [ ] **PROMETHEUS**: Fix input tensor shapes

## 6. Create Simple Test Script
- [ ] Create a `scripts/test_models.py` that tests basic forward pass for each model
- [ ] This will help identify issues before full training

## Notes
- The main issue is that the models were created with references to non-existent base classes
- The training adapters assume model attributes that may not exist
- All tensor operations need explicit device placement for CUDA compatibility
- Input shapes need to match what the model architectures expect

## Priority
1. Fix imports first (can't run anything without this)
2. Fix device issues (runtime errors)
3. Fix attribute mismatches (runtime errors)
4. Fix tensor shapes (runtime errors)
5. Create test script for validation