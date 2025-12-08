# Phase 3 Training Notes

## Baseline Training Attempts

### Attempt 1: With Warmup (FAILED)
**Command**: `--warmup-epochs 5 --batch-size 4`  
**Started**: Dec 4, 2025 ~19:30  
**Completed**: Dec 5, 2025 ~01:03 (8 epochs, early stopped)

**Results**:
- Epoch 1: 40.61% val acc, 28.09% train acc ✅
- Epoch 2-8: **Divergence** - val acc dropped to 13.42%, loss increased
- Early stopping triggered after 7 epochs without improvement
- Best checkpoint: Epoch 1 (40.61% val acc)

**Root Cause**: Warmup scheduler + CosineAnnealing conflict
- Warmup increased LR from 0.0001 to 0.001 over 5 epochs
- After warmup, base LR (0.001) was too high for fine-tuning pretrained AST
- This caused the model to diverge and lose pretrained knowledge

**Key Finding**: Warmup is harmful for transfer learning with discriminative LR
- Backbone LR: 5e-5 (correct for fine-tuning)
- Head LR: 1e-3 (correct for new layers)
- Warmup forced both to increase, breaking the delicate balance

### Attempt 2: No Warmup (RUNNING)
**Command**: `--warmup-epochs 0 --batch-size 4`  
**Started**: Dec 5, 2025 ~01:06  
**Expected Results**: Stable convergence, >55% val acc

**Hypothesis**: 
- Without warmup, discriminative LR will work as intended
- Expected epoch 1: ~48-50% val acc (matching smoke test)
- Expected epoch 10-15: ~60-65% val acc
- Expected final: ~65-70% val acc

**Configuration**:
- Model: AST (86.2M params, AudioSet pretrained)
- Features: Log-Mel Spectrograms (128, 173)
- Loss: FocalLoss (γ=2.0, α=None)
- Optimizer: AdamW (backbone_lr=5e-5, head_lr=1e-3, weight_decay=1e-2)
- Scheduler: CosineAnnealingLR (no warmup)
- Batch size: 4 (GPU constraint)
- Max epochs: 50 (early stopping patience=7)

## Lessons Learned

1. **Warmup is not universal**: While warmup helps training from scratch, it can harm transfer learning
2. **Discriminative LR is fragile**: Different LR for backbone vs head requires careful scheduling
3. **Monitor epoch 1 carefully**: First epoch performance is the best indicator of correct LR
4. **Smoke tests are critical**: The smoke test (48.62% in 1 epoch) was our baseline for comparison

## Next Steps

1. Wait for v2 training to complete (~6-8 hours)
2. If v2 shows stable convergence:
   - Mark T049 complete
   - Evaluate on test set (T050-T051)
   - Proceed to Phase 5 (augmentation training)
3. If v2 still diverges:
   - Reduce head LR to 5e-4 (10x backbone LR instead of 20x)
   - Try lower batch size (2) for more stable gradients

## Expected Timeline

- V2 completion: Dec 5, ~07:00-09:00
- Evaluation: Dec 5, ~09:00-10:00
- Phase 5 launch: Dec 5, ~10:00-11:00
- Phase 5 completion: Dec 5, ~17:00-19:00
- Documentation: Dec 5, ~19:00-21:00

**Phase 3 Target Completion**: Dec 5, 21:00 (end of day)
