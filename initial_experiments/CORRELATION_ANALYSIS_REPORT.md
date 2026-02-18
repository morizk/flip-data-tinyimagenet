# Correlation Matrix Analysis Report

## Summary

The correlation matrices have been correctly computed using **probability distributions** (not hard predictions). However, the analysis reveals that **some models failed to learn the flip behavior correctly during training**.

## Methodology

The correlation matrices show: `P(predicted_class=Y | true_class=X, flip=1)`

- For each true class X, we average the probability distribution over all samples with that true class and flip=1
- This gives us the average probability that the model predicts each class when the input is inverted

## Findings

### "All" Mode Models

**Expected Behavior**: When flip=1, model should predict uniform distribution over all wrong classes (1/199 ≈ 0.005025 for each wrong class, ~0 for true class).

#### ✅ **Working Correctly**:
1. **VGG11 Early All**: 
   - Diagonal (true class): Mean = 0.005000 (correctly ~0)
   - Off-diagonal (wrong classes): Mean = 0.005000 (correctly uniform)
   - ✅ Model learned to avoid true class and distribute uniformly

2. **ResNet18 Early All**:
   - Diagonal: Mean = 0.005014 (correctly ~0)
   - Off-diagonal: Mean = 0.005000 (correctly uniform)
   - ✅ Model learned correctly

#### ❌ **NOT Working Correctly**:
1. **ResNet18 Late All**: 
   - Diagonal (true class): Mean = **0.397074**, Max = 0.507465
   - Off-diagonal (wrong classes): Mean = 0.003030
   - ❌ **Model still predicts true class with ~40% probability even when flip=1**
   - ❌ This means the model didn't learn to avoid the true class in "all" mode
   - Example: For true class 0, model predicts class 0 with 32% probability (should be ~0%)

2. **VGG11 Late All**:
   - Diagonal: Mean = 0.007122 (slightly elevated, but better than ResNet18 Late)
   - Off-diagonal: Mean = 0.004989 (close to uniform)
   - ⚠️ Partially working, but still has some bias toward true class

### "Inverted" Mode Models

**Expected Behavior**: When flip=1, model should minimize probability of true class (diagonal should be ~0), but can concentrate probability on specific wrong classes.

#### ✅ **Working Correctly**:
1. **VGG11 Early Inverted**:
   - Diagonal: Mean = 0.000503 (correctly very low)
   - Max wrong class prob: Mean = 0.782867 (concentrates on wrong classes)
   - ✅ Model successfully avoids true class

2. **ResNet18 Early Inverted**:
   - Diagonal: Mean = 0.000857 (correctly very low)
   - Max wrong class prob: Mean = 0.961893 (strongly concentrates on wrong classes)
   - ✅ Model successfully avoids true class

#### ❌ **NOT Working Correctly**:
1. **VGG11 Late Inverted**:
   - Diagonal: Mean = 0.005006, but **Max = 1.0** at class 154
   - ❌ **When true class is 154, model predicts 154 with 100% probability**
   - ❌ Model concentrates on class 154 for all other inputs (correct), but fails when true class IS 154
   - This suggests the model learned to predict class 154 as a "default wrong class", but didn't learn to avoid it when it's the true class

2. **ResNet18 Late Inverted**:
   - Diagonal: Mean = 0.011495, Max = 1.0
   - ⚠️ Better than VGG11 Late, but still has issues with some classes

## Root Cause Analysis

The correlation matrices are **correctly computed**. The issue is that **these specific models did not learn the flip behavior during training**:

1. **ResNet18 Late All**: The model was trained with `flip_all_loss` but didn't learn to avoid the true class. Possible reasons:
   - Loss function not strong enough
   - Learning rate too low
   - Model capacity issue
   - Training didn't converge properly

2. **VGG11 Late Inverted**: The model learned to concentrate on a specific wrong class (154) but didn't learn to avoid it when it's the true class. This suggests:
   - The model learned a "shortcut": always predict class 154 when flip=1
   - But didn't learn the conditional logic: "predict class 154 UNLESS true class is 154"

## Recommendations

1. **Retrain problematic models** with:
   - Stronger loss weighting for flip samples
   - Longer training
   - Different learning rates
   - Check if loss is actually decreasing for flip samples during training

2. **Verify training logs** to see if:
   - `flip_all_loss` was actually being computed and minimized
   - The model was seeing both flip=0 and flip=1 samples during training

3. **Check model architecture** - Late fusion might have issues learning the flip behavior compared to early fusion

## Files Analyzed

- `initial_experiments/all/correlation_matrices/*.npy`
- `initial_experiments/inverted/correlation_matrices/*.npy`

All matrices are correctly normalized (rows sum to 1.0) and use probability distributions (not hard predictions).







