# Why Inverted Mode Models Collapse to a Single Class

## The Problem

In inverted mode, models are collapsing to predict a **single specific wrong class** for almost all inputs when `flip=1`:

- **ResNet18 Early Inverted**: Predicts class **149** for 197/200 true classes
- **ResNet18 Late Inverted**: Predicts class **126** for 199/200 true classes  
- **VGG11 Early Inverted**: Predicts class **167** for 193/200 true classes
- **VGG11 Late Inverted**: Predicts class **154** for 199/200 true classes

## Root Cause: The Loss Function Design

The `flip_inverted_loss` function (in `losses.py`) only minimizes the probability of the **true class**:

```python
def flip_inverted_loss(predictions, true_labels):
    probs = F.softmax(predictions, dim=1)
    true_class_probs = probs[range(batch_size), true_labels]
    loss = true_class_probs.mean()  # Only minimizes P(true_class)
    return loss
```

### Why This Causes Collapse

1. **No constraint on wrong classes**: The loss only cares about `P(true_class) → 0`. It doesn't care **which** wrong class gets the probability.

2. **Easiest solution**: The model learns the simplest solution: "Always predict class X when flip=1" (where X is some arbitrary class like 149, 126, etc.)

3. **Mode collapse**: This is a form of **mode collapse** - the model collapses all probability mass onto a single wrong class because:
   - It minimizes the loss (P(true_class) ≈ 0)
   - It requires minimal learning (just predict one class)
   - There's no penalty for concentrating on one class

4. **Failure case**: When the true class **IS** that specific wrong class (e.g., true class = 149), the model fails because:
   - It learned: "When flip=1, predict class 149"
   - But didn't learn: "When flip=1, predict class 149 **UNLESS** true class is 149"

## Mathematical Explanation

The loss function is:
```
L = P(true_class)
```

To minimize this, the model can:
- Put all probability on **any** wrong class → `P(true_class) = 0` ✅
- Put probability uniformly on wrong classes → `P(true_class) = 0` ✅
- Put probability on multiple wrong classes → `P(true_class) = 0` ✅

**All of these give the same loss value!** So the model chooses the easiest one: concentrate on one class.

## Why Some Models Work Better

Looking at the results:
- **VGG11 Early Inverted**: Works better (class 167, but avoids it when true class is 167)
- **ResNet18 Early Inverted**: Partially works (class 149, but still predicts it 16% when true class is 149)
- **ResNet18 Late Inverted**: Fails completely (class 126, predicts it 100% when true class is 126)
- **VGG11 Late Inverted**: Fails completely (class 154, predicts it 100% when true class is 154)

The difference might be:
1. **Early fusion** learns the conditional better than **late fusion**
2. **VGG11** architecture might handle the conditional better than **ResNet18**
3. **Random initialization** might lead to different collapse points
