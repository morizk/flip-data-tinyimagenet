"""
Diagnostic script to visualize issues in correlation matrices.
Shows which models are working correctly and which are not.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_matrix(matrix_path, expected_mode='all'):
    """Analyze a correlation matrix and return diagnostics."""
    matrix = np.load(matrix_path)
    name = os.path.basename(matrix_path).replace('_correlation_matrix.npy', '')
    
    diagonal = np.diag(matrix)
    off_diag = matrix.copy()
    np.fill_diagonal(off_diag, 0)
    off_diag_flat = off_diag[off_diag > 0]
    
    if expected_mode == 'all':
        # For "all" mode: diagonal should be ~0, off-diagonal should be uniform ~1/199
        expected_uniform = 1.0 / 199
        diag_mean = diagonal.mean()
        off_diag_mean = off_diag_flat.mean() if len(off_diag_flat) > 0 else 0
        off_diag_std = off_diag_flat.std() if len(off_diag_flat) > 0 else 0
        
        # Check if working correctly
        diag_ok = diag_mean < 0.01  # Diagonal should be very low
        uniform_ok = abs(off_diag_mean - expected_uniform) < 0.001 and off_diag_std < 0.001
        
        status = "✅ WORKING" if (diag_ok and uniform_ok) else "❌ FAILED"
        
        return {
            'name': name,
            'status': status,
            'diag_mean': diag_mean,
            'diag_max': diagonal.max(),
            'off_diag_mean': off_diag_mean,
            'off_diag_std': off_diag_std,
            'expected_uniform': expected_uniform,
            'diag_ok': diag_ok,
            'uniform_ok': uniform_ok
        }
    
    else:  # inverted mode
        # For "inverted" mode: diagonal should be ~0, but can concentrate on wrong classes
        diag_mean = diagonal.mean()
        diag_max = diagonal.max()
        off_diag = matrix.copy()
        np.fill_diagonal(off_diag, 0)
        max_wrong_per_row = off_diag.max(axis=1)
        
        # For inverted mode, mean is more important than max (some classes might be harder)
        diag_ok = diag_mean < 0.01  # Mean diagonal should be very low
        
        status = "✅ WORKING" if diag_ok else "❌ FAILED"
        
        return {
            'name': name,
            'status': status,
            'diag_mean': diag_mean,
            'diag_max': diag_max,
            'max_wrong_mean': max_wrong_per_row.mean(),
            'max_wrong_max': max_wrong_per_row.max(),
            'diag_ok': diag_ok
        }

def main():
    all_dir = 'initial_experiments/all/correlation_matrices'
    inv_dir = 'initial_experiments/inverted/correlation_matrices'
    
    print("="*80)
    print("CORRELATION MATRIX DIAGNOSTICS")
    print("="*80)
    
    print("\n" + "="*80)
    print("'ALL' MODE MODELS")
    print("="*80)
    print("\nExpected: Diagonal (true class) ≈ 0, Off-diagonal (wrong classes) ≈ 1/199 (uniform)")
    print("-"*80)
    
    all_results = []
    for fname in sorted(os.listdir(all_dir)):
        if fname.endswith('.npy') and 'average' not in fname:
            path = os.path.join(all_dir, fname)
            result = analyze_matrix(path, expected_mode='all')
            all_results.append(result)
            
            print(f"\n{result['name']}: {result['status']}")
            print(f"  Diagonal (true class): mean={result['diag_mean']:.6f}, max={result['diag_max']:.6f}")
            if result['diag_ok']:
                print(f"    ✅ Diagonal is low (correct)")
            else:
                print(f"    ❌ Diagonal is too high - model still predicts true class!")
            
            print(f"  Off-diagonal (wrong classes): mean={result['off_diag_mean']:.6f}, std={result['off_diag_std']:.6f}")
            print(f"  Expected uniform: {result['expected_uniform']:.6f}")
            if result['uniform_ok']:
                print(f"    ✅ Off-diagonal is uniform (correct)")
            else:
                print(f"    ❌ Off-diagonal is not uniform - model not distributing correctly!")
    
    print("\n" + "="*80)
    print("'INVERTED' MODE MODELS")
    print("="*80)
    print("\nExpected: Diagonal (true class) ≈ 0, Can concentrate on specific wrong classes")
    print("-"*80)
    
    inv_results = []
    for fname in sorted(os.listdir(inv_dir)):
        if fname.endswith('.npy') and 'average' not in fname:
            path = os.path.join(inv_dir, fname)
            result = analyze_matrix(path, expected_mode='inverted')
            inv_results.append(result)
            
            print(f"\n{result['name']}: {result['status']}")
            print(f"  Diagonal (true class): mean={result['diag_mean']:.6f}, max={result['diag_max']:.6f}")
            if result['diag_ok']:
                print(f"    ✅ Diagonal is low (correct)")
            else:
                print(f"    ❌ Diagonal is too high - model predicts true class when it shouldn't!")
            
            print(f"  Max wrong class prob per row: mean={result['max_wrong_mean']:.6f}, max={result['max_wrong_max']:.6f}")
            print(f"    (Shows how much model concentrates on wrong classes)")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_working = sum(1 for r in all_results if 'WORKING' in r['status'])
    all_failed = len(all_results) - all_working
    print(f"\n'All' mode: {all_working}/{len(all_results)} working correctly, {all_failed} failed")
    
    inv_working = sum(1 for r in inv_results if 'WORKING' in r['status'])
    inv_failed = len(inv_results) - inv_working
    print(f"'Inverted' mode: {inv_working}/{len(inv_results)} working correctly, {inv_failed} failed")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("\nFor failed models:")
    print("  1. Check training logs - did flip loss decrease during training?")
    print("  2. Verify that flip samples were actually used during training")
    print("  3. Consider retraining with:")
    print("     - Stronger loss weighting for flip samples")
    print("     - Longer training")
    print("     - Different learning rates")
    print("     - Check if late fusion has architectural issues")
    print("\nSee CORRELATION_ANALYSIS_REPORT.md for detailed analysis.")

if __name__ == '__main__':
    main()

