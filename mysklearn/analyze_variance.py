#!/usr/bin/env python3
"""
Analyze feature variance in the discretized dataset and identify features to drop.
Goal: Reduce to ~12 features for binary prediction.
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Load dataset
df = pd.read_csv('input_data/bitcoin_sentiment_discretized.csv')

# Separate features and target
feature_cols = [col for col in df.columns if col != 'price_direction']
X = df[feature_cols]
y = df['price_direction']

print("="*80)
print("FEATURE VARIANCE ANALYSIS")
print("="*80)
print(f"\nTotal features: {len(feature_cols)}")
print(f"Target: price_direction (binary: {y.unique()})")
print(f"Total instances: {len(df)}")
print()

# Analyze each feature
feature_stats = []

for col in feature_cols:
    value_counts = X[col].value_counts()
    total = len(X[col])
    n_unique = X[col].nunique()
    
    # Calculate variance metric: 1 - (proportion of most common value)
    most_common_prop = value_counts.iloc[0] / total
    variance_metric = 1 - most_common_prop
    
    # Check class-specific distributions
    up_dist = df[df['price_direction'] == 'Up'][col].value_counts(normalize=True)
    down_dist = df[df['price_direction'] == 'Down'][col].value_counts(normalize=True)
    
    # Calculate chi-square test for independence (feature vs class)
    contingency_table = pd.crosstab(df[col], df['price_direction'])
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    except:
        chi2, p_value = 0, 1.0
    
    # Calculate how different distributions are between classes
    # Use KL divergence-like metric: sum of absolute differences
    common_values = set(up_dist.index) | set(down_dist.index)
    distribution_diff = sum(abs(up_dist.get(val, 0) - down_dist.get(val, 0)) for val in common_values) / 2
    
    feature_stats.append({
        'feature': col,
        'n_unique': n_unique,
        'most_common': value_counts.index[0],
        'most_common_prop': most_common_prop,
        'variance_metric': variance_metric,
        'chi2_pvalue': p_value,
        'distribution_diff': distribution_diff,
        'is_low_variance': most_common_prop > 0.95 or n_unique == 1,
        'is_informative': p_value < 0.05 and distribution_diff > 0.1  # Significant and different distributions
    })

# Sort by informativeness (chi2 p-value, then distribution difference)
feature_stats.sort(key=lambda x: (x['chi2_pvalue'], -x['distribution_diff']))

print("Feature Analysis (sorted by informativeness):")
print("-"*80)
print(f"{'Feature':<45} {'Unique':<8} {'Most Common':<12} {'Var Metric':<12} {'Chi2 p-val':<12} {'Dist Diff':<12} {'Status':<20}")
print("-"*80)

low_variance_features = []
informative_features = []

for stat in feature_stats:
    status = []
    if stat['is_low_variance']:
        status.append("LOW_VAR")
        low_variance_features.append(stat['feature'])
    if stat['is_informative']:
        status.append("INFORMATIVE")
        informative_features.append(stat['feature'])
    if not status:
        status.append("OK")
    
    status_str = ", ".join(status)
    
    print(f"{stat['feature']:<45} {stat['n_unique']:<8} {stat['most_common']:<12} "
          f"{stat['variance_metric']:<12.3f} {stat['chi2_pvalue']:<12.4f} "
          f"{stat['distribution_diff']:<12.3f} {status_str:<20}")

print()
print("="*80)
print("SUMMARY")
print("="*80)
print(f"\nLow-variance features (to drop): {len(low_variance_features)}")
for feat in low_variance_features:
    print(f"  - {feat}")

print(f"\nInformative features (keep): {len(informative_features)}")
for feat in informative_features:
    print(f"  - {feat}")

# Select top ~12 most informative features
# Prioritize: low chi2 p-value (significant), high distribution difference
top_features = sorted(feature_stats, 
                     key=lambda x: (x['chi2_pvalue'], -x['distribution_diff'], -x['variance_metric']))[:12]

print(f"\n{'='*80}")
print(f"TOP 12 FEATURES SELECTED FOR CLEANED DATASET")
print(f"{'='*80}")
print()

selected_features = []
for i, stat in enumerate(top_features, 1):
    selected_features.append(stat['feature'])
    print(f"{i:2d}. {stat['feature']:<45} "
          f"(p={stat['chi2_pvalue']:.4f}, diff={stat['distribution_diff']:.3f}, "
          f"var={stat['variance_metric']:.3f})")

print()
print(f"Selected {len(selected_features)} features")
print(f"Dropping {len(feature_cols) - len(selected_features)} features")

# Create cleaned dataset
df_cleaned = df[selected_features + ['price_direction']].copy()
df_cleaned.to_csv('input_data/bitcoin_sentiment_discretized_cleaned.csv', index=False)

print()
print(f"✓ Created cleaned dataset: input_data/bitcoin_sentiment_discretized_cleaned.csv")
print(f"  Original features: {len(feature_cols)}")
print(f"  Cleaned features: {len(selected_features)}")
print(f"  Reduction: {len(feature_cols) - len(selected_features)} features removed ({100*(len(feature_cols) - len(selected_features))/len(feature_cols):.1f}%)")

