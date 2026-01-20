"""
STEP 3: Model Training with XGBoost
Trains separate binary classifiers for each CRA form with proper imbalance handling

Input:  cra_dataset_engineered.csv
Output: models/cra_models_trained.pkl
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, roc_auc_score
import xgboost as xgb
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("STEP 3: MODEL TRAINING")
print("=" * 70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Create models directory
os.makedirs("models", exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================

print("Loading engineered dataset...")
df = pd.read_csv("cra_dataset_engineered.csv")
print(f"✓ Loaded {len(df):,} records with {len(df.columns)} features\n")

# ============================================================================
# PREPARE FEATURES
# ============================================================================

print("Preparing features...")

# Drop ID and target columns
feature_cols = [col for col in df.columns if col not in 
                ['id', 'recommended_forms', 'num_forms_required']]

X = df[feature_cols].copy()

# Encode categorical variables
print("  Encoding categorical variables...")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"  ✓ Encoded {len(categorical_cols)} categorical features")

# Handle any remaining NaN
X = X.fillna(0)

print(f"  ✓ Final feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

# ============================================================================
# SPLIT DATA
# ============================================================================

print("\nSplitting data...")

# Get primary form for stratification
primary_forms = df['recommended_forms'].str.split(',').str[0]

X_train, X_temp, y_train_raw, y_temp_raw = train_test_split(
    X, df['recommended_forms'],
    test_size=0.3,
    random_state=42,
    stratify=primary_forms
)

X_val, X_test, y_val_raw, y_test_raw = train_test_split(
    X_temp, y_temp_raw,
    test_size=0.5,
    random_state=42
)

print(f"  Training set:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"  Test set:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

# ============================================================================
# GET ALL UNIQUE FORMS
# ============================================================================

print("\nAnalyzing form distribution...")
all_forms = sorted(set(
    form 
    for forms in df['recommended_forms'] 
    for form in forms.split(',')
))

print(f"  Total unique forms: {len(all_forms)}")

# Count occurrences
form_counts = defaultdict(int)
for forms_str in df['recommended_forms']:
    for form in forms_str.split(','):
        form_counts[form] += 1

# Categorize by frequency
rare_forms = []
moderate_forms = []
common_forms = []

for form, count in form_counts.items():
    pct = count / len(df)
    if pct < 0.05:
        rare_forms.append(form)
    elif pct < 0.20:
        moderate_forms.append(form)
    else:
        common_forms.append(form)

print(f"\n  Form Categories:")
print(f"    Common (>20%):    {len(common_forms)} forms")
print(f"    Moderate (5-20%): {len(moderate_forms)} forms")
print(f"    Rare (<5%):       {len(rare_forms)} forms")

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def calculate_scale_pos_weight(y_binary):
    """Calculate XGBoost scale_pos_weight parameter"""
    n_pos = y_binary.sum()
    n_neg = len(y_binary) - n_pos
    
    if n_pos == 0:
        return 1.0
    
    weight = n_neg / n_pos
    return min(weight, 50.0)  # Cap at 50

def train_form_model(X_train, y_train, X_val, y_val, form_name, form_stats):
    """Train one binary classifier for a specific form"""
    
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_ratio = pos_count / len(y_train)
    
    if pos_count < 10:
        print(f"    ⚠️  Skipping {form_name} - only {pos_count} positive samples")
        return None, None, None
    
    # Calculate class weight
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    
    # Determine strategy based on imbalance
    if pos_ratio < 0.05:  # Severe imbalance
        strategy = "AGGRESSIVE"
        params = {
            'scale_pos_weight': scale_pos_weight * 1.5,  # Extra boost
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'eval_metric': 'aucpr',
            'random_state': 42,
            'tree_method': 'hist',
            'verbosity': 0
        }
    elif pos_ratio < 0.20:  # Moderate imbalance
        strategy = "STANDARD"
        params = {
            'scale_pos_weight': scale_pos_weight,
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.05,
            'reg_alpha': 0.05,
            'eval_metric': 'aucpr',
            'random_state': 42,
            'tree_method': 'hist',
            'verbosity': 0
        }
    else:  # Balanced
        strategy = "MINIMAL"
        params = {
            'scale_pos_weight': scale_pos_weight,
            'max_depth': 4,
            'learning_rate': 0.1,
            'n_estimators': 150,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss',
            'random_state': 42,
            'tree_method': 'hist',
            'verbosity': 0
        }
    
    # Train model
    model = xgb.XGBClassifier(**params)
    
    eval_set = [(X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    # Find optimal threshold on validation set
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Try different thresholds
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_val_pred = (y_val_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_val_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Calculate AUC
    try:
        auc_score = roc_auc_score(y_val, y_val_proba)
    except:
        auc_score = 0.0
    
    stats = {
        'form': form_name,
        'strategy': strategy,
        'pos_count': int(pos_count),
        'pos_ratio': float(pos_ratio),
        'scale_pos_weight': float(scale_pos_weight),
        'threshold': float(best_threshold),
        'val_f1': float(best_f1),
        'val_auc': float(auc_score)
    }
    
    return model, best_threshold, stats

# ============================================================================
# TRAIN ALL MODELS
# ============================================================================

print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

models = {}
thresholds = {}
training_stats = []

for i, form in enumerate(all_forms, 1):
    print(f"\n[{i}/{len(all_forms)}] Training: {form}")
    
    # Create binary labels
    y_train = df.loc[X_train.index, 'recommended_forms'].str.contains(
        form, regex=False
    ).astype(int)
    
    y_val = df.loc[X_val.index, 'recommended_forms'].str.contains(
        form, regex=False
    ).astype(int)
    
    # Train model
    model, threshold, stats = train_form_model(
        X_train, y_train, 
        X_val, y_val,
        form, form_counts[form]
    )
    
    if model is not None:
        models[form] = model
        thresholds[form] = threshold
        training_stats.append(stats)
        
        print(f"    Strategy: {stats['strategy']}")
        print(f"    Positive samples: {stats['pos_count']:,} ({stats['pos_ratio']*100:.1f}%)")
        print(f"    Scale weight: {stats['scale_pos_weight']:.2f}")
        print(f"    Threshold: {stats['threshold']:.2f}")
        print(f"    Validation F1: {stats['val_f1']:.3f}")
        print(f"    Validation AUC: {stats['val_auc']:.3f}")

print(f"\n✓ Successfully trained {len(models)} models")

# ============================================================================
# EVALUATE ON TEST SET
# ============================================================================

print("\n" + "=" * 70)
print("TEST SET EVALUATION")
print("=" * 70)

test_results = []

for form in models.keys():
    # Create binary labels
    y_test = df.loc[X_test.index, 'recommended_forms'].str.contains(
        form, regex=False
    ).astype(int)
    
    # Predict
    y_test_proba = models[form].predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= thresholds[form]).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_test_proba)
    except:
        auc = 0.0
    
    test_results.append({
        'form': form,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'support': int(y_test.sum())
    })

# Create results dataframe
results_df = pd.DataFrame(test_results).sort_values('f1_score', ascending=False)

print("\nTop 15 Models by F1-Score:")
print("-" * 70)
print(f"{'Form':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10} {'Support':>10}")
print("-" * 70)

for _, row in results_df.head(15).iterrows():
    print(f"{row['form']:<15} {row['precision']:>10.3f} {row['recall']:>10.3f} "
          f"{row['f1_score']:>10.3f} {row['auc']:>10.3f} {row['support']:>10,}")

# Overall statistics
print(f"\n{'Overall Performance':^70}")
print("-" * 70)
print(f"Average Precision: {results_df['precision'].mean():.3f}")
print(f"Average Recall:    {results_df['recall'].mean():.3f}")
print(f"Average F1-Score:  {results_df['f1_score'].mean():.3f}")
print(f"Average AUC:       {results_df['auc'].mean():.3f}")

# Identify weak models
weak_models = results_df[results_df['f1_score'] < 0.70]
if len(weak_models) > 0:
    print(f"\n⚠️  {len(weak_models)} forms with F1 < 0.70:")
    for _, row in weak_models.iterrows():
        print(f"    {row['form']:<15} F1: {row['f1_score']:.3f} (Support: {row['support']:,})")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

print("\n" + "=" * 70)
print("TOP FEATURE IMPORTANCE (Average across all models)")
print("=" * 70)

# Calculate average feature importance
feature_importance = defaultdict(float)
feature_names = X_train.columns.tolist()

for form, model in models.items():
    importances = model.feature_importances_
    for feature, importance in zip(feature_names, importances):
        feature_importance[feature] += importance

# Average
for feature in feature_importance:
    feature_importance[feature] /= len(models)

# Sort and display top 20
sorted_features = sorted(feature_importance.items(), key=lambda x: -x[1])

print("\nTop 20 Most Important Features:")
print("-" * 70)
for i, (feature, importance) in enumerate(sorted_features[:20], 1):
    bar = "█" * int(importance * 100)
    print(f"{i:2}. {feature:<35} {importance:.4f} {bar}")

# ============================================================================
# SAVE MODELS
# ============================================================================

print("\n" + "=" * 70)
print("SAVING MODELS")
print("=" * 70)

model_package = {
    'models': models,
    'thresholds': thresholds,
    'label_encoders': label_encoders,
    'feature_names': feature_names,
    'training_stats': training_stats,
    'test_results': results_df.to_dict('records'),
    'form_list': all_forms,
    'categorical_columns': categorical_cols,
    'metadata': {
        'training_date': datetime.now().isoformat(),
        'n_samples_train': len(X_train),
        'n_samples_val': len(X_val),
        'n_samples_test': len(X_test),
        'n_features': len(feature_names),
        'n_models': len(models)
    }
}

model_filename = "models/cra_models_trained.pkl"
with open(model_filename, 'wb') as f:
    pickle.dump(model_package, f)

print(f"✓ Models saved to: {model_filename}")
print(f"  File size: {os.path.getsize(model_filename) / 1024 / 1024:.1f} MB")

# Save test results
results_df.to_csv("models/test_results.csv", index=False)
print(f"✓ Test results saved to: models/test_results.csv")

# Save training stats
pd.DataFrame(training_stats).to_csv("models/training_stats.csv", index=False)
print(f"✓ Training stats saved to: models/training_stats.csv")

print(f"\n{'SUCCESS':^70}")
print("=" * 70)
print(f"✓ Trained {len(models)} models successfully")
print(f"✓ Average F1-Score: {results_df['f1_score'].mean():.3f}")
print(f"✓ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
print("\nNext Step: Run 'step4_api.py' to start the API server")
print("=" * 70)