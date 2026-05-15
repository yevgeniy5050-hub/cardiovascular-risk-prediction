"""
Modellvergleich: Logistische Regression, Random Forest, XGBoost
Kardiovaskuläre Erkrankungen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURATION
# ============================================================================
FILEPATH = 'cardio_train.csv'
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# ============================================================================
# 1. DATEN LADEN & CLEANING
# ============================================================================
df = pd.read_csv(FILEPATH, sep=';')
df_clean = df.copy()

df_clean = df_clean[(df_clean['ap_hi'] >= 50) & (df_clean['ap_hi'] <= 250)]
df_clean = df_clean[(df_clean['ap_lo'] >= 30) & (df_clean['ap_lo'] <= 150)]
df_clean = df_clean[df_clean['ap_hi'] > df_clean['ap_lo']]
df_clean = df_clean[(df_clean['height'] >= 130) & (df_clean['height'] <= 220)]
df_clean = df_clean[(df_clean['weight'] >= 30) & (df_clean['weight'] <= 200)]
df_clean = df_clean.drop_duplicates()

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
df_clean['bmi'] = df_clean['weight'] / ((df_clean['height'] / 100) ** 2)
df_clean['age_years'] = df_clean['age'] / 365
df_clean['pulse_pressure'] = df_clean['ap_hi'] - df_clean['ap_lo']
df_clean['map'] = df_clean['ap_lo'] + (df_clean['pulse_pressure'] / 3)

df_clean['bp_category'] = 0
df_clean.loc[(df_clean['ap_hi'] >= 130) | (df_clean['ap_lo'] >= 85), 'bp_category'] = 1
df_clean.loc[(df_clean['ap_hi'] >= 140) | (df_clean['ap_lo'] >= 90), 'bp_category'] = 2

df_clean['bmi_category'] = 0
df_clean.loc[df_clean['bmi'] < 18.5, 'bmi_category'] = -1
df_clean.loc[(df_clean['bmi'] >= 25) & (df_clean['bmi'] < 30), 'bmi_category'] = 1
df_clean.loc[df_clean['bmi'] >= 30, 'bmi_category'] = 2

df_clean['risk_score'] = (
        (df_clean['bp_category'] >= 1).astype(int) +
        (df_clean['cholesterol'] > 1).astype(int) +
        (df_clean['gluc'] > 1).astype(int) +
        (df_clean['smoke'] == 1).astype(int) +
        (df_clean['alco'] == 1).astype(int) +
        (df_clean['active'] == 0).astype(int)
)

# ============================================================================
# 3. VORBEREITUNG
# ============================================================================
feature_columns = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                   'cholesterol', 'gluc', 'smoke', 'alco', 'active',
                   'bmi', 'age_years', 'pulse_pressure', 'map',
                   'bp_category', 'bmi_category', 'risk_score']

X = df_clean[feature_columns]
y = df_clean['cardio']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ============================================================================
# 4. MODELLE DEFINIEREN
# ============================================================================
models = {
    'Logistische Regression': LogisticRegression(
        C=1.0, max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_split=10,
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False
    )
}

# ============================================================================
# 5. TRAINING & EVALUATION
# ============================================================================
results = {}

for name, model in models.items():
    # Skalierte Daten für LogReg, normale für Tree-Modelle
    if name == 'Logistische Regression':
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test

    # Cross-Validation
    cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='roc_auc')

    # Training
    model.fit(X_tr, y_train)

    # Prediction
    y_pred = model.predict(X_te)
    y_pred_proba = model.predict_proba(X_te)[:, 1]

    # Metriken
    results[name] = {
        'cv_roc_auc': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred_proba': y_pred_proba,
        'model': model
    }

# ============================================================================
# 6. KORRELATIONSMATRIX
# ============================================================================
plt.figure(figsize=(12, 10))
corr = df_clean[feature_columns + ['cardio']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
plt.title('Korrelationsmatrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)
plt.close()

# ============================================================================
# 7. ROC CURVES VERGLEICH
# ============================================================================
plt.figure(figsize=(8, 8))
colors = {'Logistische Regression': '#e57373', 'Random Forest': '#81c784', 'XGBoost': '#ff9800'}

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
    plt.plot(fpr, tpr, color=colors[name], lw=2, label=f"{name} (AUC = {res['roc_auc']:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Zufall (AUC = 0.500)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Modellvergleich', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curves_comparison.png', dpi=300)
plt.close()

# ============================================================================
# 8. CONFUSION MATRICES
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (name, res) in enumerate(results.items()):
    sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Gesund', 'Krank'], yticklabels=['Gesund', 'Krank'])
    axes[idx].set_title(f"{name}\nAccuracy: {res['accuracy']:.3f}")
    axes[idx].set_xlabel('Vorhergesagt')
    axes[idx].set_ylabel('Tatsächlich')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300)
plt.close()

# ============================================================================
# 9. PERFORMANCE VERGLEICH
# ============================================================================
plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.25

for idx, (name, res) in enumerate(results.items()):
    values = [res['accuracy'], res['precision'], res['recall'], res['f1'], res['roc_auc']]
    bars = plt.bar(x + idx * width, values, width, label=name, color=colors[name])
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f'{val:.3f}',
                 ha='center', va='bottom', fontsize=8)

plt.xticks(x + width, metrics)
plt.ylim(0.6, 0.85)
plt.ylabel('Score')
plt.title('Performance Vergleich', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300)
plt.close()

# ============================================================================
# 10. FEATURE IMPORTANCE
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Logistische Regression - Koeffizienten
coef = pd.DataFrame(
    {'Feature': feature_columns, 'Importance': np.abs(results['Logistische Regression']['model'].coef_[0])})
coef = coef.sort_values('Importance', ascending=True)
axes[0].barh(coef['Feature'], coef['Importance'], color='#e57373')
axes[0].set_title('Logistische Regression\n(Koeffizienten)', fontweight='bold')

# Random Forest - Feature Importance
rf_imp = pd.DataFrame(
    {'Feature': feature_columns, 'Importance': results['Random Forest']['model'].feature_importances_})
rf_imp = rf_imp.sort_values('Importance', ascending=True)
axes[1].barh(rf_imp['Feature'], rf_imp['Importance'], color='#81c784')
axes[1].set_title('Random Forest\n(Feature Importance)', fontweight='bold')

# XGBoost - Feature Importance
xgb_imp = pd.DataFrame({'Feature': feature_columns, 'Importance': results['XGBoost']['model'].feature_importances_})
xgb_imp = xgb_imp.sort_values('Importance', ascending=True)
axes[2].barh(xgb_imp['Feature'], xgb_imp['Importance'], color='#ff9800')
axes[2].set_title('XGBoost\n(Feature Importance)', fontweight='bold')

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=300)
plt.close()

# ============================================================================
# 11. ERGEBNISSE
# ============================================================================
print("\n" + "=" * 60)
print("MODELLVERGLEICH - ERGEBNISSE")
print("=" * 60)

# Tabelle
print(f"\n{'Modell':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
print("-" * 75)
for name, res in results.items():
    print(
        f"{name:<25} {res['accuracy']:>10.4f} {res['precision']:>10.4f} {res['recall']:>10.4f} {res['f1']:>10.4f} {res['roc_auc']:>10.4f}")

# Bestes Modell
best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])
print(f"\n{'=' * 60}")
print(f"BESTES MODELL: {best_model[0]} (ROC-AUC: {best_model[1]['roc_auc']:.4f})")
print(f"{'=' * 60}")