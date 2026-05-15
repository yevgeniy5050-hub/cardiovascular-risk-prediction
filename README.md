# 🫀 Cardiovascular Disease Risk Prediction — ML Model Comparison

**Kurs:** Big Data & Machine Learning 2 | TH Brandenburg, 2025  
**Autor:** Yevgeniy Gubov

## Projektübersicht

Vergleich von drei Machine-Learning-Modellen zur Vorhersage kardiovaskulärer Erkrankungen anhand eines Datensatzes mit über 70.000 Patient:innen.

## Modelle

| Modell | Bibliothek |
|--------|------------|
| Logistische Regression | scikit-learn |
| Random Forest | scikit-learn |
| XGBoost | xgboost |

## Feature Engineering

Neu berechnete Features: BMI, Pulse Pressure, MAP, Blutdruckkategorie, BMI-Kategorie, kombinierter Risiko-Score

## Ergebnisse

Evaluation mit Accuracy, Precision, Recall, F1-Score, ROC-AUC und 5-Fold Cross-Validation.  
**Bestes Modell: XGBoost** (höchste ROC-AUC)

## Technologien

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `matplotlib` · `seaborn`

## Ausführung

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
python LR.py
```
