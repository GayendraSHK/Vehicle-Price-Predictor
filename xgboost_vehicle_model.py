import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
import sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

# XGBOOST VEHICLE PRICE PREDICTION MODEL

# Load prepared data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv').squeeze()
y_test = pd.read_csv('y_test.csv').squeeze()

print("=" * 60)
print("XGBOOST VEHICLE PRICE PREDICTION MODEL")
print("=" * 60)
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print(f"Feature names: {list(X_train.columns)}")

# STEP 1: BASIC XGBOOST MODEL

print("\n" + "=" * 60)
print("STEP 1: TRAINING BASIC XGBOOST MODEL")
print("=" * 60)

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\nBasic Model Performance:")
print(f"   Training RMSE: Rs {train_rmse:,.0f}")
print(f"   Test RMSE:     Rs {test_rmse:,.0f}")
print(f"   Training MAE:  Rs {train_mae:,.0f}")
print(f"   Test MAE:      Rs {test_mae:,.0f}")
print(f"   Training R2:   {train_r2:.4f}")
print(f"   Test R2:       {test_r2:.4f}")

# STEP 2: HYPERPARAMETER TUNING

print("\n" + "=" * 60)
print("STEP 2: HYPERPARAMETER TUNING (GridSearchCV)")
print("=" * 60)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.15],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

print(f"Parameter grid: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate'])} combinations")
print("Running 3-fold cross-validation... (this may take a few minutes)")

grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    ),
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    verbose=0,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"\nBest Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")

best_xgb = grid_search.best_estimator_

y_pred_train_best = best_xgb.predict(X_train)
y_pred_test_best = best_xgb.predict(X_test)

best_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train_best))
best_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_best))
best_train_mae = mean_absolute_error(y_train, y_pred_train_best)
best_test_mae = mean_absolute_error(y_test, y_pred_test_best)
best_train_r2 = r2_score(y_train, y_pred_train_best)
best_test_r2 = r2_score(y_test, y_pred_test_best)

print(f"\nTuned Model Performance:")
print(f"   Training RMSE: Rs {best_train_rmse:,.0f}")
print(f"   Test RMSE:     Rs {best_test_rmse:,.0f}")
print(f"   Training MAE:  Rs {best_train_mae:,.0f}")
print(f"   Test MAE:      Rs {best_test_mae:,.0f}")
print(f"   Training R2:   {best_train_r2:.4f}")
print(f"   Test R2:       {best_test_r2:.4f}")

improvement = ((test_rmse - best_test_rmse) / test_rmse) * 100
print(f"   RMSE improvement: {improvement:.1f}%")

# STEP 3: CROSS-VALIDATION

print("\n" + "=" * 60)
print("STEP 3: 5-FOLD CROSS-VALIDATION")
print("=" * 60)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_xgb, X_train, y_train, cv=kfold,
                           scoring='neg_mean_squared_error')

cv_rmse = np.sqrt(-cv_scores)
for i, score in enumerate(cv_rmse):
    print(f"   Fold {i+1}: Rs {score:,.0f}")
print(f"   Mean CV RMSE: Rs {cv_rmse.mean():,.0f} (+/- {cv_rmse.std():,.0f})")

# R2 CV scores
cv_r2 = cross_val_score(best_xgb, X_train, y_train, cv=kfold, scoring='r2')
print(f"   Mean CV R2: {cv_r2.mean():.4f} (+/- {cv_r2.std():.4f})")

# STEP 4: FEATURE IMPORTANCE ANALYSIS

print("\n" + "=" * 60)
print("STEP 4: FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_xgb.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance Ranking:")
for i, row in feature_importance.iterrows():
    bar = '#' * int(row['importance'] * 50)
    print(f"   {row['feature']:15s}: {row['importance']:.4f} {bar}")

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))
top_features = feature_importance
ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'].values)
ax.set_xlabel('Importance Score (Gain)')
ax.set_title('XGBoost Feature Importances')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()
print("\nSaved: feature_importance.png")

# STEP 5: SHAP ANALYSIS (EXPLAINABILITY)

print("\n" + "=" * 60)
print("STEP 5: SHAP ANALYSIS (Model Explainability)")
print("=" * 60)
print("Calculating SHAP values...")

explainer = shap.TreeExplainer(best_xgb)

# Use a sample for SHAP (full dataset can be slow)
sample_size = min(200, len(X_test))
X_shap = X_test.sample(sample_size, random_state=42)
shap_values = explainer.shap_values(X_shap)

# 5.1 SHAP Summary Plot (beeswarm)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_shap, show=False)
plt.title('SHAP Summary Plot - Feature Impact on Vehicle Price')
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_summary.png")

# 5.2 SHAP Bar Plot (mean absolute impact)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
plt.title('Mean |SHAP| - Average Feature Impact on Price')
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_bar.png")

# 5.3 SHAP Waterfall - explain a single prediction
sample_idx = 0
plt.figure(figsize=(12, 6))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_shap.iloc[sample_idx].values,
        feature_names=list(X_shap.columns)
    ),
    show=False,
    max_display=10
)
plt.title('SHAP Waterfall - Single Prediction Explanation')
plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_waterfall.png")

# 5.4 SHAP Dependence plots for top 3 features
top3_features = feature_importance.head(3)['feature'].tolist()
for feat in top3_features:
    if feat in X_shap.columns:
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(feat, shap_values, X_shap, show=False)
        plt.title(f'SHAP Dependence: {feat}')
        plt.tight_layout()
        plt.savefig(f'shap_dep_{feat}.png', dpi=150)
        plt.close()
        print(f"Saved: shap_dep_{feat}.png")

# STEP 6: PREDICTION ANALYSIS & PLOTS

print("\n" + "=" * 60)
print("STEP 6: PREDICTION ANALYSIS")
print("=" * 60)

predictions_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred_test_best,
    'Error': y_test.values - y_pred_test_best,
    'Abs_Error': np.abs(y_test.values - y_pred_test_best),
    'Pct_Error': np.abs((y_test.values - y_pred_test_best) / y_test.values) * 100
})

print(f"\nPrediction Error Statistics:")
print(f"   Mean Absolute Error: Rs {predictions_df['Abs_Error'].mean():,.0f}")
print(f"   Median Abs Error:    Rs {predictions_df['Abs_Error'].median():,.0f}")
print(f"   Mean % Error:        {predictions_df['Pct_Error'].mean():.1f}%")
print(f"   Median % Error:      {predictions_df['Pct_Error'].median():.1f}%")

# Accuracy buckets
within_10pct = (predictions_df['Pct_Error'] <= 10).mean() * 100
within_20pct = (predictions_df['Pct_Error'] <= 20).mean() * 100
within_30pct = (predictions_df['Pct_Error'] <= 30).mean() * 100
print(f"\n   Predictions within 10% of actual: {within_10pct:.1f}%")
print(f"   Predictions within 20% of actual: {within_20pct:.1f}%")
print(f"   Predictions within 30% of actual: {within_30pct:.1f}%")

# 6.1 Actual vs Predicted scatter
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(y_test / 1e6, y_pred_test_best / 1e6, alpha=0.5, s=20, c='steelblue', edgecolors='navy', linewidth=0.3)
max_val = max(y_test.max(), y_pred_test_best.max()) / 1e6
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Price (Million Rs)', fontsize=12)
ax.set_ylabel('Predicted Price (Million Rs)', fontsize=12)
ax.set_title(f'Actual vs Predicted Vehicle Prices (R2 = {best_test_r2:.4f})', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
plt.close()
print("Saved: actual_vs_predicted.png")

# 6.2 Residuals plot
fig, ax = plt.subplots(figsize=(10, 6))
residuals = y_test - y_pred_test_best
ax.scatter(y_pred_test_best / 1e6, residuals / 1e6, alpha=0.5, s=20, c='steelblue', edgecolors='navy', linewidth=0.3)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Price (Million Rs)', fontsize=12)
ax.set_ylabel('Residual (Million Rs)', fontsize=12)
ax.set_title('Residuals Plot', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('residuals.png', dpi=150)
plt.close()
print("Saved: residuals.png")

# 6.3 Error distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(predictions_df['Pct_Error'].clip(upper=100), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(predictions_df['Pct_Error'].median(), color='red', linestyle='--', label=f"Median: {predictions_df['Pct_Error'].median():.1f}%")
ax.set_xlabel('Percentage Error (%)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Prediction Errors', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('error_distribution.png', dpi=150)
plt.close()
print("Saved: error_distribution.png")

# STEP 7: SAVE MODEL & RESULTS

print("\n" + "=" * 60)
print("STEP 7: SAVING MODEL & RESULTS")
print("=" * 60)

# Save model
best_xgb.save_model('xgb_vehicle_price_model.json')
print("Saved: xgb_vehicle_price_model.json")

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("Saved: feature_importance.csv")

# Save predictions
predictions_df.to_csv('predictions.csv', index=False)
print("Saved: predictions.csv")

# Save metrics summary
metrics = {
    'Model': 'XGBoost',
    'Basic_Test_RMSE': test_rmse,
    'Basic_Test_R2': test_r2,
    'Tuned_Train_RMSE': best_train_rmse,
    'Tuned_Test_RMSE': best_test_rmse,
    'Tuned_Train_MAE': best_train_mae,
    'Tuned_Test_MAE': best_test_mae,
    'Tuned_Train_R2': best_train_r2,
    'Tuned_Test_R2': best_test_r2,
    'CV_RMSE_mean': cv_rmse.mean(),
    'CV_RMSE_std': cv_rmse.std(),
    'CV_R2_mean': cv_r2.mean(),
    'CV_R2_std': cv_r2.std(),
    'Best_Params': str(grid_search.best_params_),
    'Within_10pct': within_10pct,
    'Within_20pct': within_20pct,
    'Within_30pct': within_30pct,
}
pd.DataFrame([metrics]).to_csv('model_metrics.csv', index=False)
print("Saved: model_metrics.csv")

# FINAL SUMMARY

print("\n" + "=" * 60)
print("XGBOOST MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"\n  Final Model Performance:")
print(f"    RMSE: Rs {best_test_rmse:,.0f}")
print(f"    MAE:  Rs {best_test_mae:,.0f}")
print(f"    R2:   {best_test_r2:.4f}")
print(f"    CV RMSE: Rs {cv_rmse.mean():,.0f} (+/- {cv_rmse.std():,.0f})")
print(f"\n  Files saved:")
print(f"    xgb_vehicle_price_model.json  - Trained model")
print(f"    feature_importance.csv/.png    - Feature rankings")
print(f"    predictions.csv               - Actual vs predicted")
print(f"    model_metrics.csv             - All metrics")
print(f"    shap_summary.png              - SHAP beeswarm plot")
print(f"    shap_bar.png                  - SHAP bar plot")
print(f"    shap_waterfall.png            - Single prediction explanation")
print(f"    actual_vs_predicted.png       - Scatter plot")
print(f"    residuals.png                 - Residuals plot")
print(f"    error_distribution.png        - Error histogram")
