"""
XGBoost Model Training for Oncology Drug Response Prediction
Trains and tests an XGBoost model on the flattened feature dataset
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import json
from datetime import datetime


def load_training_data(features_path, targets_path):
    """
    Load the flattened training dataset
    """
    print("Loading training data...")
    X = pd.read_csv(features_path)
    y = pd.read_csv(targets_path)
    
    print(f"  Features shape: {X.shape}")
    print(f"  Targets shape: {y.shape}")
    print(f"  Feature columns: {list(X.columns)}")
    
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    """
    print(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def train_xgboost_model(X_train, y_train, X_test, y_test, use_grid_search=True):
    """
    Train XGBoost model with optional hyperparameter tuning
    """
    print("\nTraining XGBoost model...")
    
    # Convert to numpy arrays for XGBoost
    y_train_array = y_train.values.ravel()
    y_test_array = y_test.values.ravel()
    
    if use_grid_search:
        print("  Performing hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Create base model
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            xgb_model, 
            param_grid, 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train_array)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"  Best parameters: {best_params}")
        print(f"  Best CV score: {-grid_search.best_score_:.4f}")
        
    else:
        print("  Training with default parameters...")
        best_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        best_model.fit(X_train, y_train_array)
        best_params = best_model.get_params()
    
    # Make predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    return best_model, y_train_pred, y_test_pred, best_params


def evaluate_model(y_train, y_train_pred, y_test, y_test_pred):
    """
    Evaluate model performance
    """
    print("\nModel Evaluation:")
    
    # Training set metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Test set metrics
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"  Training Set:")
    print(f"    RMSE: {train_rmse:.4f}")
    print(f"    MAE:  {train_mae:.4f}")
    print(f"    R²:   {train_r2:.4f}")
    
    print(f"  Test Set:")
    print(f"    RMSE: {test_rmse:.4f}")
    print(f"    MAE:  {test_mae:.4f}")
    print(f"    R²:   {test_r2:.4f}")
    
    return {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }


def plot_feature_importance(model, feature_names, top_n=15, save_path=None):
    """
    Plot and save feature importance
    """
    print(f"\nPlotting top {top_n} feature importances...")
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(top_n)
    sns.barplot(data=top_features, x='importance', y='feature')
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Feature importance plot saved to: {save_path}")
    
    plt.close()  # Close the figure to free memory
    
    return feature_importance


def plot_predictions(y_test, y_test_pred, save_path=None):
    """
    Plot actual vs predicted values
    """
    print("\nPlotting predictions...")
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Drug Response')
    plt.ylabel('Predicted Drug Response')
    plt.title('Actual vs Predicted Drug Response')
    
    # Add R² to plot
    r2 = r2_score(y_test, y_test_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Predictions plot saved to: {save_path}")
    
    plt.close()  # Close the figure to free memory


def cross_validate_model(X, y, model_params, cv_folds=5):
    """
    Perform cross-validation
    """
    print(f"\nPerforming {cv_folds}-fold cross-validation...")
    
    model = xgb.XGBRegressor(**model_params)
    cv_scores = cross_val_score(model, X, y.values.ravel(), cv=cv_folds, scoring='r2')
    
    print(f"  CV R² scores: {cv_scores}")
    print(f"  Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores


def save_model_and_results(model, results, feature_importance, output_dir):
    """
    Save model and results
    """
    print(f"\nSaving model and results to: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'xgboost_model.pkl')
    joblib.dump(model, model_path)
    print(f"  Model saved to: {model_path}")
    
    # Save results
    results_path = os.path.join(output_dir, 'model_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}")
    
    # Save feature importance
    importance_path = os.path.join(output_dir, 'feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)
    print(f"  Feature importance saved to: {importance_path}")


def main():
    """
    Main execution function
    """
    # Get input paths from environment variables or command line
    features_path = os.getenv('FEATURES_PATH', 'data/feature/X_repeated.csv')
    targets_path = os.getenv('TARGETS_PATH', 'data/feature/y_flattened.csv')
    output_dir = os.getenv('MODEL_OUTPUT_DIR', 'data/models')
    
    # Allow command line override
    if len(sys.argv) > 1:
        features_path = sys.argv[1]
    if len(sys.argv) > 2:
        targets_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
    print("XGBoost Drug Response Prediction Model")
    print("=" * 50)
    print(f"Features: {features_path}")
    print(f"Targets: {targets_path}")
    print(f"Output: {output_dir}")
    
    # Load data
    X, y = load_training_data(features_path, targets_path)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    # Train model
    model, y_train_pred, y_test_pred, best_params = train_xgboost_model(
        X_train, y_train, X_test, y_test, use_grid_search=True
    )
    
    # Evaluate model
    results = evaluate_model(y_train, y_train_pred, y_test, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_validate_model(X, y, best_params)
    results['cv_r2_mean'] = cv_scores.mean()
    results['cv_r2_std'] = cv_scores.std()
    
    # Feature importance
    feature_importance = plot_feature_importance(
        model, X.columns, top_n=15, 
        save_path=os.path.join(output_dir, 'feature_importance.png')
    )
    
    # Predictions plot
    plot_predictions(
        y_test.values.ravel(), y_test_pred,
        save_path=os.path.join(output_dir, 'predictions_plot.png')
    )
    
    # Save model and results
    save_model_and_results(model, results, feature_importance, output_dir)
    
    print(f"\nTraining complete!")
    print(f"Final model performance:")
    print(f"  Test R²: {results['test_r2']:.4f}")
    print(f"  Test RMSE: {results['test_rmse']:.4f}")
    print(f"  Cross-validation R²: {results['cv_r2_mean']:.4f} (+/- {results['cv_r2_std']:.4f})")


if __name__ == '__main__':
    main()
