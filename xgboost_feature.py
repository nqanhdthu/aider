import numpy as np
import os
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import yaml
import json


def train_xgboost_with_features():
    # Load config
    with open("train_config.yaml") as f:
        config = yaml.safe_load(f)

    # Load train+val features and labels for training
    X_train = np.load(os.path.join(config['log_dir'], "cbam_m_train_val_features.npy"))
    y_train = np.load(os.path.join(config['log_dir'], "cbam_m_train_val_labels.npy"))

    # Load test features and labels for evaluation
    X_test = np.load(os.path.join(config['log_dir'], "cbam_m_test_features.npy"))
    y_test = np.load(os.path.join(config['log_dir'], "cbam_m_test_labels.npy"))

    print(f"Train features: {X_train.shape}, Train labels: {y_train.shape}")
    print(f"Test features: {X_test.shape}, Test labels: {y_test.shape}")

    # XGBoost doesn't require feature scaling, but we'll keep the scaler for consistency
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use original features for XGBoost (tree-based algorithms don't need scaling)
    X_train_use = X_train
    X_test_use = X_test

    # Train XGBoost with default parameters
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train_use, y_train)

    print(f"Model parameters: n_estimators={xgb_model.n_estimators}, max_depth={xgb_model.max_depth}")

    # Evaluate on train set (training accuracy)
    y_train_pred = xgb_model.predict(X_train_use)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {train_accuracy:.4f}")

    # Evaluate on test set (final evaluation)
    y_test_pred = xgb_model.predict(X_test_use)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\n📊 Final Results:")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Generate and print classification report
    print("\nTest Classification Report:")
    class_report = classification_report(y_test, y_test_pred, digits=4)
    print(class_report)

    # Generate classification report as dictionary for saving
    class_report_dict = classification_report(y_test, y_test_pred, digits=4, output_dict=True)

    # Display feature importance
    if hasattr(xgb_model, 'feature_importances_'):
        print(f"\n🔍 Feature Importance Analysis:")
        importances = xgb_model.feature_importances_
        # Get top 10 most important features
        top_indices = np.argsort(importances)[-10:][::-1]
        print("Top 10 Most Important Features:")
        for i, idx in enumerate(top_indices, 1):
            print(f"  {i}. Feature {idx}: {importances[idx]:.4f}")

    # Display XGBoost complexity
    print(f"\n🚀 XGBoost Complexity:")
    print(f"Number of estimators: {xgb_model.n_estimators}")
    print(f"Max depth: {xgb_model.max_depth}")
    print(f"Learning rate: {xgb_model.learning_rate}")
    print(f"Subsample: {xgb_model.subsample}")

    # Save the trained model, scaler, and results
    model_dir = config['log_dir']
    os.makedirs(model_dir, exist_ok=True)

    # Save model and scaler
    with open(os.path.join(model_dir, "xgboost_model.pkl"), 'wb') as f:
        pickle.dump(xgb_model, f)
    with open(os.path.join(model_dir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    # Save classification report as JSON
    with open(os.path.join(model_dir, "xgboost_classification_report.json"), 'w') as f:
        json.dump(class_report_dict, f, indent=4)

    # Save classification report as text
    with open(os.path.join(model_dir, "xgboost_classification_report.txt"), 'w') as f:
        f.write("XGBoost Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model parameters: n_estimators={xgb_model.n_estimators}, max_depth={xgb_model.max_depth}\n")
        f.write(f"Train accuracy: {train_accuracy:.4f}\n")
        f.write(f"Test accuracy: {test_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)

        # Add feature importance to the report
        if hasattr(xgb_model, 'feature_importances_'):
            f.write(f"\n\nTop 10 Feature Importances:\n")
            for i, idx in enumerate(top_indices, 1):
                f.write(f"  {i}. Feature {idx}: {importances[idx]:.4f}\n")

        f.write(f"\nXGBoost Complexity:\n")
        f.write(f"Number of estimators: {xgb_model.n_estimators}\n")
        f.write(f"Max depth: {xgb_model.max_depth}\n")
        f.write(f"Learning rate: {xgb_model.learning_rate}\n")
        f.write(f"Subsample: {xgb_model.subsample}\n")

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    with open(os.path.join(model_dir, "xgboost_confusion_matrix.pkl"), 'wb') as f:
        pickle.dump(cm, f)

    # Save detailed results dictionary
    results = {
        'model_name': 'XGBoost',
        'model_params': {
            'n_estimators': xgb_model.n_estimators,
            'max_depth': xgb_model.max_depth,
            'learning_rate': xgb_model.learning_rate,
            'subsample': xgb_model.subsample,
            'colsample_bytree': xgb_model.colsample_bytree,
            'reg_alpha': xgb_model.reg_alpha,
            'reg_lambda': xgb_model.reg_lambda
        },
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'feature_importances': importances.tolist() if hasattr(xgb_model, 'feature_importances_') else None,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report_dict
    }

    with open(os.path.join(model_dir, "xgboost_results.pkl"), 'wb') as f:
        pickle.dump(results, f)

    print(f"✅ XGBoost model, scaler, and reports saved to {model_dir}")
    print(f"   - Model: xgboost_model.pkl")
    print(f"   - Scaler: scaler.pkl")
    print(f"   - Classification Report (JSON): xgboost_classification_report.json")
    print(f"   - Classification Report (Text): xgboost_classification_report.txt")
    print(f"   - Confusion Matrix: xgboost_confusion_matrix.pkl")
    print(f"   - Complete Results: xgboost_results.pkl")

    return xgb_model, scaler, results


if __name__ == "__main__":
    train_xgboost_with_features()