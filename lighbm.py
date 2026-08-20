import numpy as np
import os
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import yaml
import json


def train_lightgbm_with_features():
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

    # LightGBM doesn't require feature scaling, but we'll keep the scaler for consistency
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use original features for LightGBM (tree-based algorithms don't need scaling)
    X_train_use = X_train
    X_test_use = X_test

    # Train LightGBM with default parameters
    print("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train_use, y_train)

    print(f"Model parameters: n_estimators={lgb_model.n_estimators}, max_depth={lgb_model.max_depth}")

    # Evaluate on train set (training accuracy)
    y_train_pred = lgb_model.predict(X_train_use)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {train_accuracy:.4f}")

    # Evaluate on test set (final evaluation)
    y_test_pred = lgb_model.predict(X_test_use)
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
    if hasattr(lgb_model, 'feature_importances_'):
        print(f"\n🔍 Feature Importance Analysis:")
        importances = lgb_model.feature_importances_
        # Get top 10 most important features
        top_indices = np.argsort(importances)[-10:][::-1]
        print("Top 10 Most Important Features:")
        for i, idx in enumerate(top_indices, 1):
            print(f"  {i}. Feature {idx}: {importances[idx]:.4f}")

    # Display LightGBM complexity
    print(f"\n💡 LightGBM Complexity:")
    print(f"Number of estimators: {lgb_model.n_estimators}")
    print(f"Max depth: {lgb_model.max_depth}")
    print(f"Learning rate: {lgb_model.learning_rate}")
    print(f"Num leaves: {lgb_model.num_leaves}")

    # Save the trained model, scaler, and results
    model_dir = config['log_dir']
    os.makedirs(model_dir, exist_ok=True)

    # Save model and scaler
    with open(os.path.join(model_dir, "lightgbm_model.pkl"), 'wb') as f:
        pickle.dump(lgb_model, f)
    with open(os.path.join(model_dir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    # Save classification report as JSON
    with open(os.path.join(model_dir, "lightgbm_classification_report.json"), 'w') as f:
        json.dump(class_report_dict, f, indent=4)

    # Save classification report as text
    with open(os.path.join(model_dir, "lightgbm_classification_report.txt"), 'w') as f:
        f.write("LightGBM Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model parameters: n_estimators={lgb_model.n_estimators}, max_depth={lgb_model.max_depth}\n")
        f.write(f"Train accuracy: {train_accuracy:.4f}\n")
        f.write(f"Test accuracy: {test_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)

        # Add feature importance to the report
        if hasattr(lgb_model, 'feature_importances_'):
            f.write(f"\n\nTop 10 Feature Importances:\n")
            for i, idx in enumerate(top_indices, 1):
                f.write(f"  {i}. Feature {idx}: {importances[idx]:.4f}\n")

        f.write(f"\nLightGBM Complexity:\n")
        f.write(f"Number of estimators: {lgb_model.n_estimators}\n")
        f.write(f"Max depth: {lgb_model.max_depth}\n")
        f.write(f"Learning rate: {lgb_model.learning_rate}\n")
        f.write(f"Num leaves: {lgb_model.num_leaves}\n")

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    with open(os.path.join(model_dir, "lightgbm_confusion_matrix.pkl"), 'wb') as f:
        pickle.dump(cm, f)

    # Save detailed results dictionary
    results = {
        'model_name': 'LightGBM',
        'model_params': {
            'n_estimators': lgb_model.n_estimators,
            'max_depth': lgb_model.max_depth,
            'learning_rate': lgb_model.learning_rate,
            'num_leaves': lgb_model.num_leaves,
            'subsample': lgb_model.subsample,
            'colsample_bytree': lgb_model.colsample_bytree,
            'reg_alpha': lgb_model.reg_alpha,
            'reg_lambda': lgb_model.reg_lambda
        },
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'feature_importances': importances.tolist() if hasattr(lgb_model, 'feature_importances_') else None,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report_dict
    }

    with open(os.path.join(model_dir, "lightgbm_results.pkl"), 'wb') as f:
        pickle.dump(results, f)

    print(f"✅ LightGBM model, scaler, and reports saved to {model_dir}")
    print(f"   - Model: lightgbm_model.pkl")
    print(f"   - Scaler: scaler.pkl")
    print(f"   - Classification Report (JSON): lightgbm_classification_report.json")
    print(f"   - Classification Report (Text): lightgbm_classification_report.txt")
    print(f"   - Confusion Matrix: lightgbm_confusion_matrix.pkl")
    print(f"   - Complete Results: lightgbm_results.pkl")

    return lgb_model, scaler, results


if __name__ == "__main__":
    train_lightgbm_with_features()