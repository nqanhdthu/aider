import numpy as np
import os
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import yaml
import json


def train_catboost_with_features():
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

    # CatBoost doesn't require feature scaling, but we'll keep the scaler for consistency
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use original features for CatBoost (tree-based algorithms don't need scaling)
    X_train_use = X_train
    X_test_use = X_test

    # Train CatBoost with default parameters
    print("Training CatBoost...")
    catboost_model = CatBoostClassifier(
        random_state=42,
        verbose=False
    )
    catboost_model.fit(X_train_use, y_train)

    print(f"Model parameters: iterations={catboost_model.get_param('iterations')}, depth={catboost_model.get_param('depth')}")

    # Evaluate on train set (training accuracy)
    y_train_pred = catboost_model.predict(X_train_use)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {train_accuracy:.4f}")

    # Evaluate on test set (final evaluation)
    y_test_pred = catboost_model.predict(X_test_use)
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
    if hasattr(catboost_model, 'feature_importances_'):
        print(f"\n🔍 Feature Importance Analysis:")
        importances = catboost_model.feature_importances_
        # Get top 10 most important features
        top_indices = np.argsort(importances)[-10:][::-1]
        print("Top 10 Most Important Features:")
        for i, idx in enumerate(top_indices, 1):
            print(f"  {i}. Feature {idx}: {importances[idx]:.4f}")

    # Display CatBoost complexity
    print(f"\n🐱 CatBoost Complexity:")
    print(f"Iterations: {catboost_model.get_param('iterations')}")
    print(f"Depth: {catboost_model.get_param('depth')}")
    print(f"Learning rate: {catboost_model.get_param('learning_rate')}")
    print(f"L2 leaf reg: {catboost_model.get_param('l2_leaf_reg')}")

    # Save the trained model, scaler, and results
    model_dir = config['log_dir']
    os.makedirs(model_dir, exist_ok=True)

    # Save model and scaler
    with open(os.path.join(model_dir, "catboost_model.pkl"), 'wb') as f:
        pickle.dump(catboost_model, f)
    with open(os.path.join(model_dir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    # Save classification report as JSON
    with open(os.path.join(model_dir, "catboost_classification_report.json"), 'w') as f:
        json.dump(class_report_dict, f, indent=4)

    # Save classification report as text
    with open(os.path.join(model_dir, "catboost_classification_report.txt"), 'w') as f:
        f.write("CatBoost Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model parameters: iterations={catboost_model.get_param('iterations')}, depth={catboost_model.get_param('depth')}\n")
        f.write(f"Train accuracy: {train_accuracy:.4f}\n")
        f.write(f"Test accuracy: {test_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)

        # Add feature importance to the report
        if hasattr(catboost_model, 'feature_importances_'):
            f.write(f"\n\nTop 10 Feature Importances:\n")
            for i, idx in enumerate(top_indices, 1):
                f.write(f"  {i}. Feature {idx}: {importances[idx]:.4f}\n")

        f.write(f"\nCatBoost Complexity:\n")
        f.write(f"Iterations: {catboost_model.get_param('iterations')}\n")
        f.write(f"Depth: {catboost_model.get_param('depth')}\n")
        f.write(f"Learning rate: {catboost_model.get_param('learning_rate')}\n")
        f.write(f"L2 leaf reg: {catboost_model.get_param('l2_leaf_reg')}\n")

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    with open(os.path.join(model_dir, "catboost_confusion_matrix.pkl"), 'wb') as f:
        pickle.dump(cm, f)

    # Save detailed results dictionary
    results = {
        'model_name': 'CatBoost',
        'model_params': {
            'iterations': catboost_model.get_param('iterations'),
            'depth': catboost_model.get_param('depth'),
            'learning_rate': catboost_model.get_param('learning_rate'),
            'l2_leaf_reg': catboost_model.get_param('l2_leaf_reg'),
            'border_count': catboost_model.get_param('border_count'),
            'bagging_temperature': catboost_model.get_param('bagging_temperature')
        },
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'feature_importances': importances.tolist() if hasattr(catboost_model, 'feature_importances_') else None,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report_dict
    }

    with open(os.path.join(model_dir, "catboost_results.pkl"), 'wb') as f:
        pickle.dump(results, f)

    print(f"✅ CatBoost model, scaler, and reports saved to {model_dir}")
    print(f"   - Model: catboost_model.pkl")
    print(f"   - Scaler: scaler.pkl")
    print(f"   - Classification Report (JSON): catboost_classification_report.json")
    print(f"   - Classification Report (Text): catboost_classification_report.txt")
    print(f"   - Confusion Matrix: catboost_confusion_matrix.pkl")
    print(f"   - Complete Results: catboost_results.pkl")

    return catboost_model, scaler, results


if __name__ == "__main__":
    train_catboost_with_features()