import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle
import yaml
import json


def train_decision_tree_with_features():
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

    # Decision Tree doesn't require feature scaling, but we'll keep the scaler for consistency
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use original features for Decision Tree (tree-based algorithms don't need scaling)
    X_train_use = X_train
    X_test_use = X_test

    # Grid search for best Decision Tree parameters on train set
    param_grid = {
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None]
    }

    print("Training Decision Tree with grid search...")
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_use, y_train)

    # Get best model
    best_dt = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    # Evaluate on train set (training accuracy)
    y_train_pred = best_dt.predict(X_train_use)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {train_accuracy:.4f}")

    # Evaluate on test set (final evaluation)
    y_test_pred = best_dt.predict(X_test_use)
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
    if hasattr(best_dt, 'feature_importances_'):
        print(f"\n🔍 Feature Importance Analysis:")
        importances = best_dt.feature_importances_
        # Get top 10 most important features
        top_indices = np.argsort(importances)[-10:][::-1]
        print("Top 10 Most Important Features:")
        for i, idx in enumerate(top_indices, 1):
            print(f"  {i}. Feature {idx}: {importances[idx]:.4f}")

    # Display tree complexity
    print(f"\n🌳 Tree Complexity:")
    print(f"Tree depth: {best_dt.get_depth()}")
    print(f"Number of leaves: {best_dt.get_n_leaves()}")

    # Save the trained model, scaler, and results
    model_dir = config['log_dir']
    os.makedirs(model_dir, exist_ok=True)

    # Save model and scaler
    with open(os.path.join(model_dir, "decision_tree_model.pkl"), 'wb') as f:
        pickle.dump(best_dt, f)
    with open(os.path.join(model_dir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    # Save classification report as JSON
    with open(os.path.join(model_dir, "decision_tree_classification_report.json"), 'w') as f:
        json.dump(class_report_dict, f, indent=4)

    # Save classification report as text
    with open(os.path.join(model_dir, "decision_tree_classification_report.txt"), 'w') as f:
        f.write("Decision Tree Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best parameters: {grid_search.best_params_}\n")
        f.write(f"Best CV accuracy: {grid_search.best_score_:.4f}\n")
        f.write(f"Train accuracy: {train_accuracy:.4f}\n")
        f.write(f"Test accuracy: {test_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)

        # Add feature importance to the report
        if hasattr(best_dt, 'feature_importances_'):
            f.write(f"\n\nTop 10 Feature Importances:\n")
            for i, idx in enumerate(top_indices, 1):
                f.write(f"  {i}. Feature {idx}: {importances[idx]:.4f}\n")

        f.write(f"\nTree Complexity:\n")
        f.write(f"Tree depth: {best_dt.get_depth()}\n")
        f.write(f"Number of leaves: {best_dt.get_n_leaves()}\n")

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    with open(os.path.join(model_dir, "decision_tree_confusion_matrix.pkl"), 'wb') as f:
        pickle.dump(cm, f)

    # Save detailed results dictionary
    results = {
        'model_name': 'Decision Tree',
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'tree_depth': best_dt.get_depth(),
        'n_leaves': best_dt.get_n_leaves(),
        'feature_importances': importances.tolist() if hasattr(best_dt, 'feature_importances_') else None,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report_dict
    }

    with open(os.path.join(model_dir, "decision_tree_results.pkl"), 'wb') as f:
        pickle.dump(results, f)

    print(f"✅ Decision Tree model, scaler, and reports saved to {model_dir}")
    print(f"   - Model: decision_tree_model.pkl")
    print(f"   - Scaler: scaler.pkl")
    print(f"   - Classification Report (JSON): decision_tree_classification_report.json")
    print(f"   - Classification Report (Text): decision_tree_classification_report.txt")
    print(f"   - Confusion Matrix: decision_tree_confusion_matrix.pkl")
    print(f"   - Complete Results: decision_tree_results.pkl")

    return best_dt, scaler, results


if __name__ == "__main__":
    train_decision_tree_with_features()