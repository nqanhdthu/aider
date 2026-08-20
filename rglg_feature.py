import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import yaml
import json


def train_logistic_regression_with_features():
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

    # Normalize features using train set statistics (required for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression with default parameters
    print("Training Logistic Regression...")
    #lr = LogisticRegression(random_state=42, max_iter=1000)
    lr = LogisticRegression(solver='liblinear', max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    #print(f"Model parameters: C={lr.C}, penalty={lr.penalty}, solver={lr.solver}")

    # Evaluate on train set (training accuracy)
    y_train_pred = lr.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {train_accuracy:.4f}")

    # Evaluate on test set (final evaluation)
    y_test_pred = lr.predict(X_test_scaled)
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

    # Display coefficient information
    if hasattr(lr, 'coef_'):
        print(f"\n🔍 Model Coefficients Analysis:")
        coef = lr.coef_[0] if lr.coef_.ndim > 1 else lr.coef_
        # Get top 10 most important features (by absolute coefficient value)
        top_indices = np.argsort(np.abs(coef))[-10:][::-1]
        print("Top 10 Most Important Features (by |coefficient|):")
        for i, idx in enumerate(top_indices, 1):
            print(f"  {i}. Feature {idx}: {coef[idx]:.4f}")

    # Display model complexity
    print(f"\n📈 Logistic Regression Complexity:")
    print(f"Number of features: {X_train_scaled.shape[1]}")
    print(f"Regularization strength (C): {lr.C}")
    print(f"Penalty: {lr.penalty}")
    print(f"Solver: {lr.solver}")

    # Save the trained model, scaler, and results
    model_dir = config['log_dir']
    os.makedirs(model_dir, exist_ok=True)

    # Save model and scaler
    with open(os.path.join(model_dir, "logistic_regression_model.pkl"), 'wb') as f:
        pickle.dump(lr, f)
    with open(os.path.join(model_dir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    # Save classification report as JSON
    with open(os.path.join(model_dir, "logistic_regression_classification_report.json"), 'w') as f:
        json.dump(class_report_dict, f, indent=4)

    # Save classification report as text
    with open(os.path.join(model_dir, "logistic_regression_classification_report.txt"), 'w') as f:
        f.write("Logistic Regression Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model parameters: C={lr.C}, penalty={lr.penalty}, solver={lr.solver}\n")
        f.write(f"Train accuracy: {train_accuracy:.4f}\n")
        f.write(f"Test accuracy: {test_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)

        # Add coefficient information to the report
        if hasattr(lr, 'coef_'):
            f.write(f"\n\nTop 10 Feature Coefficients:\n")
            for i, idx in enumerate(top_indices, 1):
                f.write(f"  {i}. Feature {idx}: {coef[idx]:.4f}\n")

        f.write(f"\nModel Complexity:\n")
        f.write(f"Number of features: {X_train_scaled.shape[1]}\n")
        f.write(f"Regularization strength (C): {lr.C}\n")
        f.write(f"Penalty: {lr.penalty}\n")
        f.write(f"Solver: {lr.solver}\n")

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    with open(os.path.join(model_dir, "logistic_regression_confusion_matrix.pkl"), 'wb') as f:
        pickle.dump(cm, f)

    # Save detailed results dictionary
    results = {
        'model_name': 'Logistic Regression',
        'model_params': {
            'C': lr.C,
            'penalty': lr.penalty,
            'solver': lr.solver,
            'max_iter': lr.max_iter
        },
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'n_features': X_train_scaled.shape[1],
        'regularization_C': lr.C,
        'penalty': lr.penalty,
        'solver': lr.solver,
        'coefficients': coef.tolist() if hasattr(lr, 'coef_') else None,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report_dict
    }

    with open(os.path.join(model_dir, "logistic_regression_results.pkl"), 'wb') as f:
        pickle.dump(results, f)

    print(f"✅ Logistic Regression model, scaler, and reports saved to {model_dir}")
    print(f"   - Model: logistic_regression_model.pkl")
    print(f"   - Scaler: scaler.pkl")
    print(f"   - Classification Report (JSON): logistic_regression_classification_report.json")
    print(f"   - Classification Report (Text): logistic_regression_classification_report.txt")
    print(f"   - Confusion Matrix: logistic_regression_confusion_matrix.pkl")
    print(f"   - Complete Results: logistic_regression_results.pkl")

    return lr, scaler, results


if __name__ == "__main__":
    train_logistic_regression_with_features()