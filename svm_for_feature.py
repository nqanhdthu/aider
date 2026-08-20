import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle
import yaml


def train_svm_with_features():
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

    # Normalize features using train set statistics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use same scaler for test

    # Grid search for best SVM parameters on train set
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    print("Training SVM with grid search...")
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Get best model
    best_svm = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    # Evaluate on train set (training accuracy)
    y_train_pred = best_svm.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {train_accuracy:.4f}")

    # Evaluate on test set (final evaluation)
    y_test_pred = best_svm.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"\n📊 Final Results:")
    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred,digits=4))

    # Save the trained model and scaler
    model_dir = config['log_dir']
    with open(os.path.join(model_dir, "svm_model.pkl"), 'wb') as f:
        pickle.dump(best_svm, f)
    with open(os.path.join(model_dir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    print(f"✅ SVM model and scaler saved to {model_dir}")

    return best_svm, scaler


if __name__ == "__main__":
    train_svm_with_features()