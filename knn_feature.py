import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import yaml
import json


def train_knn_with_features():
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

    # Normalize features using train set statistics (required for KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train KNN with default parameters
    print("Training K-Nearest Neighbors...")
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
    knn.fit(X_train_scaled, y_train)

    # Evaluate on train set (training accuracy)
    y_train_pred = knn.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {train_accuracy:.4f}")

    # Evaluate on test set (final evaluation)
    y_test_pred = knn.predict(X_test_scaled)
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

    # Display model complexity
    print(f"\n📈 K-Nearest Neighbors Complexity:")
    print(f"Number of features: {X_train_scaled.shape[1]}")
    print(f"Number of neighbors (k): {knn.n_neighbors}")
    print(f"Weight function: {knn.weights}")
    print(f"Distance metric: {knn.metric}")
    print(f"Training samples: {X_train_scaled.shape[0]}")

    # Save the trained model, scaler, and results
    model_dir = config['log_dir']
    os.makedirs(model_dir, exist_ok=True)

    # Save model and scaler
    with open(os.path.join(model_dir, "knn_model.pkl"), 'wb') as f:
        pickle.dump(knn, f)
    with open(os.path.join(model_dir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    # Save classification report as JSON
    with open(os.path.join(model_dir, "knn_classification_report.json"), 'w') as f:
        json.dump(class_report_dict, f, indent=4)

    # Save classification report as text
    with open(os.path.join(model_dir, "knn_classification_report.txt"), 'w') as f:
        f.write("K-Nearest Neighbors Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model parameters: k={knn.n_neighbors}, weights={knn.weights}, metric={knn.metric}\n")
        f.write(f"Train accuracy: {train_accuracy:.4f}\n")
        f.write(f"Test accuracy: {test_accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)

        f.write(f"\nModel Complexity:\n")
        f.write(f"Number of features: {X_train_scaled.shape[1]}\n")
        f.write(f"Number of neighbors (k): {knn.n_neighbors}\n")
        f.write(f"Weight function: {knn.weights}\n")
        f.write(f"Distance metric: {knn.metric}\n")
        f.write(f"Training samples: {X_train_scaled.shape[0]}\n")

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    with open(os.path.join(model_dir, "knn_confusion_matrix.pkl"), 'wb') as f:
        pickle.dump(cm, f)

    # Save detailed results dictionary
    results = {
        'model_name': 'K-Nearest Neighbors',
        'model_params': {
            'n_neighbors': knn.n_neighbors,
            'weights': knn.weights,
            'metric': knn.metric,
            'algorithm': knn.algorithm
        },
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'n_features': X_train_scaled.shape[1],
        'n_training_samples': X_train_scaled.shape[0],
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report_dict
    }

    with open(os.path.join(model_dir, "knn_results.pkl"), 'wb') as f:
        pickle.dump(results, f)

    print(f"✅ KNN model, scaler, and reports saved to {model_dir}")
    print(f"   - Model: knn_model.pkl")
    print(f"   - Scaler: scaler.pkl")
    print(f"   - Classification Report (JSON): knn_classification_report.json")
    print(f"   - Classification Report (Text): knn_classification_report.txt")
    print(f"   - Confusion Matrix: knn_confusion_matrix.pkl")
    print(f"   - Complete Results: knn_results.pkl")

    return knn, scaler, results


if __name__ == "__main__":
    train_knn_with_features()