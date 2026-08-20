import numpy as np
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle
import yaml
import time


def train_multiple_classifiers():
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

    # Normalize features for algorithms that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define classifiers and their parameter grids
    classifiers = {
        # 'SVM': {
        #     'model': SVC(random_state=42),
        #     'params': {
        #         'C': [0.1, 1, 10, 100],
        #         'kernel': ['linear', 'rbf'],
        #         'gamma': ['scale', 'auto']
        #     },
        #     'use_scaled': True
        # },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'use_scaled': False
        },
        # 'Decision Tree': {
        #     'model': DecisionTreeClassifier(random_state=42),
        #     'params': {
        #         'max_depth': [10, 20, 30, None],
        #         'min_samples_split': [2, 5, 10],
        #         'min_samples_leaf': [1, 2, 4]
        #     },
        #     'use_scaled': False
        # },
        # 'KNN': {
        #     'model': KNeighborsClassifier(),
        #     'params': {
        #         'n_neighbors': [3, 5, 7, 9, 11],
        #         'weights': ['uniform', 'distance'],
        #         'metric': ['euclidean', 'manhattan']
        #     },
        #     'use_scaled': True
        # },
        # 'Logistic Regression': {
        #     'model': LogisticRegression(random_state=42, max_iter=1000),
        #     'params': {
        #         'C': [0.1, 1, 10, 100],
        #         'penalty': ['l2'],
        #         'solver': ['liblinear', 'lbfgs']
        #     },
        #     'use_scaled': True
        # },
        # 'XGBoost': {
        #     'model': XGBClassifier(random_state=42, eval_metric='mlogloss'),
        #     'params': {
        #         'n_estimators': [100, 200, 300],
        #         'max_depth': [3, 6, 9],
        #         'learning_rate': [0.01, 0.1, 0.2],
        #         'subsample': [0.8, 1.0]
        #     },
        #     'use_scaled': False
        # }
    }

    results = {}
    best_models = {}

    print("=" * 80)
    print("TRAINING MULTIPLE CLASSIFIERS WITH XGBOOST")
    print("=" * 80)

    for clf_name, clf_info in classifiers.items():
        print(f"\n🔄 Training {clf_name}...")
        start_time = time.time()

        # Choose scaled or original features
        X_train_use = X_train_scaled if clf_info['use_scaled'] else X_train
        X_test_use = X_test_scaled if clf_info['use_scaled'] else X_test

        # Grid search
        grid_search = GridSearchCV(
            clf_info['model'],
            clf_info['params'],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train_use, y_train)

        # Get best model
        best_model = grid_search.best_estimator_
        train_time = time.time() - start_time

        # Evaluate on train set
        y_train_pred = best_model.predict(X_train_use)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        # Evaluate on test set
        start_eval = time.time()
        y_test_pred = best_model.predict(X_test_use)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        eval_time = time.time() - start_eval

        # Store results
        results[clf_name] = {
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_time': train_time,
            'eval_time': eval_time
        }
        best_models[clf_name] = best_model

        print(f"✅ {clf_name} completed in {train_time:.2f}s")
        print(f"   Best params: {grid_search.best_params_}")
        print(f"   CV accuracy: {grid_search.best_score_:.4f}")
        print(f"   Train accuracy: {train_accuracy:.4f}")
        print(f"   Test accuracy: {test_accuracy:.4f}")

    # Print summary results
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Classifier':<20} {'CV Score':<10} {'Train Acc':<12} {'Test Acc':<12} {'Time(s)':<10}")
    print("-" * 80)

    for clf_name, result in results.items():
        print(f"{clf_name:<20} {result['cv_score']:<10.4f} {result['train_accuracy']:<12.4f} "
              f"{result['test_accuracy']:<12.4f} {result['train_time']:<10.2f}")

    # Find best classifier
    best_clf_name = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    best_result = results[best_clf_name]

    print(f"\n🏆 Best classifier: {best_clf_name} (Test Accuracy: {best_result['test_accuracy']:.4f})")

    # Show detailed classification report for best classifier
    print(f"\n📊 Detailed Classification Report for {best_clf_name}:")
    X_test_best = X_test_scaled if classifiers[best_clf_name]['use_scaled'] else X_test
    y_test_pred_best = best_models[best_clf_name].predict(X_test_best)
    print(classification_report(y_test, y_test_pred_best, digits=4))

    # Save all models and results
    model_dir = config['log_dir']
    os.makedirs(model_dir, exist_ok=True)

    # Save scaler
    with open(os.path.join(model_dir, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    # Save all models
    for clf_name, model in best_models.items():
        model_filename = f"{clf_name.lower().replace(' ', '_')}_model.pkl"
        with open(os.path.join(model_dir, model_filename), 'wb') as f:
            pickle.dump(model, f)

    # Save results summary
    with open(os.path.join(model_dir, "classifier_results.pkl"), 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✅ All models and results saved to {model_dir}")

    return best_models, results, scaler


if __name__ == "__main__":
    train_multiple_classifiers()