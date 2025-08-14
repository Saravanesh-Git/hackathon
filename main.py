import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import urllib.parse
import joblib
import os


# -------- FEATURE EXTRACTION --------
def extract_features(url: str):
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or ''
    features = {
        'url_length': len(url),
        'is_https': int(parsed.scheme == 'https'),
        'contains_verify': int('verify' in url.lower()),
        'contains_login': int('login' in url.lower()),
        'num_subdomains': max(len(hostname.split('.')) - 2, 0),
        'contains_fake': int('fake' in url.lower()),
        'contains_bank': int('bank' in url.lower()),
    }
    return features


# -------- MAIN TRAINING FUNCTION --------
def main():
    dataset_path = "dataset.csv.xls"  # <-- Change if needed
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Load dataset
    data = pd.read_csv(dataset_path)
    print("Dataset loaded:", data.shape)
    if 'url' not in data.columns or 'label' not in data.columns:
        raise ValueError("Dataset must have 'url' and 'label' columns.")

    # Duplicate check
    print("Duplicate rows:", data.duplicated().sum())

    # Feature engineering
    features_df = pd.DataFrame([extract_features(u) for u in data['url']])
    print("Extracted features:", features_df.columns.tolist())

    X = features_df.fillna(0)
    y = data['label']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------- Train models --------
    print("\nTraining models...")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                             solver='adam', max_iter=500, random_state=42)
    nn_model.fit(X_train_scaled, y_train)

    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)

    # Ensemble
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('nn', nn_model),
            ('xgb', xgb_model)
        ],
        voting='hard'
    )
    ensemble_model.fit(X_train_scaled, y_train)

    # -------- Evaluate --------
    print("\nEvaluating Ensemble...")
    ensemble_pred = ensemble_model.predict(X_test_scaled)
    print(f"Test Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test, ensemble_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, ensemble_pred))

    cv_scores = cross_val_score(ensemble_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # -------- Visualization --------
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, ensemble_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Phishing'])
    disp.plot(ax=ax)
    plt.title("Ensemble Confusion Matrix")
    plt.show()

    # -------- Save model & scaler --------
    joblib.dump(ensemble_model, "phishing_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\nModel and scaler saved: phishing_model.pkl, scaler.pkl")

    # -------- Test example --------
    def predict_url(url):
        feats = extract_features(url)
        df = pd.DataFrame([feats]).fillna(0)
        scaled = scaler.transform(df)
        pred = ensemble_model.predict(scaled)[0]
        return "Phishing" if pred == 1 else "Legitimate"

    test_url = "https://google.com"
    print(f"Prediction for {test_url}: {predict_url(test_url)}")


if __name__ == "__main__":
    main()
