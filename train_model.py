import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

csv_path = r"E:\Major Project MCA\isl_landmarks.csv"
model_path = r"E:\Major Project MCA\isl_model.pkl"

df = pd.read_csv(csv_path, low_memory=False)

df["label"] = df["label"].astype(str)

X = df.drop(columns=["label"])
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, model_path)
print("Saved model to:", model_path)