"""
Run this script ONCE locally to train and save your model.
It generates model.pkl and scaler.pkl which Vercel will use.

Usage:
    python train_and_save.py
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("breast_cancer.csv")

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
df['diagnosis'] = df['diagnosis'].apply(lambda val: 1 if val == 'M' else 0)
df.drop('id', axis=1, inplace=True)

# ── 3. Feature selection (drop highly correlated, threshold 0.92) ─────────────
corr_matrix = df.corr().abs()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask)
to_drop = [x for x in tri_df.columns if any(tri_df[x] > 0.92)]
df = df.drop(to_drop, axis=1)
print(f"Features after selection: {df.shape[1] - 1}")
print("Remaining features:", list(df.drop('diagnosis', axis=1).columns))

# ── 4. Split ──────────────────────────────────────────────────────────────────
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ── 5. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 6. Train best model (SVC with tuned params from notebook) ────────────────
model = SVC(C=15, gamma=0.01, probability=True)
model.fit(X_train_scaled, y_train)

from sklearn.metrics import accuracy_score
print(f"Test Accuracy: {accuracy_score(y_test, model.predict(X_test_scaled)):.4f}")

# ── 7. Save ───────────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ model.pkl and scaler.pkl saved successfully!")
print("Now deploy your project to Vercel.")
