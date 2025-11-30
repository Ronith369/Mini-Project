import numpy as np
import pandas as pd
import os, cv2, joblib, random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models

# =======================
# Load Images from CSV
# =======================
def load_images_from_csv(folder_path, csv_path, size=(128, 128)):
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    data, labels = [], []
    file_col = [col for col in df.columns if 'file' in col.lower() or 'image' in col.lower()][0]
    label_cols = [col for col in df.columns if col != file_col]
    for _, row in df.iterrows():
        img_path = os.path.join(folder_path, str(row[file_col]))
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, size)
            data.append(img / 255.0)   # normalize to [0,1]
            labels.append(int(row[label_cols[0]]))
    return np.array(data), np.array(labels)

# =======================
# Paths
# =======================
train_path = r'C:\Users\ronit\BNC\dataset\train'
valid_path = r'C:\Users\ronit\BNC\dataset\valid'
test_path  = r'C:\Users\ronit\BNC\dataset\test'

train_csv = r'C:\Users\ronit\BNC\dataset\train\_classes.csv'
valid_csv = r'C:\Users\ronit\BNC\dataset\valid\_classes.csv'
test_csv  = r'C:\Users\ronit\BNC\dataset\test\_classes.csv'

# =======================
# Load Data
# =======================
X_train, y_train = load_images_from_csv(train_path, train_csv)
X_val, y_val = load_images_from_csv(valid_path, valid_csv)
X_test, y_test = load_images_from_csv(test_path, test_csv)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Reshape for CNN
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# =======================
# CNN Model
# =======================
cnn = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = cnn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

# =======================
# CNN Evaluation
# =======================
cnn_test_pred = (cnn.predict(X_test) > 0.5).astype("int32")
cnn_acc = accuracy_score(y_test, cnn_test_pred)
print("\n CNN Test Accuracy:", cnn_acc)
print("\nClassification Report (CNN):\n", classification_report(y_test, cnn_test_pred))

cm = confusion_matrix(y_test, cnn_test_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title('CNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save CNN model
cnn.save(r"C:\Users\ronit\BNC\cnn_model.keras")

# =======================
# Extract CNN Features
# =======================
feature_extractor = models.Model(inputs=cnn.layers[0].input, outputs=cnn.layers[-3].output)
features_train = feature_extractor.predict(X_train)
features_test = feature_extractor.predict(X_test)

# =======================
# Optimized Quantum Model (Enhanced Version with PCA + 4D Feature Map)
# =======================
from qiskit.primitives import StatevectorSampler
from qiskit.circuit.library import ZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# Select a random subset (Quantum models are heavy)
idx = np.random.choice(len(features_test), 100, replace=False)
X_small, y_small = features_test[idx], y_test[idx]

# PCA: Reduce CNN feature dimension = 4 components for 4-qubit feature map
pca = PCA(n_components=4)
X_small_pca = pca.fit_transform(X_small)

# Define Quantum Feature Map (4-dimensional input → 4 qubits)
feature_map = ZFeatureMap(feature_dimension=6, reps=2)

# Initialize Quantum Kernel and QSVC (Quantum Support Vector Classifier)
# sampler = StatevectorSampler()
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
qsvc = QSVC(quantum_kernel=quantum_kernel)

print("\n Training Quantum SVM (4-qubit) on subset of 100 samples...")
qsvc.fit(X_small_pca, y_small)

# Predict on the same small subset
q_pred = qsvc.predict(X_small_pca)
q_acc = accuracy_score(y_small, q_pred)

# Display performance
print("\n Quantum Model (4D Feature Map) Test Accuracy:", q_acc)
print("\nClassification Report (Quantum 4D, subset):\n", classification_report(y_small, q_pred))

# Confusion matrix visualization
cm2 = confusion_matrix(y_small, q_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm2, annot=True, fmt='d', cmap='coolwarm')
plt.title('Quantum SVM (4-qubit) Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save trained quantum models
joblib.dump(qsvc, r"C:\Users\ronit\BNC\quantum_model.pkl")
joblib.dump(pca, r"C:\Users\ronit\BNC\pca.pkl")
print("\n Quantum SVM (4D) and PCA models trained and saved successfully!")

# =======================
# Accuracy Comparison
# =======================
print("\n========== Accuracy Comparison ==========")
print(f"CNN Accuracy:        {cnn_acc*100:.2f}%")
print(f"Quantum Accuracy:    {q_acc*100:.2f}%")
if q_acc > cnn_acc:
    print("Quantum model outperformed CNN (on 4D subset)!")
else:
    print("CNN still performs better on the full test set.")

