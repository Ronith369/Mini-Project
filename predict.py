import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model

# ===========================
# Load Trained Models
# ===========================
cnn_model_path = r"C:\Users\ronit\BNC\cnn_model.keras"
quantum_model_path = r"C:\Users\ronit\BNC\quantum_model.pkl"
pca_model_path = r"C:\Users\ronit\BNC\pca.pkl"

cnn = load_model(cnn_model_path)
qsvc = joblib.load(quantum_model_path)
pca = joblib.load(pca_model_path)

# Feature extractor (same layer used in training)
feature_extractor = Model(inputs=cnn.layers[0].input, outputs=cnn.layers[-3].output)

print("✅ CNN, Quantum, and PCA models loaded successfully!")

# ===========================
# Helper Function: Estimate Stage
# ===========================
def estimate_stage(prob):
    if prob < 25:
        return "Stage 1"
    elif prob < 50:
        return "Stage 2"
    elif prob < 75:
        return "Stage 3"
    else:
        return "Stage 4"

# ===========================
# Prediction Function
# ===========================
def predict_image(image_path, img_size=(128, 128)):
    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("❌ Invalid image path or unreadable image!")

    # Resize and normalize
    img_resized = cv2.resize(img, img_size)
    img_norm = img_resized / 255.0

    # ---------------------------
    # CNN Prediction
    # ---------------------------
    cnn_input = img_norm.reshape(1, img_size[0], img_size[1], 1)
    cnn_prob = float(cnn.predict(cnn_input, verbose=0)[0][0])
    cnn_pred = 1 if cnn_prob >= 0.5 else 0
    cnn_conf = cnn_prob * 100

    # ---------------------------
    # Extract CNN features for Quantum model
    # ---------------------------
    cnn_features = feature_extractor.predict(cnn_input, verbose=0)

    # Apply PCA → Reduce features
    cnn_features_pca = pca.transform(cnn_features)

    # Quantum Prediction
    q_pred = qsvc.predict(cnn_features_pca)[0]
    q_score = qsvc.decision_function(cnn_features_pca)
    q_conf = 1 / (1 + np.exp(-abs(q_score[0]))) * 100

    # ---------------------------
    # Display Results
    # ---------------------------
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    print("\n================= 🧠 PREDICTION RESULTS =================")
    if cnn_pred == 1:
        print(f"🤖 CNN Prediction: Cancer Detected ({cnn_conf:.2f}% confidence)")
    else:
        print(f"🤖 CNN Prediction: Normal Bone ({100 - cnn_conf:.2f}% confidence)")

    if q_pred == 1:
        stage = estimate_stage(q_conf)
        print(f"⚛️ Quantum Prediction: Cancer Detected ({q_conf:.2f}% confidence)")
        print(f"   ➤ Estimated Stage: {stage}")
    else:
        print(f"⚛️ Quantum Prediction: Normal Bone ({100 - q_conf:.2f}% confidence)")

    # ---------------------------
    # Compare Models
    # ---------------------------
    print("\n================= 🔍 COMPARISON =================")
    if (cnn_pred == 1 and q_pred == 1):
        better = "Quantum" if q_conf > cnn_conf else "CNN"
        print(f"✅ Both detected cancer, but {better} model is more confident.")
        plt.title(f"Both Models: Cancer Detected\n({better} more confident)")
    elif (cnn_pred == 0 and q_pred == 0):
        better = "Quantum" if q_conf < cnn_conf else "CNN"
        print(f"✅ Both predicted normal bone. {better} model is more confident.")
        plt.title("Normal Bone (Both Models Agree)")
    else:
        print("⚠️ Models Disagree — Cross-check recommended!")
        plt.title("⚠️ Disagreement Between Models")

    plt.show()

# ===========================
# Example Usage
# ===========================
if __name__ == "__main__":
    image_path = r"C:\Users\ronit\Downloads\image\bonecancer5.jpg"  # Replace with your unseen image
    predict_image(image_path)
