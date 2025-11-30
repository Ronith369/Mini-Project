from flask import Flask, render_template, request, jsonify, send_from_directory,json
import os, cv2, numpy as np, joblib, tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static', template_folder='templates')

# ------------------ Upload Folder ------------------
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------ Model Paths ------------------
CNN_PATH = r"C:/Users/ronit/BNC/cnn_model.keras"
QUANTUM_PATH = r"C:/Users/ronit/BNC/quantum_model.pkl"
PCA_PATH = r"C:/Users/ronit/BNC/pca.pkl"

cnn = None
qsvc = None
pca = None
feature_extractor = None

# ------------------ Load Models ------------------
try:
    cnn = tf.keras.models.load_model(CNN_PATH)
    qsvc = joblib.load(QUANTUM_PATH)
    pca = joblib.load(PCA_PATH)

    # Use CNN as feature extractor
    feature_extractor = tf.keras.models.Model(
        inputs=cnn.layers[0].input,
        outputs=cnn.layers[-3].output
    )

    print("✅ Models loaded successfully!")
except Exception as e:
    print("⚠ Warning: Models not loaded:", e)

# ------------------ Routes ------------------

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/compare')
def compare():
    # Placeholder metrics for comparison page
    cnn_metrics = {
        "cnn_train_acc": 0.,
        "cnn_val_acc": 0.95,
        "cnn_test_acc": 0.94,
        "cnn_report": """Precision  Recall  F1
Normal   0.95   0.96  0.95
Cancer   0.93   0.92  0.92""",
        "cnn_confusion": [[45,5],[4,46]]
    }

    quantum_metrics = {
        "q_acc": 0.91,
        "q_report": """Precision  Recall  F1
Normal   0.91   0.92  0.91
Cancer   0.90   0.89  0.89""",
        "q_confusion": [[42,8],[7,43]]
    }

    return render_template("compare.html", **cnn_metrics, **quantum_metrics)

@app.route('/report')
def report():
    with open("metrics.json", "r") as f:
        data = json.load(f)

    models = []
    for name, values in data.items():
        models.append({
            "name": name,
            "train_acc": values["train_acc"],
            "val_acc": values["val_acc"],
            "test_acc": values["test_acc"],
            "classification_report": values["classification_report"],
            "confusion_matrix": values["confusion_matrix"]
        })

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # Preprocess image
    try:
        img = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({'error': 'Failed to read image'}), 500

        img = cv2.resize(img, (128, 128))
        img_arr = img / 255.0
        img_arr = img_arr.reshape(1, 128, 128, 1)

    except Exception as e:
        return jsonify({'error': 'Failed to process image', 'details': str(e)}), 500

    # CNN prediction
    cnn_label = 'Normal'
    if cnn:
        pred_cnn = cnn.predict(img_arr)[0][0]
        cnn_label = 'Cancerous' if pred_cnn > 0.5 else 'Normal'

    # Quantum model prediction
    q_label = 'Normal'
    if qsvc and feature_extractor and pca:
        features = feature_extractor.predict(img_arr)
        features_pca = pca.transform(features)
        q_pred = qsvc.predict(features_pca)[0]
        q_label = 'Cancerous' if int(q_pred) == 1 else 'Normal'

    # Final prediction (if either predicts cancer)
    final_prediction = 'Cancerous' if cnn_label=='Cancerous' or q_label=='Cancerous' else 'Normal'

    return jsonify({
        'prediction': final_prediction,
        'cnn_label': cnn_label,
        'quantum_label': q_label,
        'image': f'/uploads/{filename}'
    })

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ------------------ Run App ------------------
if __name__ == "__main__":
    app.run(debug=True)
