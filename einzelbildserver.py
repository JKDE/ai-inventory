from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import traceback

app = Flask(__name__)
CORS(app)

model = YOLO("yolov8s-worldv2.pt")  # YOLO Modell laden

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # Standard-Grenzwert für die Erkennungsgenauigkeit
        confidence_threshold = float(data.get("confidence_threshold", 0.15))  # Standardwert: 0.25

        # Klassenfilterung, falls angegeben
        class_names = data.get("class_names", [])  # Liste der gewünschten Klassen (optional)
        class_names = [cls.lower().strip() for cls in class_names] if class_names else None

        # Dekodiere das Bild
        image_data = base64.b64decode(data["image"])
        image_np = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        # YOLOv8 Vorhersage
        #results = model(image)
        results = model(image, conf=0.15, iou=0.2)
        
        if not results or len(results[0].boxes.xyxy) == 0:
            return jsonify({"error": "No objects detected"}), 200

        boxes = []
        for box, conf, cls in zip(results[0].boxes.xyxy.cpu().numpy(),
                                  results[0].boxes.conf.cpu().numpy(),
                                  results[0].boxes.cls.cpu().numpy()):
            try:
                # **Genauigkeitsfilterung**
                if conf < confidence_threshold:
                    continue  # Überspringe Boxen unterhalb des Schwellwerts

                class_id = int(cls)
                label = model.names.get(class_id, "Unknown").lower().strip()

                # **Falls eine Klassenliste angegeben ist, nur diese berücksichtigen**
                if class_names and label not in class_names:
                    continue  

                boxes.append({
                    "x": int(box[0]),
                    "y": int(box[1]),
                    "width": int(box[2] - box[0]),
                    "height": int(box[3] - box[1]),
                    "label": label,
                    "confidence": float(conf)  # Genauigkeitswert zurückgeben
                })
            except Exception as e:
                print(f"⚠ Fehler beim Verarbeiten einer Bounding Box: {str(e)}")
                continue

        return jsonify({"boxes": boxes})

    except Exception as e:
        print("🔥 Fehler aufgetreten:", str(e))
        traceback.print_exc()
        return jsonify({"error": f"Serverfehler: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=11500, debug=False)
