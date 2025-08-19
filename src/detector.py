"""
detector.py
------------
Handles vehicle detection using YOLOv8 (Ultralytics).
Filters detections to vehicles only and returns structured outputs.
"""

import os
from ultralytics import YOLO

# Path to YOLO model weights
MODEL_PATH = os.path.join("models", "yolov8n.pt")

# COCO classes relevant to traffic (YOLOv8 is trained on COCO dataset)
VEHICLE_CLASSES = {
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck"
}


class VehicleDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize YOLOv8 model for vehicle detection.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. "
                "Download yolov8n.pt and place in models/ folder."
            )
        self.model = YOLO(model_path)

    def detect_vehicles(self, frame):
        """
        Run detection on a frame and return only vehicle objects.

        Args:
            frame (numpy.ndarray): Input video frame (BGR from OpenCV)

        Returns:
            List[dict]: List of detections with bounding boxes and class info
                        Format: [
                            {"bbox": [x1, y1, x2, y2],
                             "confidence": 0.92,
                             "class_id": 2,
                             "class_name": "car"}
                        ]
        """
        results = self.model(frame, verbose=False)[0]  # First (and only) result

        detections = []
        for box in results.boxes:
            class_id = int(box.cls)
            if class_id in VEHICLE_CLASSES:  # filter vehicles only
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(box.conf),
                    "class_id": class_id,
                    "class_name": VEHICLE_CLASSES[class_id]
                })

        return detections


if __name__ == "__main__":
    # Quick self-test with a sample image
    import cv2

    detector = VehicleDetector()

    # Test with a single frame from video
    video_path = "data/traffic_video.mp4"
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if ret:
        detections = detector.detect_vehicles(frame)
        print("Sample detections:", detections[:5])  # show first few
    else:
        print("Could not read frame from video.")

    cap.release()
