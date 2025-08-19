"""
tracker.py
-----------
Implements vehicle tracking using SORT algorithm.
Assigns unique IDs to detected vehicles and maintains tracks across frames.
"""

import numpy as np
from src.sort.sort import Sort  # using SORT implementation

class VehicleTracker:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        """
        Initialize SORT tracker.

        Args:
            max_age (int): Maximum number of missed frames before a track is deleted.
            min_hits (int): Minimum number of detections before a track is confirmed.
            iou_threshold (float): IOU threshold for matching.
        """
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

    def update(self, detections):
        """
        Update tracker with current frame detections.

        Args:
            detections (list[dict]): List of detections from detector.py
                Format: {"bbox": [x1, y1, x2, y2], "confidence": 0.92, ...}

        Returns:
            List[dict]: List of tracked objects with IDs
                        Format: [
                            {"track_id": 1,
                             "bbox": [x1, y1, x2, y2],
                             "confidence": 0.92,
                             "class_id": 2,
                             "class_name": "car"}
                        ]
        """
        if len(detections) == 0:
            dets = np.empty((0, 5))
        else:
            dets = np.array([
                det["bbox"] + [det["confidence"]] for det in detections
            ])

        tracks = self.tracker.update(dets)

        tracked_objects = []
        for i, track in enumerate(tracks):
            x1, y1, x2, y2, track_id = track
            # find matching detection (optional: to preserve class info)
            match = None
            for det in detections:
                iou = self._compute_iou(det["bbox"], [x1, y1, x2, y2])
                if iou > 0.5:  # heuristic
                    match = det
                    break

            tracked_objects.append({
                "track_id": int(track_id),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": match["confidence"] if match else None,
                "class_id": match["class_id"] if match else None,
                "class_name": match["class_name"] if match else None,
            })

        return tracked_objects

    @staticmethod
    def _compute_iou(boxA, boxB):
        """Compute Intersection over Union (IoU) between two boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea)


if __name__ == "__main__":
    # Quick self-test with fake detections
    tracker = VehicleTracker()

    detections = [
        {"bbox": [100, 150, 200, 300], "confidence": 0.9, "class_id": 2, "class_name": "car"},
        {"bbox": [400, 180, 500, 350], "confidence": 0.85, "class_id": 7, "class_name": "truck"},
    ]

    tracked = tracker.update(detections)
    print("Tracked objects:", tracked)
