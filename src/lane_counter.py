"""
lane_counter.py
---------------
Assigns vehicles to lanes and maintains per-lane counts.
"""

import cv2
import numpy as np


class LaneCounter:
    def __init__(self, lanes: dict):
        """
        Args:
            lanes (dict): Dictionary defining lanes with polygon points
                          Example:
                          {
                              1: [(100, 300), (250, 300), (250, 720), (100, 720)],
                              2: [(250, 300), (400, 300), (400, 720), (250, 720)],
                              3: [(400, 300), (550, 300), (550, 720), (400, 720)]
                          }
        """
        self.lanes = lanes
        self.counts = {lane_id: 0 for lane_id in lanes}
        self.tracked_ids = {lane_id: set() for lane_id in lanes}

    def assign_lane(self, bbox):
        """
        Determines which lane a vehicle belongs to.

        Args:
            bbox (list): [x1, y1, x2, y2] bounding box

        Returns:
            lane_id (int) or None
        """
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        for lane_id, polygon in self.lanes.items():
            if cv2.pointPolygonTest(
                np.array(polygon, np.int32), (cx, cy), False
            ) >= 0:
                return lane_id
        return None

    def update_counts(self, tracked_objects):
        """
        Update lane counts based on tracked vehicle objects.

        Args:
            tracked_objects (list): Output from tracker.py
                [
                    {"track_id": 1, "bbox": [...], "class_name": "car"},
                    {"track_id": 2, "bbox": [...], "class_name": "truck"}
                ]

        Returns:
            dict: Updated lane counts {lane_id: count}
        """
        for obj in tracked_objects:
            lane_id = self.assign_lane(obj["bbox"])
            if lane_id is not None:
                # Ensure we don't double-count the same vehicle in the same lane
                if obj["track_id"] not in self.tracked_ids[lane_id]:
                    self.tracked_ids[lane_id].add(obj["track_id"])
                    self.counts[lane_id] += 1
        return self.counts

    def draw_lanes(self, frame):
        """
        Draw lane boundaries and live counts on the video frame.

        Args:
            frame (numpy.ndarray): Input video frame (BGR)

        Returns:
            frame (numpy.ndarray): Frame with lane overlays
        """
        for lane_id, polygon in self.lanes.items():
            cv2.polylines(frame, [np.array(polygon, np.int32)], True, (0, 255, 0), 2)
            x, y = polygon[0]
            cv2.putText(frame, f"Lane {lane_id}: {self.counts[lane_id]}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame


if __name__ == "__main__":
    import numpy as np

    # Example lanes (x, y coordinates of polygons)
    lanes = {
        1: [(100, 300), (250, 300), (250, 720), (100, 720)],
        2: [(250, 300), (400, 300), (400, 720), (250, 720)],
        3: [(400, 300), (550, 300), (550, 720), (400, 720)]
    }

    counter = LaneCounter(lanes)

    # Fake tracked objects (like output from tracker.py)
    tracked_objects = [
        {"track_id": 1, "bbox": [120, 350, 200, 450], "class_name": "car"},
        {"track_id": 2, "bbox": [300, 360, 370, 460], "class_name": "truck"},
        {"track_id": 3, "bbox": [460, 370, 530, 480], "class_name": "bus"}
    ]

    counts = counter.update_counts(tracked_objects)
    print("Lane counts:", counts)
