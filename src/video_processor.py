"""
video_processor.py
------------------
Handles video reading, annotation (bboxes, IDs, lanes, counts),
live preview, and saving the processed video.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm  # ✅ progress bar


class VideoProcessor:
    def __init__(self, input_path, output_path, lane_counter, show_preview=True):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f" Input video not found: {input_path}")

        self.cap = cv2.VideoCapture(input_path)
        if not self.cap.isOpened():
            raise ValueError(f" Could not open video: {input_path}")

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # ✅ for tqdm

        # Define video writer
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        if not self.out.isOpened():
            raise RuntimeError(f" Could not open VideoWriter for {output_path}")

        self.lane_counter = lane_counter
        self.show_preview = show_preview  # ✅ toggle live preview

    def process_frame(self, frame, tracked_objects):
        # Draw bounding boxes + track IDs
        for obj in tracked_objects:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            track_id = obj["track_id"]
            class_name = obj.get("class_name", "vehicle")
            conf = float(obj.get("confidence") or 0.0)  # ✅ always numeric

            color = (0, 255, 0)  # green
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {track_id} {class_name} {conf:.2f}",
                        (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        # Update lane counts + draw lanes
        counts = self.lane_counter.update_counts(tracked_objects)
        frame = self.lane_counter.draw_lanes(frame)

        return frame, counts

    def run(self, detector, tracker, csv_writer):
        frame_idx = 0
        pbar = tqdm(total=self.total_frames, desc="Processing Video", unit="frame")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_idx += 1
                timestamp = frame_idx / self.fps  # seconds

                # Step 1: Detect vehicles
                detections = detector.detect_vehicles(frame)

                # Step 2: Track vehicles
                tracked_objects = tracker.update(detections)

                # Step 3: Annotate frame
                annotated_frame, _ = self.process_frame(frame, tracked_objects)

                # Step 4: Write frame to video
                self.out.write(annotated_frame)

                # ✅ Step 4.5: Show live preview
                if self.show_preview:
                    cv2.imshow("Traffic Analysis", annotated_frame)
                    # press 'q' to quit early
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\n User quit preview early.")
                        break

                # Step 5: Save results to CSV
                for obj in tracked_objects:
                    lane_id = self.lane_counter.assign_lane(obj["bbox"])
                    if lane_id is not None:
                        csv_writer(obj["track_id"], lane_id, frame_idx, timestamp)

                pbar.update(1)

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user. Saving progress...")

        finally:
            # Release resources even if interrupted
            self.cap.release()
            self.out.release()
            pbar.close()
            cv2.destroyAllWindows()
            print("Video processing complete. Outputs saved.")


if __name__ == "__main__":
    print("This module is meant to be used inside main.py")
