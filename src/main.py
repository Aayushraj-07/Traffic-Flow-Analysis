"""
main.py
-------
Entry point for Traffic Flow Analysis project.
"""

import os
import cv2
from src.detector import VehicleDetector
from src.tracker import VehicleTracker
from src.lane_counter import LaneCounter
from src.video_processor import VideoProcessor
from src.utils import CSVLogger, print_summary
from src.video_downloader import download_video


def define_dynamic_lanes(video_path, num_lanes=3, lane_height_ratio=0.4):
    """
    Dynamically defines lane polygons based on video resolution.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    lane_width = width // num_lanes
    start_y = int(height * lane_height_ratio)

    lanes = {}
    for i in range(num_lanes):
        x1 = i * lane_width
        x2 = (i + 1) * lane_width
        lanes[i + 1] = [
            (x1, start_y), (x2, start_y),
            (x2, height), (x1, height)
        ]
    return lanes


def main():

    # Step 1: Ensure video dataset

    video_path = "data/traffic_video.mp4"
    if not os.path.exists(video_path):
        print(" Traffic video not found, downloading...")
        download_video("https://www.youtube.com/watch?v=MNn9qKG2UFI")


    # Step 2: Define lanes dynamically

    lanes = define_dynamic_lanes(video_path, num_lanes=3, lane_height_ratio=0.4)

    # Step 3: Initialize components

    os.makedirs("outputs", exist_ok=True)  # ensure outputs folder exists

    detector = VehicleDetector()
    tracker = VehicleTracker()
    lane_counter = LaneCounter(lanes)
    csv_logger = CSVLogger("outputs/results.csv")

    processor = VideoProcessor(
        input_path=video_path,
        output_path="outputs/processed_video.mp4",
        lane_counter=lane_counter
    )


    # Step 4: Run pipeline

    processor.run(detector, tracker, csv_logger.log)

    # Step 5: Close CSV & print summary
    
    csv_logger.close()
    print_summary(lane_counter.counts)


if __name__ == "__main__":
    main()
