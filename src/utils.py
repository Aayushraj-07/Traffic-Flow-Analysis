"""
utils.py
--------
Helper functions for CSV logging, summaries, and utilities.
"""

import os
import csv
from datetime import timedelta


class CSVLogger:
    def __init__(self, filepath="outputs/results.csv"):
        """
        Initializes CSV logger.

        Args:
            filepath (str): Path to save CSV file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.filepath = filepath
        self.file = open(filepath, mode="w", newline="")
        self.writer = csv.writer(self.file)

        # Write header
        self.writer.writerow(["VehicleID", "Lane", "Frame", "Timestamp"])

    def log(self, vehicle_id, lane_id, frame_idx, timestamp):
        """
        Logs a single detection record into the CSV.

        Args:
            vehicle_id (int): Unique track ID
            lane_id (int): Lane number
            frame_idx (int): Frame index
            timestamp (float): Timestamp in seconds
        """
        time_str = str(timedelta(seconds=int(timestamp)))
        self.writer.writerow([vehicle_id, lane_id, frame_idx, time_str])

    def close(self):
        """Close the CSV file."""
        self.file.close()
        print(f"Results saved to {self.filepath}")


def print_summary(counts: dict):
    """
    Prints total vehicle counts per lane.

    Args:
        counts (dict): {lane_id: count}
    """
    print("\n Traffic Summary:")
    for lane_id, count in counts.items():
        print(f"  Lane {lane_id}: {count} vehicles")


if __name__ == "__main__":
    # Self-test
    logger = CSVLogger("outputs/results.csv")
    logger.log(1, 2, 45, 90.5)   # VehicleID=1, Lane=2, Frame=45, Timestamp=90.5s
    logger.log(2, 1, 60, 120.0)  # VehicleID=2, Lane=1, Frame=60, Timestamp=120s
    logger.close()

    print_summary({1: 10, 2: 15, 3: 7})
