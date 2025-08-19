# Traffic Flow Analysis

This project detects, tracks, and counts vehicles across lanes in traffic videos using YOLOv8 for detection and SORT for tracking. It outputs:

-> A CSV file with vehicle IDs, lane assignments, and timestamps.

-> A processed video with bounding boxes, IDs, and live lane counts.

# Project Structure

Traffic-Flow-Analysis/

│── data/ # raw videos (ignored in git)

|     └── traffic_video.mp4

│── models/ # YOLO weights (ignored in git)

│     └── yolov8n.pt
│
│── outputs/ # auto-generated results (ignored in git)

│     ├── processed_video.mp4
 
│     └── results.csv

│── src/ # source code
 
│     ├── main.py

│     ├── detector.py

│     ├── tracker.py
 
│     ├── lane_counter.py

│     ├── video_processor.py

│     ├── utils.py

│     └── sort/

│          └── sort.py

│── requirements.txt # Python dependencies

│── README.md # Project documentation

│── .gitignore

# Setup Instructions

1. Clone the repository

   git clone https://github.com/Aayushraj-07/Traffic-Flow-Analysis.git

   cd Traffic-Flow-Analysis

2. Create & activate virtual environment

   python -m venv venv

   venv\Scripts\activate # On Windows

   source venv/bin/activate # On Mac/Linux

3. Install dependencies

   pip install -r requirements.txt

# Required Files

Since large files are ignored by git, you’ll need to download them manually:

1. Traffic Video

   -> Place a sample traffic video at:

   data/traffic_video.mp4

   -> (Optional: The code can auto-download a YouTube sample if missing).

2. YOLOv8 Weights

   -> Download from: Ultralytics YOLOv8

   Place in:- models/yolov8n.pt

# Run the Project

python -m src.main - in Terminal

# Outputs

-> After running, check the outputs/ folder:

    - processed_video.mp4 → Annotated video with detections + lane counts.

    - results.csv → Tabular data:

    ex:-    VehicleID,Lane,Frame,Timestamp
                  1,2,45,0:01:30
                  2,1,60,0:02:00
