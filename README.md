# AnonTrack

**Anonymous Person Tracking Desktop Application**

AnonTrack is a desktop application for anonymous person tracking in video feeds. It detects and monitors individuals' movements while strictly preserving their privacy through real-time face anonymisation — without storing or processing any personally identifiable data.

Built as a BSc thesis project at the Budapest University of Technology and Economics.

---

## Features

- **Face Anonymisation** — Blur or pixelate faces before any AI inference
- **Person Detection** — Real-time detection using YOLO11
- **Multi-Object Tracking** — Consistent anonymous IDs via DeepSORT
- **Cross-Video Re-identification** — Persistent vector embedding database with cosine similarity matching
- **Heatmaps** — Visualise crowd density across the video
- **Trajectory Maps** — Per-person movement paths with direction vectors
- **Movement Statistics** — Speed, direction, presence %, occupancy and more
- **Export** — Save heatmaps, trajectory maps, and CSV statistics

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| GUI | Qt for Python (PySide6) |
| Face Detection | RetinaFace (InsightFace) |
| Person Detection | YOLO11 (Ultralytics) |
| Tracking | DeepSORT |
| Computer Vision | OpenCV |
| Embeddings | MobileNet + pickle database |

---

## How It Works

1. **Upload** a video file (.mp4, .avi, .mov, .mkv)
2. **Select** frame interval and anonymisation method
3. **Run** detection and tracking — watch results live
4. **Explore** heatmaps, trajectories, and statistics in the Inference tab

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

Supervisor: **Dr. Mohammad Saleem**, Budapest University of Technology and Economics
