"""
AnonTrack

Author: Marsel Ildarovich Yuldashev - 2025
"""

import sys
import os
import shutil
import pickle
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from PySide6.QtCore import (
    Signal, Slot, Qt, QUrl, QObject, QThread, QTimer
)
from PySide6.QtGui import QPixmap, QImage, QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QStackedWidget, QLabel, QListWidget, QListWidgetItem,
    QFileDialog, QFrame, QSlider, QSizePolicy, QMessageBox, QProgressBar, QComboBox,
    QTabWidget, QGroupBox, QGridLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QTextEdit, QSplitter, QCheckBox
)
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget

class HeatmapGenerator:
    """
    Generates heatmaps showing person density over time.
    
    Uses Gaussian blur to create smooth density visualizations from
    discrete detection points. Higher intensity = more frequent presence.
    """
    
    def __init__(self, frame_shape):
        self.frame_shape = frame_shape
        # Accumulator for heat intensity at each pixel
        self.heatmap = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)
        self.person_count = 0
        self.frame_count = 0
        # Gaussian parameters control smoothness of heatmap
        self.gaussian_kernel_size = 25
        self.gaussian_sigma = 15

    def add_detection(self, bbox):
        """Add a single person detection to the heatmap."""
        x1, y1, x2, y2 = bbox
        # Use center point for heatmap location
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        
        if 0 <= y_center < self.frame_shape[0] and 0 <= x_center < self.frame_shape[1]:
            # Create single-point heatmap and blur for smooth visualization
            temp_heatmap = np.zeros(self.heatmap.shape, dtype=np.float32)
            temp_heatmap[y_center, x_center] = 1.0
            temp_heatmap = cv2.GaussianBlur(temp_heatmap,
                                          (self.gaussian_kernel_size, self.gaussian_kernel_size),
                                          self.gaussian_sigma)
            self.heatmap += temp_heatmap
            self.person_count += 1

    def add_multiple_detections(self, bboxes):
        """Add multiple detections from a single frame."""
        for bbox in bboxes:
            self.add_detection(bbox)
        self.frame_count += 1

    def get_normalized_heatmap(self):
        """Return heatmap normalized to 0-255 range for visualization."""
        if self.person_count == 0:
            return np.zeros_like(self.heatmap)
        
        normalized = self.heatmap / (self.heatmap.max() + 1e-10)  # Avoid division by zero
        # Apply JET colormap: red = hot (high density), blue = cold
        heatmap_colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return heatmap_colored

    def get_overlay_heatmap(self, background_frame):
        """Overlay heatmap on background with statistics text."""
        heatmap_colored = self.get_normalized_heatmap()
        
        # Ensure heatmap matches background dimensions
        if heatmap_colored.shape != background_frame.shape:
            heatmap_colored = cv2.resize(heatmap_colored,
                                       (background_frame.shape[1], background_frame.shape[0]))
        
        # Convert background to grayscale for better heatmap visibility
        gray_background = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
        gray_background = cv2.cvtColor(gray_background, cv2.COLOR_GRAY2BGR)
        gray_background = cv2.convertScaleAbs(gray_background, alpha=0.7, beta=0)
        
        # Blend heatmap with background
        overlay = cv2.addWeighted(gray_background, 0.4, heatmap_colored, 0.6, 0)
        
        # Add statistics overlay
        stats_text = f"Total Detections: {self.person_count} | Frames: {self.frame_count}"
        text_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x, text_y = 10, 30
        
        # Draw background for text legibility
        cv2.rectangle(overlay, (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
        cv2.putText(overlay, stats_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return overlay

    def get_statistics(self):
        """Return quantitative heatmap statistics."""
        if self.person_count == 0:
            return {
                'total_detections': 0,
                'frame_count': 0,
                'average_per_frame': 0,
                'max_intensity': 0,
                'total_intensity': 0
            }
        return {
            'total_detections': self.person_count,
            'frame_count': self.frame_count,
            'average_per_frame': self.person_count / max(1, self.frame_count),
            'max_intensity': float(self.heatmap.max()),
            'total_intensity': float(self.heatmap.sum())
        }

    def save_heatmap(self, filepath):
        """Save heatmap visualization to file."""
        heatmap_colored = self.get_normalized_heatmap()
        cv2.imwrite(filepath, heatmap_colored)

    def clear(self):
        """Reset heatmap accumulators."""
        self.heatmap = np.zeros((self.frame_shape[0], self.frame_shape[1]), dtype=np.float32)
        self.person_count = 0
        self.frame_count = 0

class TrajectoryVisualizer:
    """
    Visualizes person movement trajectories as connected paths.
    
    Draws lines connecting detection points over time, with optional
    direction arrows and color coding by person.
    """
    
    def __init__(self, frame_shape):
        self.frame_shape = frame_shape
        self.trajectory_image = None
        self.background_frame = None
        self.trajectories = {}  # person_id -> list of (x, y, frame) points

    def set_background(self, background_frame):
        """Set background image for trajectory visualization."""
        self.background_frame = background_frame.copy()
        self.trajectory_image = background_frame.copy()

    def add_trajectory(self, person_id, trajectory_points):
        """Add trajectory points for a person."""
        self.trajectories[person_id] = trajectory_points

    def visualize_trajectories(self, show_vectors=True, show_points=True, line_thickness=2,
                              point_size=4, vector_length=20, color_by_person=True):
        """
        Render trajectories with customizable visualization options.
        
        Args:
            show_vectors: Show direction arrows at trajectory ends
            show_points: Show individual trajectory points
            line_thickness: Thickness of trajectory lines
            point_size: Size of trajectory points
            vector_length: Length of direction arrows
            color_by_person: Use different colors for different persons
        """
        if self.background_frame is None:
            self.trajectory_image = np.zeros((self.frame_shape[0], self.frame_shape[1], 3), dtype=np.uint8)
        else:
            self.trajectory_image = self.background_frame.copy()
        
        if not self.trajectories:
            return self.trajectory_image
        
        # Predefined color palette for distinguishing persons
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 0),    # Dark Green
            (128, 0, 0),    # Maroon
        ]
        
        for idx, (person_id, points) in enumerate(self.trajectories.items()):
            if len(points) < 2:
                continue  # Need at least 2 points for a trajectory
            
            color = colors[idx % len(colors)] if color_by_person else (0, 255, 0)
            
            # Draw lines connecting consecutive points
            for i in range(1, len(points)):
                pt1 = (int(points[i-1]['x']), int(points[i-1]['y']))
                pt2 = (int(points[i]['x']), int(points[i]['y']))
                cv2.line(self.trajectory_image, pt1, pt2, color, line_thickness + 1)
                
                if show_points:
                    cv2.circle(self.trajectory_image, pt1, point_size, color, -1)
            
            # Draw last point
            if show_points and points:
                last_pt = (int(points[-1]['x']), int(points[-1]['y']))
                cv2.circle(self.trajectory_image, last_pt, point_size, color, -1)
            
            # Draw direction vector from last two points
            if show_vectors and len(points) >= 2:
                pt_second_last = (int(points[-2]['x']), int(points[-2]['y']))
                pt_last = (int(points[-1]['x']), int(points[-1]['y']))
                dx = pt_last[0] - pt_second_last[0]
                dy = pt_last[1] - pt_second_last[1]
                magnitude = np.sqrt(dx**2 + dy**2)
                
                if magnitude > 0:  # Avoid division by zero
                    dx_norm = dx / magnitude
                    dy_norm = dy / magnitude
                    arrow_end_x = int(pt_last[0] + dx_norm * vector_length)
                    arrow_end_y = int(pt_last[1] + dy_norm * vector_length)
                    cv2.arrowedLine(self.trajectory_image, pt_last, (arrow_end_x, arrow_end_y),
                                   color, 2, tipLength=0.4)
            
            # Label trajectory starting point
            if points:
                start_pt = (int(points[0]['x']), int(points[0]['y'] - 10))
                text_size = cv2.getTextSize(person_id, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(self.trajectory_image,
                            (start_pt[0] - 2, start_pt[1] - text_size[1] - 2),
                            (start_pt[0] + text_size[0] + 2, start_pt[1] + 2),
                            (0, 0, 0), -1)
                cv2.putText(self.trajectory_image, person_id, start_pt,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return self.trajectory_image

    def get_trajectory_statistics(self):
        """Calculate movement statistics for all trajectories."""
        stats = {
            'total_persons': len(self.trajectories),
            'person_stats': {}
        }
        
        for person_id, points in self.trajectories.items():
            if len(points) < 2:
                continue
            
            frame_count = len(points)
            total_distance = 0
            
            # Calculate total path length
            for i in range(1, len(points)):
                dx = points[i]['x'] - points[i-1]['x']
                dy = points[i]['y'] - points[i-1]['y']
                total_distance += np.sqrt(dx**2 + dy**2)
            
            # Movement metrics
            avg_speed = total_distance / (frame_count - 1) if frame_count > 1 else 0
            start_x, start_y = points[0]['x'], points[0]['y']
            end_x, end_y = points[-1]['x'], points[-1]['y']
            straight_distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            
            # Efficiency = straight line distance / actual path length
            efficiency = straight_distance / total_distance if total_distance > 0 else 0
            
            # Determine main direction of movement
            dx = end_x - start_x
            dy = end_y - start_y
            if abs(dx) > abs(dy):
                direction = "Right" if dx > 0 else "Left"
            else:
                direction = "Down" if dy > 0 else "Up"
            
            stats['person_stats'][person_id] = {
                'frame_count': frame_count,
                'total_distance': total_distance,
                'straight_distance': straight_distance,
                'avg_speed': avg_speed,
                'efficiency': efficiency,
                'direction': direction,
                'start_point': (start_x, start_y),
                'end_point': (end_x, end_y)
            }
        
        return stats

    def save_trajectory_image(self, filepath):
        """Save trajectory visualization to file."""
        if self.trajectory_image is not None:
            cv2.imwrite(filepath, self.trajectory_image)

    def clear(self):
        """Clear all stored trajectories."""
        self.trajectories = {}
        self.trajectory_image = None

class PeopleStatisticsWidget(QWidget):
    """
    Comprehensive statistics display for people detection and tracking.
    
    Organized in tabs for different aspects of analysis.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.people_stats = None
        self.layout = QVBoxLayout(self)
        
        title_label = QLabel("<h2>People Detection Statistics</h2>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(title_label)
        
        # Tabbed interface for different statistic categories
        self.tab_widget = QTabWidget()
        self.summary_tab = self._create_summary_tab()
        self.tab_widget.addTab(self.summary_tab, "Summary")
        
        self.frame_stats_tab = self._create_frame_stats_tab()
        self.tab_widget.addTab(self.frame_stats_tab, "Frame Statistics")
        
        self.person_presence_tab = self._create_person_presence_tab()
        self.tab_widget.addTab(self.person_presence_tab, "Person Presence")
        
        self.movement_tab = self._create_movement_tab()
        self.tab_widget.addTab(self.movement_tab, "Movement Analysis")
        
        self.layout.addWidget(self.tab_widget)
        
        # Export functionality
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self.export_button = QPushButton("Export Statistics to CSV")
        self.export_button.clicked.connect(self.export_statistics)
        self.export_button.setEnabled(False)
        export_layout.addWidget(self.export_button)
        export_layout.addStretch()
        self.layout.addLayout(export_layout)

    def _create_summary_tab(self):
        """Create tab with high-level summary statistics."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        grid = QGridLayout()
        
        self.summary_stats = {}
        stats_labels = [
            ("Total Frames Analyzed:", "total_frames"),
            ("Total Person Detections:", "total_detections"),
            ("Unique Persons Detected:", "unique_persons_count"),
            ("Peak People in Frame:", "max_people"),
            ("Minimum People in Frame:", "min_people"),
            ("Average People per Frame:", "avg_people"),
            ("Median People per Frame:", "median_people"),
            ("Std Dev of People per Frame:", "std_people"),
            ("Average Unique Persons per Frame:", "avg_unique_per_frame"),
            ("Frame with Most People:", "max_people_frame"),
            ("Frame with Least People:", "min_people_frame")
        ]
        
        for i, (label_text, key) in enumerate(stats_labels):
            label = QLabel(label_text)
            label.setStyleSheet("font-weight: bold;")
            value = QLabel("N/A")
            value.setStyleSheet("color: #2E7D32; font-size: 12pt;")
            self.summary_stats[key] = value
            grid.addWidget(label, i, 0)
            grid.addWidget(value, i, 1)
        
        layout.addLayout(grid)
        layout.addStretch()
        return tab

    def _create_frame_stats_tab(self):
        """Create tab with frame-by-frame statistics."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        desc = QLabel("Frame-by-frame people count statistics:")
        layout.addWidget(desc)
        
        self.frame_table = QTableWidget()
        self.frame_table.setColumnCount(3)
        self.frame_table.setHorizontalHeaderLabels(["Frame", "People Count", "Unique Persons"])
        self.frame_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.frame_table)
        
        # Distribution analysis
        distribution_group = QGroupBox("Distribution Analysis")
        distribution_layout = QVBoxLayout()
        self.distribution_text = QTextEdit()
        self.distribution_text.setReadOnly(True)
        self.distribution_text.setMaximumHeight(150)
        distribution_layout.addWidget(self.distribution_text)
        distribution_group.setLayout(distribution_layout)
        layout.addWidget(distribution_group)
        
        return tab

    def _create_person_presence_tab(self):
        """Create tab with individual person presence statistics."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        desc = QLabel("Individual person presence across frames:")
        layout.addWidget(desc)
        
        self.person_table = QTableWidget()
        self.person_table.setColumnCount(3)
        self.person_table.setHorizontalHeaderLabels(["Person ID", "Frames Present", "Presence %"])
        self.person_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.person_table)
        
        # Presence statistics
        presence_group = QGroupBox("Presence Statistics")
        presence_layout = QVBoxLayout()
        self.presence_stats_text = QTextEdit()
        self.presence_stats_text.setReadOnly(True)
        self.presence_stats_text.setMaximumHeight(100)
        presence_layout.addWidget(self.presence_stats_text)
        presence_group.setLayout(presence_layout)
        layout.addWidget(presence_group)
        
        return tab

    def _create_movement_tab(self):
        """Create tab with movement pattern analysis."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        desc = QLabel("Movement patterns and trajectories:")
        layout.addWidget(desc)
        
        movement_text = QTextEdit()
        movement_text.setReadOnly(True)
        movement_text.setHtml("""
        <h3>Movement Analysis</h3>
        <p>This section analyzes the movement patterns of detected persons:</p>
        <ul>
        <li><b>Trajectory Length:</b> Number of frames each person was tracked</li>
        <li><b>Movement Range:</b> Area covered by each person</li>
        <li><b>Average Speed:</b> Pixel movement per frame</li>
        <li><b>Direction Analysis:</b> Movement patterns and directions</li>
        </ul>
        <p>Process a video to see detailed movement statistics.</p>
        """)
        layout.addWidget(movement_text)
        
        self.movement_table = QTableWidget()
        self.movement_table.setColumnCount(5)
        self.movement_table.setHorizontalHeaderLabels(
            ["Person ID", "Trajectory Length", "Movement Range", "Avg Speed", "Main Direction"]
        )
        self.movement_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.movement_table)
        
        return tab

    def update_statistics(self, people_stats):
        """Update all tabs with new statistics data."""
        self.people_stats = people_stats
        self.export_button.setEnabled(True)
        
        self._update_summary_tab()
        self._update_frame_stats_tab()
        self._update_person_presence_tab()
        self._update_movement_tab()

    def _update_summary_tab(self):
        """Populate summary tab with calculated statistics."""
        stats = self.people_stats
        unique_persons_count = len(stats.get('unique_persons', set()))
        frame_counts = stats.get('frame_counts', [])
        unique_per_frame = stats.get('unique_persons_per_frame', [])
        
        if frame_counts:
            max_people_frame = frame_counts.index(stats['max_people']) if stats['max_people'] in frame_counts else "N/A"
            min_people_frame = frame_counts.index(stats['min_people']) if stats['min_people'] in frame_counts else "N/A"
            avg_unique_per_frame = np.mean(unique_per_frame) if unique_per_frame else 0
        else:
            max_people_frame = "N/A"
            min_people_frame = "N/A"
            avg_unique_per_frame = 0
        
        # Update all summary labels
        self.summary_stats['total_frames'].setText(f"{stats.get('total_frames', 0):,}")
        self.summary_stats['total_detections'].setText(f"{stats.get('total_detections', 0):,}")
        self.summary_stats['unique_persons_count'].setText(f"{unique_persons_count:,}")
        self.summary_stats['max_people'].setText(f"{stats.get('max_people', 0):,}")
        self.summary_stats['min_people'].setText(f"{stats.get('min_people', 0):,}")
        self.summary_stats['avg_people'].setText(f"{stats.get('avg_people', 0):.2f}")
        self.summary_stats['median_people'].setText(f"{stats.get('median_people', 0):.2f}")
        self.summary_stats['std_people'].setText(f"{stats.get('std_people', 0):.2f}")
        self.summary_stats['avg_unique_per_frame'].setText(f"{avg_unique_per_frame:.2f}")
        self.summary_stats['max_people_frame'].setText(f"{max_people_frame}")
        self.summary_stats['min_people_frame'].setText(f"{min_people_frame}")

    def _update_frame_stats_tab(self):
        """Populate frame statistics table."""
        stats = self.people_stats
        frame_counts = stats.get('frame_counts', [])
        unique_per_frame = stats.get('unique_persons_per_frame', [])
        
        self.frame_table.setRowCount(len(frame_counts))
        for i, (count, unique_count) in enumerate(zip(frame_counts, unique_per_frame)):
            self.frame_table.setItem(i, 0, QTableWidgetItem(f"Frame {i+1}"))
            self.frame_table.setItem(i, 1, QTableWidgetItem(str(count)))
            self.frame_table.setItem(i, 2, QTableWidgetItem(str(unique_count)))
        
        # Calculate distribution
        if frame_counts:
            counter = Counter(frame_counts)
            total_frames = len(frame_counts)
            dist_text = "<b>People Count Distribution:</b><br>"
            for count in sorted(counter.keys()):
                percentage = (counter[count] / total_frames) * 100
                dist_text += f"{count} person(s): {counter[count]} frames ({percentage:.1f}%)<br>"
            self.distribution_text.setHtml(dist_text)

    def _update_person_presence_tab(self):
        """Populate person presence statistics."""
        stats = self.people_stats
        person_presence = stats.get('person_presence', {})
        total_frames = stats.get('total_frames', 1)
        
        # Sort by presence (most to least)
        sorted_persons = sorted(person_presence.items(), key=lambda x: x[1], reverse=True)
        self.person_table.setRowCount(len(sorted_persons))
        
        for i, (person_id, frames_present) in enumerate(sorted_persons):
            presence_percentage = (frames_present / total_frames) * 100
            self.person_table.setItem(i, 0, QTableWidgetItem(person_id))
            self.person_table.setItem(i, 1, QTableWidgetItem(str(frames_present)))
            self.person_table.setItem(i, 2, QTableWidgetItem(f"{presence_percentage:.1f}%"))
        
        # Calculate aggregate presence statistics
        if person_presence:
            avg_presence = np.mean(list(person_presence.values()))
            max_presence = max(person_presence.values())
            min_presence = min(person_presence.values())
            presence_text = f"""
            <b>Presence Statistics:</b><br>
            • Average frames per person: {avg_presence:.1f}<br>
            • Maximum presence: {max_presence} frames<br>
            • Minimum presence: {min_presence} frames<br>
            • Most frequent person: {sorted_persons[0][0]} ({sorted_persons[0][1]} frames)
            """
            self.presence_stats_text.setHtml(presence_text)

    def _update_movement_tab(self):
        """Populate movement analysis table."""
        stats = self.people_stats
        trajectories = stats.get('person_trajectories', {})
        self.movement_table.setRowCount(len(trajectories))
        
        for i, (person_id, trajectory) in enumerate(trajectories.items()):
            if len(trajectory) > 1:
                # Calculate movement metrics
                trajectory_length = len(trajectory)
                
                # Movement range (bounding box of trajectory)
                xs = [point['x'] for point in trajectory]
                ys = [point['y'] for point in trajectory]
                x_range = max(xs) - min(xs)
                y_range = max(ys) - min(ys)
                movement_range = f"{x_range:.0f}×{y_range:.0f}"
                
                # Average speed (pixels per frame)
                total_distance = 0
                for j in range(1, len(trajectory)):
                    dx = trajectory[j]['x'] - trajectory[j-1]['x']
                    dy = trajectory[j]['y'] - trajectory[j-1]['y']
                    total_distance += np.sqrt(dx**2 + dy**2)
                avg_speed = total_distance / (len(trajectory) - 1) if len(trajectory) > 1 else 0
                
                # Determine main direction
                if len(trajectory) >= 2:
                    start_x, start_y = trajectory[0]['x'], trajectory[0]['y']
                    end_x, end_y = trajectory[-1]['x'], trajectory[-1]['y']
                    dx = end_x - start_x
                    dy = end_y - start_y
                    
                    if abs(dx) > abs(dy):
                        direction = "Horizontal" + (" right" if dx > 0 else " left")
                    else:
                        direction = "Vertical" + (" down" if dy > 0 else " up")
                else:
                    direction = "Static"
            else:
                # Single-point trajectory
                trajectory_length = len(trajectory)
                movement_range = "0×0"
                avg_speed = 0
                direction = "Static"
            
            # Populate table row
            self.movement_table.setItem(i, 0, QTableWidgetItem(person_id))
            self.movement_table.setItem(i, 1, QTableWidgetItem(str(trajectory_length)))
            self.movement_table.setItem(i, 2, QTableWidgetItem(movement_range))
            self.movement_table.setItem(i, 3, QTableWidgetItem(f"{avg_speed:.1f} px/frame"))
            self.movement_table.setItem(i, 4, QTableWidgetItem(direction))

    @Slot()
    def export_statistics(self):
        """Export all statistics to CSV file."""
        if not self.people_stats:
            QMessageBox.warning(self, "No Data", "No statistics data to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Statistics", "people_statistics.csv",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            try:
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Write header
                    writer.writerow(["People Statistics", "Value"])
                    writer.writerow([])
                    
                    # Summary statistics
                    writer.writerow(["SUMMARY STATISTICS"])
                    stats = self.people_stats
                    unique_persons_count = len(stats.get('unique_persons', set()))
                    
                    summary_data = [
                        ["Total Frames Analyzed", stats.get('total_frames', 0)],
                        ["Total Person Detections", stats.get('total_detections', 0)],
                        ["Unique Persons Detected", unique_persons_count],
                        ["Peak People in Frame", stats.get('max_people', 0)],
                        ["Minimum People in Frame", stats.get('min_people', 0)],
                        ["Average People per Frame", f"{stats.get('avg_people', 0):.2f}"],
                        ["Median People per Frame", f"{stats.get('median_people', 0):.2f}"],
                        ["Std Dev of People per Frame", f"{stats.get('std_people', 0):.2f}"]
                    ]
                    
                    for row in summary_data:
                        writer.writerow(row)
                    
                    writer.writerow([])
                    
                    # Frame-by-frame statistics
                    writer.writerow(["FRAME-BY-FRAME STATISTICS"])
                    writer.writerow(["Frame", "People Count", "Unique Persons"])
                    frame_counts = stats.get('frame_counts', [])
                    unique_per_frame = stats.get('unique_persons_per_frame', [])
                    
                    for i, (count, unique_count) in enumerate(zip(frame_counts, unique_per_frame)):
                        writer.writerow([f"Frame {i+1}", count, unique_count])
                    
                    writer.writerow([])
                    
                    # Person presence statistics
                    writer.writerow(["PERSON PRESENCE STATISTICS"])
                    writer.writerow(["Person ID", "Frames Present", "Presence %"])
                    person_presence = stats.get('person_presence', {})
                    total_frames = stats.get('total_frames', 1)
                    
                    for person_id, frames_present in sorted(person_presence.items(), key=lambda x: x[1], reverse=True):
                        presence_percentage = (frames_present / total_frames) * 100
                        writer.writerow([person_id, frames_present, f"{presence_percentage:.1f}%"])
                
                QMessageBox.information(self, "Export Successful", f"Statistics exported to:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export statistics:\n{str(e)}")

class TrajectoryVisualizationWidget(QWidget):
    """
    Interactive widget for visualizing person movement trajectories.
    
    Allows customization of visualization parameters and background.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.trajectory_visualizer = None
        self.current_trajectory_image = None
        self.current_video_name = None
        self.people_stats = None
        self.background_frame = None
        
        self.layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("<h2>Person Trajectory Visualization</h2>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(
            "<p>Visualize person movement trajectories as vectors on video frames.</p>" \
            "<br><br><br><br><br><br>"
        )
        desc_label.setWordWrap(True)
        self.layout.addWidget(desc_label)
        
        # Split layout: controls on left, visualization on right
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Visualization controls
        controls_group = QGroupBox("Visualization Controls")
        controls_layout = QVBoxLayout()
        
        # Background selection
        bg_layout = QHBoxLayout()
        bg_label = QLabel("Background:")
        self.bg_combo = QComboBox()
        self.bg_combo.addItems(["Original Frame", "Black Background", "White Background"])
        self.bg_combo.setCurrentIndex(0)
        bg_layout.addWidget(bg_label)
        bg_layout.addWidget(self.bg_combo)
        controls_layout.addLayout(bg_layout)
        
        # Visualization options
        self.show_vectors_check = QCheckBox("Show Direction Arrows")
        self.show_vectors_check.setChecked(True)
        controls_layout.addWidget(self.show_vectors_check)
        
        self.show_points_check = QCheckBox("Show Trajectory Points")
        self.show_points_check.setChecked(True)
        controls_layout.addWidget(self.show_points_check)
        
        self.color_by_person_check = QCheckBox("Different Colors for Different Persons")
        self.color_by_person_check.setChecked(True)
        controls_layout.addWidget(self.color_by_person_check)
        
        # Line thickness control
        thickness_layout = QHBoxLayout()
        thickness_label = QLabel("Line Thickness:")
        self.thickness_slider = QSlider(Qt.Orientation.Horizontal)
        self.thickness_slider.setRange(1, 5)
        self.thickness_slider.setValue(3)
        self.thickness_value = QLabel("3")
        thickness_layout.addWidget(thickness_label)
        thickness_layout.addWidget(self.thickness_slider)
        thickness_layout.addWidget(self.thickness_value)
        controls_layout.addLayout(thickness_layout)
        
        # Arrow length control
        vector_layout = QHBoxLayout()
        vector_label = QLabel("Arrow Length:")
        self.vector_slider = QSlider(Qt.Orientation.Horizontal)
        self.vector_slider.setRange(10, 50)
        self.vector_slider.setValue(25)
        self.vector_value = QLabel("25")
        vector_layout.addWidget(vector_label)
        vector_layout.addWidget(self.vector_slider)
        vector_layout.addWidget(self.vector_value)
        controls_layout.addLayout(vector_layout)
        
        controls_group.setLayout(controls_layout)
        left_layout.addWidget(controls_group)
        
        # Statistics display
        stats_group = QGroupBox("Trajectory Statistics")
        stats_layout = QVBoxLayout()
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)
        left_layout.addWidget(stats_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        self.generate_button = QPushButton("Generate Trajectories")
        self.generate_button.setToolTip("Generate trajectory visualization")
        self.generate_button.clicked.connect(self.generate_trajectories)
        self.generate_button.setEnabled(False)
        
        self.save_button = QPushButton("Save Image")
        self.save_button.setToolTip("Save trajectory visualization as image")
        self.save_button.clicked.connect(self.save_trajectory_image)
        self.save_button.setEnabled(False)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Clear trajectory visualization")
        self.clear_button.clicked.connect(self.clear_trajectories)
        
        button_layout.addWidget(self.generate_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.clear_button)
        left_layout.addLayout(button_layout)
        left_layout.addStretch()
        
        # Right panel: visualization display
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.trajectory_label = QLabel(
            "No trajectory data available.\nProcess a video to see trajectory visualization."
        )
        self.trajectory_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trajectory_label.setMinimumHeight(400)
        self.trajectory_label.setStyleSheet("border: 2px solid #cccccc; background-color: #f0f0f0;")
        right_layout.addWidget(self.trajectory_label)
        
        # Add both panels to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 500])
        self.layout.addWidget(splitter)
        
        # Connect control signals
        self.thickness_slider.valueChanged.connect(lambda v: self.thickness_value.setText(str(v)))
        self.vector_slider.valueChanged.connect(lambda v: self.vector_value.setText(str(v)))
        self.show_vectors_check.stateChanged.connect(self.on_visualization_changed)
        self.show_points_check.stateChanged.connect(self.on_visualization_changed)
        self.color_by_person_check.stateChanged.connect(self.on_visualization_changed)
        self.bg_combo.currentIndexChanged.connect(self.on_visualization_changed)

    def set_video_name(self, video_name):
        """Update UI for current video."""
        self.current_video_name = video_name
        if video_name:
            self.generate_button.setEnabled(True)
            self.generate_button.setText(f"Generate Trajectories for '{video_name}'")
        else:
            self.generate_button.setEnabled(False)
            self.generate_button.setText("Generate Trajectories")

    def update_statistics(self, people_stats):
        """Receive and store people statistics for trajectory generation."""
        self.people_stats = people_stats
        if self.current_video_name:
            # Load first frame as background
            frames_dir = f"{self.current_video_name}_frames"
            if os.path.exists(frames_dir):
                frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
                if frame_files:
                    first_frame_path = os.path.join(frames_dir, frame_files[0])
                    self.background_frame = cv2.imread(first_frame_path)

    def generate_trajectories(self):
        """Generate trajectory visualization from stored data."""
        if not self.people_stats or 'person_trajectories' not in self.people_stats:
            QMessageBox.warning(self, "No Data", "No trajectory data available. Please process a video first.")
            return
        
        trajectories = self.people_stats.get('person_trajectories', {})
        if not trajectories:
            QMessageBox.warning(self, "No Trajectories", "No person trajectories found in the processed video.")
            return
        
        # Determine visualization dimensions
        if self.background_frame is not None:
            frame_height, frame_width = self.background_frame.shape[:2]
        else:
            # Estimate dimensions from trajectory points
            first_trajectory = next(iter(trajectories.values()))
            if first_trajectory:
                max_x = max(point['x'] for point in first_trajectory)
                max_y = max(point['y'] for point in first_trajectory)
                frame_width = int(max_x * 1.2)  # Add 20% margin
                frame_height = int(max_y * 1.2)
            else:
                frame_width, frame_height = 800, 600  # Default
        
        # Initialize visualizer
        self.trajectory_visualizer = TrajectoryVisualizer((frame_height, frame_width, 3))
        
        # Set background based on selection
        bg_choice = self.bg_combo.currentText()
        if bg_choice == "Original Frame" and self.background_frame is not None:
            self.trajectory_visualizer.set_background(self.background_frame)
        elif bg_choice == "White Background":
            white_bg = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
            self.trajectory_visualizer.set_background(white_bg)
        else:  # Black background
            black_bg = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            self.trajectory_visualizer.set_background(black_bg)
        
        # Add all trajectories
        for person_id, points in trajectories.items():
            self.trajectory_visualizer.add_trajectory(person_id, points)
        
        # Render visualization with current settings
        show_vectors = self.show_vectors_check.isChecked()
        show_points = self.show_points_check.isChecked()
        color_by_person = self.color_by_person_check.isChecked()
        line_thickness = self.thickness_slider.value()
        vector_length = self.vector_slider.value()
        
        trajectory_image = self.trajectory_visualizer.visualize_trajectories(
            show_vectors=show_vectors,
            show_points=show_points,
            color_by_person=color_by_person,
            line_thickness=line_thickness,
            vector_length=vector_length
        )
        
        # Display the visualization
        height, width, channel = trajectory_image.shape
        bytes_per_line = 3 * width
        qimage = QImage(trajectory_image.data, width, height, bytes_per_line,
                       QImage.Format.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimage)
        
        self.trajectory_label.setPixmap(
            pixmap.scaled(self.trajectory_label.size(),
                         Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
        )
        
        # Update statistics display
        stats = self.trajectory_visualizer.get_trajectory_statistics()
        self.update_statistics_display(stats)
        
        # Enable save button
        self.save_button.setEnabled(True)
        self.current_trajectory_image = trajectory_image

    def on_visualization_changed(self):
        """Regenerate visualization when settings change."""
        if self.trajectory_visualizer is not None:
            self.generate_trajectories()

    def update_statistics_display(self, stats):
        """Display trajectory statistics in text box."""
        stats_text = f"<b>Trajectory Statistics:</b><br>"
        stats_text += f"• Total Persons with Trajectories: <b>{stats['total_persons']}</b><br>"
        
        if stats['person_stats']:
            stats_text += f"<br><b>Individual Statistics:</b><br>"
            for person_id, person_stats in stats['person_stats'].items():
                stats_text += f"<br><b>{person_id}:</b><br>"
                stats_text += f"  • Frames Tracked: {person_stats['frame_count']}<br>"
                stats_text += f"  • Total Distance: {person_stats['total_distance']:.1f} pixels<br>"
                stats_text += f"  • Straight Distance: {person_stats['straight_distance']:.1f} pixels<br>"
                stats_text += f"  • Movement Efficiency: {person_stats['efficiency']:.2%}<br>"
                stats_text += f"  • Average Speed: {person_stats['avg_speed']:.1f} px/frame<br>"
                stats_text += f"  • Main Direction: {person_stats['direction']}<br>"
        
        self.stats_text.setHtml(stats_text)

    @Slot()
    def save_trajectory_image(self):
        """Save current trajectory visualization to file."""
        if self.current_trajectory_image is None:
            QMessageBox.warning(self, "No Image", "No trajectory image to save.")
            return
        
        default_name = f"{self.current_video_name}_trajectories.png" if self.current_video_name else "trajectories.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Trajectory Visualization", default_name,
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*.*)"
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.current_trajectory_image)
                QMessageBox.information(self, "Success", f"Trajectory visualization saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image:\n{str(e)}")

    @Slot()
    def clear_trajectories(self):
        """Clear current trajectory visualization."""
        reply = QMessageBox.question(
            self,
            "Clear Trajectories",
            "Are you sure you want to clear the trajectory visualization?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.trajectory_visualizer = None
            self.current_trajectory_image = None
            self.trajectory_label.setText(
                "Trajectory visualization cleared.\nProcess a video to generate new visualization."
            )
            self.stats_text.clear()
            self.save_button.setEnabled(False)

class PersonEmbeddingDB:
    """
    Manages persistent storage of person embeddings for cross-video re-identification.
    
    Uses cosine similarity to match new detections against known persons.
    Embeddings are stored as high-dimensional vectors representing unique individuals.
    
    Attributes:
        db_path: Path to the pickle file storing embeddings
        similarity_threshold: Minimum cosine similarity (0-1) to consider a match
        embeddings: List of stored embedding vectors
        person_ids: Parallel list of person IDs corresponding to embeddings
        next_person_id: Counter for generating new person IDs
    """
    
    def __init__(self, db_path="person_embeddings.pkl", similarity_threshold=0.73):
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.embeddings = []
        self.person_ids = []
        self.next_person_id = 1
        self.load()

    def load(self):
        """Load embeddings from disk, recovering from any corruption."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data.get('embeddings', [])
                    self.person_ids = data.get('person_ids', [])
                    # Find the highest existing ID to avoid conflicts
                    if self.person_ids:
                        max_id = max([int(pid.split('_')[1]) for pid in self.person_ids if '_' in pid])
                        self.next_person_id = max_id + 1
            except Exception as e:
                # Corrupted file - start fresh
                self.embeddings = []
                self.person_ids = []

    def save(self):
        """Save embeddings to disk with error handling."""
        try:
            data = {
                'embeddings': self.embeddings,
                'person_ids': self.person_ids
            }
            with open(self.db_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            # Silently fail - better to lose recent data than crash
            pass

    def find_match(self, new_embedding):
        """
        Find the most similar person in the database.
        
        Args:
            new_embedding: Feature vector from a new detection
            
        Returns:
            tuple: (matched_person_id, similarity_score) or (None, similarity)
                   if no match above threshold
        """
        if not self.embeddings:
            return None, 0.0
        
        embeddings_array = np.array(self.embeddings)
        new_embedding_array = np.array(new_embedding).reshape(1, -1)
        
        # Use cosine similarity for orientation-independent comparison
        similarities = cosine_similarity(new_embedding_array, embeddings_array)[0]
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= self.similarity_threshold:
            return self.person_ids[best_idx], best_similarity
        else:
            return None, best_similarity

    def add_person(self, embedding):
        """Add a new person to the database with auto-incremented ID."""
        person_id = f"Person_{self.next_person_id:03d}"
        self.embeddings.append(embedding)
        self.person_ids.append(person_id)
        self.next_person_id += 1
        self.save()
        return person_id

    def clear(self):
        """Clear all stored data and delete the database file."""
        self.embeddings = []
        self.person_ids = []
        self.next_person_id = 1
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def get_person_count(self):
        """Return count of unique persons in database."""
        return len(set(self.person_ids))

    def get_all_persons(self):
        """Return list of all unique person IDs."""
        return list(set(self.person_ids))

class FaceAnonymizer:
    """
    Anonymizes faces in frames using blurring or pixelation.
    
    Uses InsightFace's RetinaFace for face detection to maintain privacy.
    """
    
    def __init__(self, anonymization_method='blur', det_threshold=0.5, det_size=(640, 640)):
        self.anonymization_method = anonymization_method.lower()
        self.det_threshold = det_threshold
        self.det_size = det_size
        self.model = None
        self._model_loaded = False

    def load_model(self):
        """Lazy-load the face detection model to reduce startup time."""
        if self._model_loaded:
            return
        
        try:
            from insightface.app import FaceAnalysis
            self.model = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.model.prepare(ctx_id=0, det_size=self.det_size)
            self._model_loaded = True
        except ImportError:
            print("Error: insightface not installed. Please run: pip install insightface")
            raise
        except Exception as e:
            print(f"Error loading RetinaFace model: {e}")
            raise

    def detect_faces(self, frame):
        """Detect faces in frame with confidence above threshold."""
        if not self._model_loaded:
            self.load_model()
        
        try:
            faces = self.model.get(frame)
            detected_faces = []
            for face in faces:
                if face.det_score >= self.det_threshold:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    detected_faces.append((x1, y1, x2, y2))
            return detected_faces
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []

    def blur_faces(self, frame, faces):
        """Apply Gaussian blur to detected face regions."""
        anonymized_frame = frame.copy()
        for (x1, y1, x2, y2) in faces:
            # Expand bounding box slightly to ensure full face coverage
            x1 = max(0, x1 - 10)
            y1 = max(0, y1 - 10)
            x2 = min(frame.shape[1], x2 + 10)
            y2 = min(frame.shape[0], y2 + 10)
            
            face_roi = anonymized_frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                blurred_face = cv2.GaussianBlur(face_roi, (23, 23), 30)
                anonymized_frame[y1:y2, x1:x2] = blurred_face
        return anonymized_frame

    def pixelate_faces(self, frame, faces, pixel_size=15):
        """Apply pixelation effect to detected face regions."""
        anonymized_frame = frame.copy()
        for (x1, y1, x2, y2) in faces:
            # Expand bounding box slightly
            x1 = max(0, x1 - 10)
            y1 = max(0, y1 - 10)
            x2 = min(frame.shape[1], x2 + 10)
            y2 = min(frame.shape[0], y2 + 10)
            
            face_roi = anonymized_frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue
            
            h, w = face_roi.shape[:2]
            if h < pixel_size or w < pixel_size:
                # Fall back to blur if face is too small for pixelation
                pixelated = cv2.GaussianBlur(face_roi, (15, 15), 15)
            else:
                # Downsample then upsample for pixelation effect
                small = cv2.resize(face_roi, (w // pixel_size, h // pixel_size),
                                  interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            anonymized_frame[y1:y2, x1:x2] = pixelated
        return anonymized_frame

    def anonymize_frame(self, frame):
        """
        Apply selected anonymization method to all faces in frame.
        
        Returns:
            Frame with faces anonymized, or original frame if no faces detected
        """
        faces = self.detect_faces(frame)
        if len(faces) == 0:
            return frame.copy()
        
        if self.anonymization_method == 'blur':
            return self.blur_faces(frame, faces)
        elif self.anonymization_method == 'pixelate':
            return self.pixelate_faces(frame, faces)
        else:
            return self.blur_faces(frame, faces)

class InferencePage(QWidget):
    """
    Analytics and visualization page with multiple tabs.
    
    Provides database statistics, people analytics, heatmaps, and trajectory visualization.
    """
    
    def __init__(self, embedding_db, parent=None):
        super().__init__(parent)
        self.embedding_db = embedding_db
        self.heatmap_generator = None
        self.current_heatmap_image = None
        self.current_video_name = None
        
        # Create sub-widgets
        self.people_stats_widget = PeopleStatisticsWidget()
        self.trajectory_widget = TrajectoryVisualizationWidget()
        
        # Tabbed interface
        self.tab_widget = QTabWidget()
        self.database_tab = self._create_database_tab()
        self.people_stats_tab = self._create_people_stats_tab()
        self.heatmap_tab = self._create_heatmap_tab()
        self.trajectory_tab = self._create_trajectory_tab()
        
        self.tab_widget.addTab(self.database_tab, "Database Statistics")
        self.tab_widget.addTab(self.people_stats_tab, "People Statistics")
        self.tab_widget.addTab(self.heatmap_tab, "Heatmap Analytics")
        self.tab_widget.addTab(self.trajectory_tab, "Trajectory Visualization")
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tab_widget)
        
        # Initial statistics update
        self.update_statistics()

    def _create_database_tab(self):
        """Create database statistics tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.title_label = QLabel("Database Statistics")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.title_label.font()
        font.setPointSize(16)
        font.setBold(True)
        self.title_label.setFont(font)
        layout.addWidget(self.title_label)
        
        # Statistics frame
        stats_frame = QFrame()
        stats_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
        stats_layout = QVBoxLayout(stats_frame)
        
        stats_title = QLabel("<b>Cross-Video Person Database</b>")
        stats_layout.addWidget(stats_title)
        
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_frame)
        
        # Explanation section
        analysis_group = QGroupBox("Cross-Video Analysis")
        analysis_layout = QVBoxLayout()
        analysis_desc = QLabel(
            "<p>This database stores vector embeddings of detected persons across multiple videos.</p>"
            "<p><b>How it works:</b></p>"
            "<ul>"
            "<li>Each person is represented by a unique vector embedding</li>"
            "<li>85% similarity threshold for person re-identification</li>"
            "<li>Persistent anonymous IDs (Person_001, Person_002, etc.)</li>"
            "<li>Enables tracking across different videos and sessions</li>"
            "</ul>"
        )
        analysis_desc.setWordWrap(True)
        analysis_layout.addWidget(analysis_desc)
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        layout.addStretch()
        return tab

    def _create_people_stats_tab(self):
        """Create people statistics tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        title_label = QLabel("People Detection Statistics")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = title_label.font()
        font.setPointSize(16)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)
        
        desc_label = QLabel(
            "<p>Comprehensive statistics about people detection and tracking in the processed video.</p>"
            "<p>Process a video in the Detection & Tracking tab to see detailed statistics.</p>"
        )
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        layout.addWidget(self.people_stats_widget)
        layout.addStretch()
        return tab

    def _create_heatmap_tab(self):
        """Create heatmap visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        heatmap_title = QLabel("Person Density Heatmap")
        heatmap_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = heatmap_title.font()
        font.setPointSize(16)
        font.setBold(True)
        heatmap_title.setFont(font)
        layout.addWidget(heatmap_title)
        
        desc_label = QLabel(
            "<p>Heatmap visualization shows areas where people appear most frequently in the video.</p>"
            "<p><b>Color Legend:</b> Hot, Red (high density) → Medium, Yellow → Cold, Blue (low density)</p>"
        )
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)
        
        # Heatmap visualization group
        self.heatmap_group = QGroupBox("Heatmap Visualization")
        heatmap_group_layout = QVBoxLayout()
        
        self.heatmap_label = QLabel(
            "No heatmap data available.\nProcess a video in the Detection & Tracking tab first."
        )
        self.heatmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.heatmap_label.setMinimumHeight(400)
        self.heatmap_label.setStyleSheet("border: 2px solid #cccccc; background-color: #f0f0f0;")
        heatmap_group_layout.addWidget(self.heatmap_label)
        
        # Control buttons
        controls_layout = QHBoxLayout()
        self.generate_heatmap_btn = QPushButton("Generate Heatmap")
        self.generate_heatmap_btn.setToolTip("Generate heatmap from processed video data")
        self.generate_heatmap_btn.clicked.connect(self.generate_heatmap)
        self.generate_heatmap_btn.setEnabled(False)
        
        self.save_heatmap_btn = QPushButton("Save Heatmap")
        self.save_heatmap_btn.setToolTip("Save heatmap as image file")
        self.save_heatmap_btn.clicked.connect(self.save_heatmap)
        self.save_heatmap_btn.setEnabled(False)
        
        self.clear_heatmap_btn = QPushButton("Clear Heatmap")
        self.clear_heatmap_btn.setToolTip("Clear current heatmap data")
        self.clear_heatmap_btn.clicked.connect(self.clear_heatmap)
        
        controls_layout.addWidget(self.generate_heatmap_btn)
        controls_layout.addWidget(self.save_heatmap_btn)
        controls_layout.addWidget(self.clear_heatmap_btn)
        heatmap_group_layout.addLayout(controls_layout)
        
        # Statistics display
        self.heatmap_stats_label = QLabel("")
        self.heatmap_stats_label.setWordWrap(True)
        heatmap_group_layout.addWidget(self.heatmap_stats_label)
        
        self.heatmap_group.setLayout(heatmap_group_layout)
        layout.addWidget(self.heatmap_group)
        layout.addStretch()
        return tab

    def _create_trajectory_tab(self):
        """Create trajectory visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(self.trajectory_widget)
        layout.addStretch()
        return tab

    def update_statistics(self):
        """Update database statistics display."""
        person_count = self.embedding_db.get_person_count()
        all_persons = self.embedding_db.get_all_persons()
        
        stats_text = f"""
        <p><b>Total Unique Persons:</b> <span style='color: #2E7D32; font-size: 14pt;'>{person_count}</span></p>
        <p><b>Current Person IDs:</b> {', '.join(sorted(all_persons)) if all_persons else 'None'}</p>
        <p><b>Similarity Threshold:</b> {self.embedding_db.similarity_threshold * 100}%</p>
        <p><b>Database Status:</b> {'Loaded' if os.path.exists(self.embedding_db.db_path) else 'Empty'}</p>
        <p><b>Cross-Video Tracking:</b> {'Enabled' if person_count > 0 else 'Waiting for first video'}</p>
        """
        self.stats_label.setText(stats_text)

    def set_video_name(self, video_name):
        """Update UI for current video."""
        self.current_video_name = video_name
        if video_name:
            self.generate_heatmap_btn.setEnabled(True)
            self.generate_heatmap_btn.setText(f"Generate Heatmap for '{video_name}'")
            self.trajectory_widget.set_video_name(video_name)
        else:
            self.generate_heatmap_btn.setEnabled(False)
            self.generate_heatmap_btn.setText("Generate Heatmap")
            self.trajectory_widget.set_video_name(None)

    @Slot(list)
    def on_heatmap_data_received(self, heatmap_data):
        """Receive and process heatmap data from tracker."""
        if not heatmap_data:
            return
        
        data = heatmap_data[0]
        detections = data['detections']
        frame_width = data['frame_width']
        frame_height = data['frame_height']
        
        # Update statistics widgets
        if 'people_stats' in data:
            self.people_stats_widget.update_statistics(data['people_stats'])
            self.trajectory_widget.update_statistics(data['people_stats'])
        
        # Generate heatmap from accumulated detections
        self.heatmap_generator = HeatmapGenerator((frame_height, frame_width, 3))
        for frame_detections in detections:
            self.heatmap_generator.add_multiple_detections(frame_detections)
        
        self.display_heatmap()

    @Slot(dict)
    def on_people_statistics_received(self, people_stats):
        """Receive people statistics from tracker."""
        self.people_stats_widget.update_statistics(people_stats)
        self.trajectory_widget.update_statistics(people_stats)

    @Slot()
    def generate_heatmap(self):
        """Generate heatmap from processed video data."""
        if not self.current_video_name:
            QMessageBox.warning(self, "No Video", "Please process a video first.")
            return
        
        processed_dir = f"{self.current_video_name}_processed"
        if not os.path.exists(processed_dir):
            QMessageBox.warning(self, "No Data",
                               f"No processed frames found for '{self.current_video_name}'.\n"
                               f"Please run detection and tracking first.")
            return
        
        # Confirm regeneration if heatmap already exists
        if self.heatmap_generator is not None:
            reply = QMessageBox.question(
                self,
                "Regenerate Heatmap",
                "Heatmap data already exists. Regenerate from processed frames?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        QMessageBox.information(self, "Heatmap Generation",
                               "Heatmap data should be automatically generated during tracking.\n"
                               "If no heatmap is shown, please run detection and tracking first.")

    def display_heatmap(self):
        """Display current heatmap visualization."""
        if self.heatmap_generator is None:
            self.heatmap_label.setText("No heatmap data available.")
            self.heatmap_stats_label.setText("")
            self.save_heatmap_btn.setEnabled(False)
            return
        
        # Get background frame for overlay
        background = self.get_background_frame()
        if background is not None:
            # Resize if dimensions don't match
            if background.shape[:2] != (self.heatmap_generator.frame_shape[0], 
                                       self.heatmap_generator.frame_shape[1]):
                background = cv2.resize(background, (self.heatmap_generator.frame_shape[1],
                                                    self.heatmap_generator.frame_shape[0]))
        else:
            # Use black background if no frame available
            background = np.zeros((self.heatmap_generator.frame_shape[0],
                                  self.heatmap_generator.frame_shape[1], 3), dtype=np.uint8)
        
        # Generate overlay visualization
        heatmap_overlay = self.heatmap_generator.get_overlay_heatmap(background)
        
        # Convert to QImage for display
        height, width, channel = heatmap_overlay.shape
        bytes_per_line = 3 * width
        qimage = QImage(heatmap_overlay.data, width, height, bytes_per_line,
                       QImage.Format.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qimage)
        
        self.heatmap_label.setPixmap(
            pixmap.scaled(self.heatmap_label.size(),
                         Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
        )
        
        # Display statistics
        stats = self.heatmap_generator.get_statistics()
        stats_text = f"""
        <b>Heatmap Statistics:</b>
        <p>• Total Person Detections: <b>{stats['total_detections']}</b></p>
        <p>• Frames Analyzed: <b>{stats['frame_count']}</b></p>
        <p>• Average Detections per Frame: <b>{stats['average_per_frame']:.2f}</b></p>
        <p>• Maximum Heat Intensity: <b>{stats['max_intensity']:.2f}</b></p>
        <p>• Total Heat Intensity: <b>{stats['total_intensity']:.2f}</b></p>
        """
        self.heatmap_stats_label.setText(stats_text)
        
        # Enable save button
        self.save_heatmap_btn.setEnabled(True)
        self.current_heatmap_image = heatmap_overlay

    def get_background_frame(self):
        """Get background frame for heatmap overlay."""
        if not self.current_video_name:
            return None
        
        # Try processed frames first
        processed_dir = f"{self.current_video_name}_processed"
        if os.path.exists(processed_dir):
            frame_files = sorted([f for f in os.listdir(processed_dir) if f.endswith(('.jpg', '.png'))])
            if frame_files:
                first_frame_path = os.path.join(processed_dir, frame_files[0])
                return cv2.imread(first_frame_path)
        
        # Fall back to original frames
        frames_dir = f"{self.current_video_name}_frames"
        if os.path.exists(frames_dir):
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
            if frame_files:
                first_frame_path = os.path.join(frames_dir, frame_files[0])
                return cv2.imread(first_frame_path)
        
        return None

    @Slot()
    def save_heatmap(self):
        """Save current heatmap visualization to file."""
        if self.current_heatmap_image is None:
            QMessageBox.warning(self, "No Heatmap", "No heatmap to save.")
            return
        
        default_name = f"{self.current_video_name}_heatmap.png" if self.current_video_name else "heatmap.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Heatmap", default_name,
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*.*)"
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.current_heatmap_image)
                QMessageBox.information(self, "Success", f"Heatmap saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save heatmap:\n{str(e)}")

    @Slot()
    def clear_heatmap(self):
        """Clear current heatmap data."""
        if self.heatmap_generator is not None:
            reply = QMessageBox.question(
                self,
                "Clear Heatmap",
                "Are you sure you want to clear the heatmap data?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.heatmap_generator = None
                self.current_heatmap_image = None
                self.heatmap_label.setText("Heatmap cleared.\nProcess a video to generate new heatmap.")
                self.heatmap_stats_label.setText("")
                self.save_heatmap_btn.setEnabled(False)


class ObjectTracker(QObject):
    """
    Core tracking engine that processes video frames.

    Performs person detection, tracking, re-identification, and face anonymization.
    Runs in a separate thread to keep UI responsive.
    """

    frameProcessed = Signal(str)
    trackingFinished = Signal(list)
    progress = Signal(str)
    newPersonDetected = Signal(str)
    heatmapData = Signal(list)
    peopleStatistics = Signal(dict)

    def __init__(self, video_name, embedding_db, anonymization_method='blur', parent=None):
        super().__init__(parent)
        self.video_name = video_name
        self.embedding_db = embedding_db
        self.anonymization_method = anonymization_method.lower()

        # Models will be loaded lazily
        self.model = None
        self.tracker = None
        self.face_anonymizer = None

        # Directory paths
        self.input_dir = f"{video_name}_frames"
        self.output_dir = f"{video_name}_processed"

        # Tracking state
        self.track_id_to_person_id = {}  # Maps temporary track IDs to persistent person IDs
        self.track_embeddings_cache = {}  # Cache of recent embeddings per track
        self.detection_data = []  # Accumulated detections for heatmap generation

        # Statistics collection
        self.people_stats = {
            'frame_counts': [],  # People count per frame
            'person_presence': {},  # Frames each person appears in
            'person_trajectories': {},  # Movement paths per person
            'unique_persons_per_frame': [],  # Unique persons per frame
            'total_frames': 0,
            'max_people': 0,
            'min_people': float('inf'),
            'avg_people': 0,
            'median_people': 0,
            'std_people': 0,
            'total_detections': 0,
            'unique_persons': set()  # All unique persons seen
        }

    @Slot()
    def run_tracking(self):
        """
        Main tracking loop - processes all frames in the input directory.

        For each frame:
          1. Anonymize faces
          2. Detect people with YOLO
          3. Track with DeepSort
          4. Re-identify using embedding database
          5. Save processed frame with annotations
        """
        try:
            self.progress.emit(f"Loading models and starting cross-video tracking for '{self.video_name}'...")

            # Initialize models
            self.face_anonymizer = FaceAnonymizer(
                anonymization_method=self.anonymization_method,
                det_threshold=0.3,  # Lower threshold for better face detection
                det_size=(640, 640)
            )

            self.model = YOLO("yolo11x.pt")  # Pre-trained YOLO model
            self.tracker = DeepSort(
                max_age=30,  # Frames to keep lost tracks before removal
                n_init=5,    # Frames to confirm new tracks
                nms_max_overlap=1.0,
                max_cosine_distance=0.22,  # Similarity threshold for track association
                nn_budget=None,
                override_track_class=None,
                embedder="mobilenet",  # Feature extractor for appearance matching
                half=True,  # Use half precision for speed
                bgr=True,
                embedder_gpu=True,
                embedder_model_name=None,
                embedder_wts=None,
                polygon=False,
                today=None
            )

            # Check for input frames
            if not os.path.exists(self.input_dir):
                self.progress.emit(f"Error: No frames found for '{self.video_name}'. Please slice the video first.")
                self.trackingFinished.emit([])
                return

            # Prepare output directory
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir, exist_ok=True)

            # Get sorted list of frame files
            frame_files = sorted(
                [f for f in os.listdir(self.input_dir) if f.endswith(('.jpg', '.png'))]
            )

            if not frame_files:
                self.progress.emit(f"Error: No frames found in '{self.input_dir}'.")
                self.trackingFinished.emit([])
                return

            processed_frame_paths = []
            total_frames = len(frame_files)
            new_persons_count = 0

            # Get frame dimensions from first frame
            first_frame = cv2.imread(os.path.join(self.input_dir, frame_files[0]))
            if first_frame is None:
                self.progress.emit("Error: Could not read first frame for dimensions.")
                self.trackingFinished.emit([])
                return

            frame_height, frame_width = first_frame.shape[:2]
            self.detection_data = []
            self.people_stats['total_frames'] = total_frames

            # Statistics accumulators
            frame_people_counts = []
            current_frame_persons = set()

            # Process each frame
            for i, frame_name in enumerate(frame_files):
                frame_path = os.path.join(self.input_dir, frame_name)
                original_frame = cv2.imread(frame_path)

                if original_frame is None:
                    self.progress.emit(f"Warning: Could not read frame {frame_name}")
                    continue

                # Step 1: Anonymize faces in frame
                anonymized_frame = self.face_anonymizer.anonymize_frame(original_frame)

                # Step 2: Detect people with YOLO (class 0 = person)
                results = self.model.predict(
                    source=anonymized_frame,
                    classes=[0],
                    conf=0.5,  # Confidence threshold
                    verbose=False,
                    imgsz=1280  # Input size for better accuracy
                )

                # Format detections for DeepSort
                detections = []
                frame_detections = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w, h = x2 - x1, y2 - y1
                    conf = box.conf[0].cpu().numpy()
                    detections.append(
                        ([int(x1), int(y1), int(w), int(h)], float(conf), "person")
                    )
                    frame_detections.append([int(x1), int(y1), int(x2), int(y2)])

                # Step 3: Track with DeepSort
                tracks = self.tracker.update_tracks(detections, frame=anonymized_frame)

                # Prepare output frame with annotations
                output_frame = original_frame.copy()
                current_frame_persons.clear()
                frame_person_count = 0

                # Step 4: Process each track
                for track in tracks:
                    if not track.is_confirmed():
                        continue  # Skip unconfirmed tracks

                    track_id = track.track_id
                    ltrb = track.to_tlbr()  # [left, top, right, bottom]
                    frame_person_count += 1

                    # Extract appearance features for re-identification
                    if hasattr(track, 'features') and track.features:
                        recent_feature = track.features[-1]

                        # Cache recent features for this track
                        if track_id not in self.track_embeddings_cache:
                            self.track_embeddings_cache[track_id] = []
                        self.track_embeddings_cache[track_id].append(recent_feature)

                        # Use average of last 5 features for stability
                        recent_features = self.track_embeddings_cache[track_id][-5:]
                        avg_embedding = np.mean(recent_features, axis=0)
                        embedding_list = avg_embedding.tolist()

                        # Re-identification: match against database
                        if track_id in self.track_id_to_person_id:
                            persistent_id = self.track_id_to_person_id[track_id]
                        else:
                            matched_id, similarity = self.embedding_db.find_match(embedding_list)
                            if matched_id:
                                # Known person
                                persistent_id = matched_id
                                self.progress.emit(f"Track {track_id} matched to {persistent_id} (similarity: {similarity:.3f})")
                            else:
                                # New person - add to database
                                persistent_id = self.embedding_db.add_person(embedding_list)
                                self.track_id_to_person_id[track_id] = persistent_id
                                new_persons_count += 1
                                self.newPersonDetected.emit(persistent_id)
                                self.progress.emit(f"New person detected: {persistent_id} (similarity: {similarity:.3f})")
                    else:
                        # No features available - use temporary ID
                        persistent_id = f"Temp_{track_id}"

                    # Update tracking mappings
                    self.track_id_to_person_id[track_id] = persistent_id
                    current_frame_persons.add(persistent_id)

                    # Update statistics
                    if persistent_id not in self.people_stats['person_presence']:
                        self.people_stats['person_presence'][persistent_id] = 0
                    self.people_stats['person_presence'][persistent_id] += 1

                    # Record trajectory point
                    if persistent_id not in self.people_stats['person_trajectories']:
                        self.people_stats['person_trajectories'][persistent_id] = []

                    center_x = (ltrb[0] + ltrb[2]) / 2
                    center_y = (ltrb[1] + ltrb[3]) / 2
                    self.people_stats['person_trajectories'][persistent_id].append({
                        'frame': i,
                        'x': center_x,
                        'y': center_y,
                        'bbox': [ltrb[0], ltrb[1], ltrb[2], ltrb[3]]
                    })

                    # Draw bounding box
                    cv2.rectangle(
                        output_frame,
                        (int(ltrb[0]), int(ltrb[1])),
                        (int(ltrb[2]), int(ltrb[3])),
                        (0, 255, 0),  # Green for all persons
                        2
                    )

                    # Color code: green for known persons, orange for temporary
                    if persistent_id.startswith("Person_"):
                        color = (0, 255, 0)
                        id_text = f"{persistent_id}"
                    else:
                        color = (255, 165, 0)
                        id_text = f"{persistent_id}"

                    # Draw person ID above bounding box
                    cv2.putText(
                        output_frame,
                        id_text,
                        (int(ltrb[0]), int(ltrb[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

                # Update frame-level statistics
                frame_people_counts.append(frame_person_count)
                self.people_stats['unique_persons_per_frame'].append(len(current_frame_persons))
                self.people_stats['unique_persons'].update(current_frame_persons)
                self.people_stats['total_detections'] += frame_person_count

                # Step 5: Save processed frame (with face anonymization applied again)
                final_frame = self.face_anonymizer.anonymize_frame(output_frame)
                output_path = os.path.join(self.output_dir, frame_name)
                cv2.imwrite(output_path, final_frame)
                processed_frame_paths.append(output_path)

                # Accumulate detections for heatmap
                if frame_detections:
                    self.detection_data.append(frame_detections)

                # Signal UI update
                self.frameProcessed.emit(output_path)

                # Progress reporting
                if (i + 1) % 10 == 0 or (i + 1) == total_frames:
                    self.progress.emit(f"Processing '{self.video_name}': {i+1}/{total_frames} frames")

            # Calculate final statistics
            if frame_people_counts:
                self.people_stats['frame_counts'] = frame_people_counts
                self.people_stats['max_people'] = max(frame_people_counts)
                self.people_stats['min_people'] = min(frame_people_counts)
                self.people_stats['avg_people'] = np.mean(frame_people_counts)
                self.people_stats['median_people'] = np.median(frame_people_counts)
                self.people_stats['std_people'] = np.std(frame_people_counts)

            # Final reporting
            self.progress.emit(f"Cross-video tracking complete for '{self.video_name}'. Found {new_persons_count} new persons.")
            self.progress.emit(f"Database now contains {self.embedding_db.get_person_count()} unique persons.")

            # Prepare heatmap data for inference page
            if self.detection_data:
                heatmap_data_with_dims = {
                    'detections': self.detection_data,
                    'frame_width': frame_width,
                    'frame_height': frame_height,
                    'total_frames': total_frames,
                    'people_stats': self.people_stats
                }
                self.heatmapData.emit([heatmap_data_with_dims])

            # Emit final statistics
            self.peopleStatistics.emit(self.people_stats)
            self.trackingFinished.emit(processed_frame_paths)

        except Exception as e:
            self.progress.emit(f"Tracking Error for '{self.video_name}': {e}")
            import traceback
            traceback.print_exc()
            self.trackingFinished.emit([])


    # Add this method to your ObjectTracker class in the app code:
    def save_mot_format(self, output_path):
        """
        Save tracking results in MOT Challenge format for comparison.

        Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>, <class>
        """
        try:
            with open(output_path, 'w') as f:
                # We need to reconstruct frame-by-frame data from trajectories
                # First, collect all track data by frame
                frames_data = {}

                for person_id, trajectory in self.people_stats['person_trajectories'].items():
                    for point in trajectory:
                        frame = point['frame'] + 1  # MOT format uses 1-based indexing
                        bbox = point['bbox']  # [x1, y1, x2, y2]

                        if frame not in frames_data:
                            frames_data[frame] = []

                        # Convert person_id to integer for MOT format
                        # If it's "Person_001", extract 001
                        try:
                            if person_id.startswith("Person_"):
                                track_id = int(person_id.split("_")[1])
                            else:
                                track_id = -1  # Unknown person
                        except:
                               track_id = -1

                        # Calculate bounding box dimensions
                        bb_left = bbox[0]
                        bb_top = bbox[1]
                        bb_width = bbox[2] - bbox[0]
                        bb_height = bbox[3] - bbox[1]

                        # Use a default confidence (you might want to store actual confidences)
                        conf = 1.0

                        # For MOT format, we need x,y,z coordinates (set to -1 if not available)
                        x, y, z = -1, -1, -1

                        frames_data[frame].append({
                            'track_id': track_id,
                            'bb_left': bb_left,
                            'bb_top': bb_top,
                            'bb_width': bb_width,
                            'bb_height': bb_height,
                            'conf': conf,
                            'x': x,
                            'y': y,
                            'z': z
                        })

                # Write in frame order
                for frame in sorted(frames_data.keys()):
                    for detection in frames_data[frame]:
                        line = f"{frame},{detection['track_id']},{detection['bb_left']:.2f},{detection['bb_top']:.2f},"
                        line += f"{detection['bb_width']:.2f},{detection['bb_height']:.2f},{detection['conf']:.2f},"
                        line += f"{detection['x']},{detection['y']},{detection['z']},0\n"
                        f.write(line)

            return True
        except Exception as e:
            print(f"Error saving MOT format: {e}")
            return False




class TrackingPage(QWidget):
    """
    Page for running and viewing person tracking results.
    
    Shows processed frames with annotations and provides playback controls.
    """
    
    newPersonDetected = Signal(str)

    def __init__(self, embedding_db, parent=None):
        super().__init__(parent)
        self.embedding_db = embedding_db
        self.tracker_thread = None
        self.tracker = None
        self.processed_frames = []
        self.current_frame_index = 0
        self.current_video_name = None
        self.current_anonymization_method = 'blur'
        
        # Timer for frame replay
        self.replay_timer = QTimer(self)
        self.replay_timer.timeout.connect(self.next_frame)
        self.replay_timer.setInterval(100)  # Default 10 FPS
        
        layout = QVBoxLayout(self)
        
        # Main display for processed frames
        self.image_display = QLabel("Select and process a video in the Input tab first.")
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout.addWidget(self.image_display, 1)
        
        # Status display
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Start tracking button
        self.start_button = QPushButton("Start Tracking")
        self.start_button.clicked.connect(self.start_tracking)
        layout.addWidget(self.start_button)
        
        # Replay controls (hidden initially)
        self.replay_controls = self._create_replay_controls()
        layout.addWidget(self.replay_controls)
        self.replay_controls.setVisible(False)

        """
        self.save_mot_button = QPushButton("Save MOT Format")
        self.save_mot_button.clicked.connect(self.save_mot_format)
        self.save_mot_button.setEnabled(False)
        layout.addWidget(self.save_mot_button)
        """

    def _create_replay_controls(self):
        """Create playback controls for reviewing processed frames."""
        controls_widget = QWidget()
        layout = QVBoxLayout(controls_widget)
        
        # Seek slider
        seek_layout = QHBoxLayout()
        self.replay_current_label = QLabel("0")
        self.replay_slider = QSlider(Qt.Orientation.Horizontal)
        self.replay_slider.setValue(0)
        self.replay_slider.sliderMoved.connect(self.set_frame)
        self.replay_total_label = QLabel("0")
        seek_layout.addWidget(self.replay_current_label)
        seek_layout.addWidget(self.replay_slider)
        seek_layout.addWidget(self.replay_total_label)
        layout.addLayout(seek_layout)
        
        # Playback buttons
        buttons_layout = QHBoxLayout()
        self.replay_play_pause = QPushButton("Play")
        self.replay_play_pause.clicked.connect(self.toggle_replay)
        self.replay_stop = QPushButton("Stop")
        self.replay_stop.clicked.connect(self.stop_replay)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.replay_play_pause)
        buttons_layout.addWidget(self.replay_stop)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        return controls_widget

    def set_video_name(self, video_name):
        """Update UI for the current video."""
        self.current_video_name = video_name
        if video_name:
            self.image_display.setText(f"Ready for cross-video tracking of '{video_name}'. Click 'Start' to begin.")
            self.start_button.setEnabled(True)
        else:
            self.image_display.setText("Select and process a video in the Input tab first.")
            self.start_button.setEnabled(False)

    def set_anonymization_method(self, method):
        """Set the anonymization method for tracking."""
        self.current_anonymization_method = method.lower()

    @Slot()
    def start_tracking(self):
        """Start the tracking process in a background thread."""
        if not self.current_video_name:
            self.status_label.setText("Error: No video selected. Please process a video first.")
            return
        
        # Check for sliced frames
        frames_dir = f"{self.current_video_name}_frames"
        if not os.path.exists(frames_dir):
            self.status_label.setText(f"Error: No frames found for '{self.current_video_name}'. Please slice the video first.")
            return
        
        # Reset UI state
        self.stop_replay()
        self.processed_frames = []
        self.replay_controls.setVisible(False)
        self.status_label.setVisible(True)
        self.start_button.setEnabled(False)
        self.start_button.setText("Processing (Cross-Video Tracking)...")
        
        # Get anonymization method from input page
        main_window = self.window()
        if hasattr(main_window, 'input_page'):
            method = main_window.input_page.anonymization_combo.currentText().lower()
            self.set_anonymization_method(method)
        
        # Create and start tracker thread
        self.tracker_thread = QThread()
        self.tracker = ObjectTracker(self.current_video_name, self.embedding_db, self.current_anonymization_method)
        self.tracker.moveToThread(self.tracker_thread)
        
        # Connect signals
        self.tracker_thread.started.connect(self.tracker.run_tracking)
        self.tracker.progress.connect(self.status_label.setText)
        self.tracker.frameProcessed.connect(self.display_frame)
        self.tracker.trackingFinished.connect(self.on_tracking_finished)
        self.tracker.newPersonDetected.connect(self.newPersonDetected)
        self.tracker.heatmapData.connect(main_window.on_heatmap_data_received)
        self.tracker.peopleStatistics.connect(main_window.on_people_statistics_received)
        self.tracker.trackingFinished.connect(self.tracker_thread.quit)
        self.tracker.trackingFinished.connect(self.tracker.deleteLater)
        self.tracker_thread.finished.connect(self.tracker_thread.deleteLater)
        
        self.tracker_thread.start()

    @Slot(str)
    def display_frame(self, image_path):
        """Update display with newly processed frame."""
        QTimer.singleShot(0, lambda: self._update_display(image_path))

    def _update_display(self, image_path):
        """Load and display image with proper scaling."""
        pixmap = QPixmap(image_path)
        self.image_display.setPixmap(pixmap.scaled(
            self.image_display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    @Slot(list)
    def on_tracking_finished(self, processed_frame_paths):
        """Handle completion of tracking process."""
        self.processed_frames = processed_frame_paths
        self.start_button.setEnabled(True)
        self.start_button.setText("Start Tracking")
        self.start_button.setVisible(False)
        self.status_label.setVisible(False)
        
        if self.processed_frames:
            # Show replay controls
            self.replay_controls.setVisible(True)
            self.replay_slider.setRange(0, len(self.processed_frames) - 1)
            self.replay_total_label.setText(str(len(self.processed_frames) - 1))
            self.set_frame(0)
            
            # Enable inference page in main window
            main_window = self.window()
            if main_window:
                main_window.enable_inference_page()
                main_window.inference_page.update_statistics()

            self.save_mot_button.setEnabled(True)

        else:
            # Tracking failed
            self.status_label.setText(f"Cross-video tracking failed or no frames found for '{self.current_video_name}'.")
            self.status_label.setVisible(True)
            self.start_button.setVisible(True)

    @Slot()
    def toggle_replay(self):
        """Toggle between play and pause for frame replay."""
        if self.replay_timer.isActive():
            self.replay_timer.stop()
            self.replay_play_pause.setText("Play")
        else:
            self.replay_timer.start()
            self.replay_play_pause.setText("Pause")

    @Slot()
    def stop_replay(self):
        """Stop replay and reset to first frame."""
        self.replay_timer.stop()
        self.replay_play_pause.setText("Play")
        self.set_frame(0)

    @Slot()
    def next_frame(self):
        """Advance to next frame in replay."""
        next_index = self.current_frame_index + 1
        if next_index >= len(self.processed_frames):
            self.stop_replay()
            return
        self.set_frame(next_index)

    @Slot(int)
    def set_frame(self, index):
        """Display specific frame by index."""
        if 0 <= index < len(self.processed_frames):
            self.current_frame_index = index
            self.replay_slider.setValue(index)
            self.replay_current_label.setText(str(index))
            self._update_display(self.processed_frames[index])


    @Slot()
    def save_mot_format(self):
        """Save tracking results in MOT format for evaluation."""
        if not self.current_video_name:
            QMessageBox.warning(self, "No Video", "No video processed yet.")
            return

        default_name = f"{self.current_video_name}_tracking_mot.txt"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save MOT Format Results", default_name,
            "Text Files (*.txt);;All Files (*.*)"
        )

        if file_path and self.tracker:
            if hasattr(self.tracker, 'save_mot_format'):
                success = self.tracker.save_mot_format(file_path)
                if success:
                    QMessageBox.information(self, "Success",
                        f"MOT format results saved to:\n{file_path}\n\n"
                        f"You can now compare with the provided det.txt file.")
                else:
                    QMessageBox.warning(self, "Error", "Failed to save MOT format.")


class VideoSlicer(QObject):
    """
    Worker for slicing videos into frames at specified intervals.

    Runs in a separate thread to avoid UI freezing during processing.
    """

    finished = Signal()
    progress = Signal(str)
    progress_percent = Signal(int)

    @Slot(str, int, str)
    def slice_video(self, file_path, interval_ms, output_dir):
        """
        Extract frames from video at regular time intervals.

        Args:
            file_path: Path to input video file
            interval_ms: Time interval between extracted frames in milliseconds
            output_dir: Directory to save extracted frames
        """
        try:
            self.progress.emit(f"Starting video slicing to '{output_dir}'...")
            cap = cv2.VideoCapture(file_path)

            if not cap.isOpened():
                self.progress.emit(f"Error: Could not open video file.")
                self.finished.emit()
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps == 0:
                self.progress.emit("Error: Video FPS is zero.")
                self.finished.emit()
                return

            # Calculate frame skip based on desired time interval
            frame_skip = int((fps * interval_ms) / 1000)
            if frame_skip == 0:
                frame_skip = 1  # Minimum 1 frame

            frames_to_save = total_frames // frame_skip + 1

            # Create output directory
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            saved_count = 0
            frame_count = 0

            while True:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                if not ret:
                    break

                file_name = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                cv2.imwrite(file_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                saved_count += 1

                # Update progress every 10 frames
                if saved_count % 10 == 0:
                    self.progress.emit(f"Saved frame {saved_count}...")
                    if total_frames > 0:
                        percent = int((frame_count / total_frames) * 100)
                        self.progress_percent.emit(min(percent, 100))

                frame_count += frame_skip
                QThread.msleep(1)  # Small delay to prevent UI freezing

            cap.release()
            self.progress.emit(f"Slicing complete. {saved_count} frames saved to '{output_dir}'.")
            self.progress_percent.emit(100)
            self.finished.emit()

        except Exception as e:
            self.progress.emit(f"Error during slicing: {e}")
            self.finished.emit()

class InputPage(QWidget):
    """
    First page of the application for video selection and slicing.
    
    Provides video preview, interval selection, and slicing controls.
    """
    
    fileSelected = Signal(str)
    frameIntervalChanged = Signal(int)
    proceedClicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_file_path = None
        self.current_video_name = None
        self.slice_thread = None
        self.slicer = None
        
        # Set up media player for video preview
        self.media_player = QMediaPlayer()
        self.view_stack = QStackedWidget(self)
        self.browse_page = self._create_browse_page()
        self.player_page = self._create_player_page()
        
        self.view_stack.addWidget(self.browse_page)
        self.view_stack.addWidget(self.player_page)
        
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.view_stack)
        
        # Initialize UI state
        self.on_slider_changed(self.interval_slider.value())
        
        # Connect media player signals
        self.media_player.positionChanged.connect(self.update_position)
        self.media_player.durationChanged.connect(self.update_duration)
        self.media_player.playbackStateChanged.connect(self.update_play_pause_button)

    def _create_browse_page(self):
        """Create the file selection interface shown initially."""
        browse_widget = QWidget()
        layout = QVBoxLayout(browse_widget)
        
        self.prompt_label = QLabel("Select a video file to begin processing:")
        layout.addWidget(self.prompt_label)
        
        self.browse_button = QPushButton("Browse Files...")
        self.browse_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.browse_button)
        
        self.selected_file_label = QLabel("No file selected.")
        self.selected_file_label.setWordWrap(True)
        layout.addWidget(self.selected_file_label)
        layout.addStretch()
        
        return browse_widget

    def _create_player_page(self):
        """Create the video preview and slicing interface."""
        player_widget = QWidget()
        layout = QVBoxLayout(player_widget)
        
        # Video playback widget
        self.video_widget = QVideoWidget()
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.setLoops(QMediaPlayer.Loops.Infinite)
        layout.addWidget(self.video_widget, 1)
        
        # Seek controls
        seek_layout = QHBoxLayout()
        self.current_time_label = QLabel("00:00")
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 100)
        self.seek_slider.sliderMoved.connect(self.media_player.setPosition)
        self.total_time_label = QLabel("00:00")
        seek_layout.addWidget(self.current_time_label)
        seek_layout.addWidget(self.seek_slider)
        seek_layout.addWidget(self.total_time_label)
        layout.addLayout(seek_layout)
        
        # Playback controls
        controls_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("Pause")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_playback)
        controls_layout.addStretch()
        controls_layout.addWidget(self.play_pause_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Divider
        self.divider = QFrame()
        self.divider.setFrameShape(QFrame.Shape.HLine)
        self.divider.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(self.divider)
        
        # Frame interval selection
        slider_layout = QHBoxLayout()
        slider_label = QLabel("Frame Interval:")
        self.interval_slider = QSlider(Qt.Orientation.Horizontal)
        self.interval_slider.setRange(100, 2000)  # 100ms to 2s
        self.interval_slider.setValue(100)
        self.interval_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.interval_slider.setTickInterval(100)
        self.interval_value_label = QLabel("100 ms")
        self.interval_slider.valueChanged.connect(self.on_slider_changed)
        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(self.interval_slider)
        slider_layout.addWidget(self.interval_value_label)
        layout.addLayout(slider_layout)
        
        # Anonymization method selection
        anonymization_layout = QHBoxLayout()
        anonymization_label = QLabel("Anonymization Method:")
        self.anonymization_combo = QComboBox()
        self.anonymization_combo.addItems(["Blur", "Pixelate"])
        self.anonymization_combo.setCurrentIndex(0)
        anonymization_layout.addWidget(anonymization_label)
        anonymization_layout.addWidget(self.anonymization_combo)
        anonymization_layout.addStretch()
        layout.addLayout(anonymization_layout)
        
        # Progress indicators
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Action buttons
        bottom_buttons_layout = QHBoxLayout()
        self.proceed_button = QPushButton("Proceed")
        self.proceed_button.clicked.connect(self._on_proceed_clicked)
        self.remove_button = QPushButton("Remove Video")
        self.remove_button.clicked.connect(self._show_browse_view)
        bottom_buttons_layout.addWidget(self.proceed_button)
        bottom_buttons_layout.addWidget(self.remove_button)
        layout.addLayout(bottom_buttons_layout)
        
        return player_widget

    def format_time(self, ms):
        """Convert milliseconds to MM:SS format."""
        if ms < 0: ms = 0
        seconds = (ms // 1000) % 60
        minutes = (ms // (1000 * 60)) % 60
        return f"{minutes:02}:{seconds:02}"

    @Slot()
    def toggle_play_pause(self):
        """Toggle between play and pause states."""
        if self.media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    @Slot()
    def stop_playback(self):
        """Stop video playback and reset to beginning."""
        self.media_player.stop()

    @Slot(int)
    def update_position(self, position):
        """Update seek slider and time label with current position."""
        self.seek_slider.setValue(position)
        self.current_time_label.setText(self.format_time(position))

    @Slot(int)
    def update_duration(self, duration):
        """Update seek slider range with total video duration."""
        self.seek_slider.setRange(0, duration)
        self.total_time_label.setText(self.format_time(duration))

    @Slot(QMediaPlayer.PlaybackState)
    def update_play_pause_button(self, state):
        """Update button text based on playback state."""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_pause_button.setText("Pause")
        else:
            self.play_pause_button.setText("Play")

    @Slot()
    def open_file_dialog(self):
        """Show file dialog for video selection."""
        video_filters = "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", video_filters
        )
        
        if file_path:
            self.current_file_path = file_path
            self.current_video_name = os.path.splitext(os.path.basename(file_path))[0]
            self.selected_file_label.setText(f"Selected: {file_path}")
            
            # Load and play selected video
            self.fileSelected.emit(file_path)
            self.media_player.setSource(QUrl.fromLocalFile(file_path))
            self.media_player.play()
            self.view_stack.setCurrentIndex(1)  # Switch to player view
        else:
            # No file selected - reset state
            self.current_file_path = None
            self.current_video_name = None
            self.selected_file_label.setText("No file selected.")
            self.fileSelected.emit("")

    @Slot()
    def _show_browse_view(self):
        """Reset to file selection view."""
        self.media_player.stop()
        self.media_player.setSource(QUrl())
        self.current_file_path = None
        self.current_video_name = None
        self.selected_file_label.setText("No file selected.")
        self.fileSelected.emit("")
        
        # Reset UI elements
        self.seek_slider.setValue(0)
        self.current_time_label.setText("00:00")
        self.total_time_label.setText("00:00")
        self.status_label.setText("")
        self.progress_bar.setVisible(False)
        self.view_stack.setCurrentIndex(0)  # Switch back to browse view

    @Slot(int)
    def on_slider_changed(self, value):
        """Handle frame interval slider changes."""
        self.interval_value_label.setText(f"{value} ms")
        self.frameIntervalChanged.emit(value)

    @Slot(str)
    def update_status_label(self, message):
        """Update status label with progress messages."""
        self.status_label.setText(message)

    @Slot(int)
    def update_progress_bar(self, value):
        """Update progress bar percentage."""
        self.progress_bar.setValue(value)

    @Slot()
    def on_slicing_finished(self):
        """Handle completion of video slicing."""
        self.proceed_button.setEnabled(True)
        self.remove_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        
        # Clean up thread
        if self.slice_thread:
            self.slice_thread.quit()
            self.slice_thread.wait()
            self.slice_thread.deleteLater()
            self.slice_thread = None
        
        if self.slicer:
            self.slicer.deleteLater()
            self.slicer = None
        
        # Signal that we're ready to proceed
        self.proceedClicked.emit(self.current_video_name)

    @Slot()
    def _on_proceed_clicked(self):
        """Start video slicing process."""
        if not self.current_file_path or not self.current_video_name:
            self.update_status_label("Error: No file selected.")
            return
        
        output_dir = f"{self.current_video_name}_frames"
        
        # Check if output directory already exists
        if os.path.exists(output_dir):
            reply = QMessageBox.question(
                self,
                "Directory Exists",
                f"A folder named '{output_dir}' already exists with sliced frames.\n"
                f"Do you want to overwrite it?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel
            )
            if reply == QMessageBox.StandardButton.Cancel:
                self.update_status_label("Slicing cancelled.")
                return
            else:
                # Remove existing directory
                try:
                    shutil.rmtree(output_dir)
                    os.makedirs(output_dir, exist_ok=True)
                except Exception as e:
                    self.update_status_label(f"Error clearing directory: {e}")
                    return
        
        # Stop video playback during slicing
        self.media_player.stop()
        self.proceed_button.setEnabled(False)
        self.remove_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start slicing in background thread
        file_path = self.current_file_path
        interval = self.interval_slider.value()
        
        self.slice_thread = QThread()
        self.slicer = VideoSlicer()
        self.slicer.moveToThread(self.slice_thread)
        
        # Connect signals
        self.slice_thread.started.connect(
            lambda: self.slicer.slice_video(file_path, interval, output_dir)
        )
        self.slicer.progress.connect(self.update_status_label)
        self.slicer.progress_percent.connect(self.update_progress_bar)
        self.slicer.finished.connect(self.on_slicing_finished)
        self.slicer.finished.connect(self.slice_thread.quit)
        self.slice_thread.finished.connect(self.slice_thread.deleteLater)
        
        self.slice_thread.start()

class HomePage(QWidget):
    """
    Application home page showing dashboard and database status.
    
    Provides overview of cross-video tracking system and database management.
    """
    
    def __init__(self, embedding_db, parent=None):
        super().__init__(parent)
        self.embedding_db = embedding_db
        self.layout = QVBoxLayout(self)
        
        # Title section
        self.title_label = QLabel("<h1>AnonTrack</h1>")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.title_label)
        
        self.subtitle_label = QLabel("<h2>Persistent Anonymous Person Tracking</h2>")
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.subtitle_label)
        
        # Dashboard frame
        self.dashboard_frame = QFrame()
        self.dashboard_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        
        dashboard_layout = QVBoxLayout(self.dashboard_frame)
        dashboard_title = QLabel("<h3>Cross-Video Re-identification Dashboard</h3>")
        dashboard_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dashboard_layout.addWidget(dashboard_title)
        
        # Database statistics
        self.embeddings_count_label = QLabel("")
        self.embeddings_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dashboard_layout.addWidget(self.embeddings_count_label)
        
        # Database management
        self.clear_db_button = QPushButton("Clear the embedding database")
        self.clear_db_button.clicked.connect(self.clear_database)
        dashboard_layout.addWidget(self.clear_db_button)
        
        # Recent activity
        self.recent_activity_label = QLabel("<b>Recent Activity:</b>")
        dashboard_layout.addWidget(self.recent_activity_label)
        
        self.recent_activity_list = QListWidget()
        self.recent_activity_list.setMaximumHeight(150)
        dashboard_layout.addWidget(self.recent_activity_list)
        
        self.layout.addWidget(self.dashboard_frame)
        
        # Workflow description
        description = QLabel(
            "<p><b>Workflow:</b></p>"
            "<ol>"
            "<li>Upload video in the <b>Input</b> tab</li>"
            "<li>Process video for tracking in the <b>Detection & Tracking</b> tab</li>"
            "<li>View analytics and insights in the <b>Inference</b> tab</li>"
            "</ol>"
            "<p>The system will <b>remember persons across videos</b> while maintaining complete anonymity.</p>"
        )
        description.setWordWrap(True)
        self.layout.addWidget(description)
        
        # Initial display update
        self.update_display()

    def update_display(self):
        """Refresh dashboard with current database statistics."""
        person_count = self.embedding_db.get_person_count()
        self.embeddings_count_label.setText(
            f"<span style='font-size: 16pt;'>"
            f"<b>{person_count} Unique Persons</b> stored in database"
            f"</span>"
        )
        
        # Show most recent persons
        all_persons = self.embedding_db.get_all_persons()
        self.recent_activity_list.clear()
        if all_persons:
            for person_id in sorted(all_persons)[-5:]:  # Last 5 persons
                self.recent_activity_list.addItem(f"{person_id} - Stored in database")
        else:
            self.recent_activity_list.addItem("No persons stored yet. Process a video to begin.")

    @Slot()
    def clear_database(self):
        """Clear all stored person data with confirmation."""
        reply = QMessageBox.question(
            self,
            "Clear Person Database",
            "Are you sure you want to clear ALL stored person data?\n\n"
            "This will delete all person embeddings and reset the cross-video tracking.\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.embedding_db.clear()
            self.update_display()
            
            # Notify other components
            main_window = self.window()
            if hasattr(main_window, 'on_database_cleared'):
                main_window.on_database_cleared()
            
            QMessageBox.information(
                self,
                "Database Cleared",
                "All person data has been cleared from the database."
               )

    @Slot(str)
    def on_new_person_detected(self, person_id):
        """Add newly detected person to recent activity list."""
        self.recent_activity_list.addItem(f"New: {person_id} - Added to database")
        self.update_display()

class MainWindow(QWidget):
    """
    Main application window managing navigation between pages.
    
    Coordinates communication between different components and manages
    application state.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AnonTrack")
        self.current_video_path = None
        self.current_video_name = None
        self.current_frame_interval = 100  # Default 100ms interval
        
        # Initialize embedding database
        self.embedding_db = PersonEmbeddingDB(similarity_threshold=0.85)
        
        main_layout = QHBoxLayout(self)
        
        # Left navigation menu
        self.menu = QListWidget()
        self.menu.addItem(QListWidgetItem("Home"))
        self.menu.addItem(QListWidgetItem("Input"))
        self.menu.addItem(QListWidgetItem("Detection & Tracking"))
        self.menu.addItem(QListWidgetItem("Inference"))
        self.menu.setFixedWidth(150)
        main_layout.addWidget(self.menu)
        
        # Right content area (stacked widget)
        self.stack = QStackedWidget()
        
        # Create pages
        self.home_page = HomePage(self.embedding_db, self)
        self.stack.addWidget(self.home_page)
        
        self.input_page = InputPage()
        self.stack.addWidget(self.input_page)
        
        self.tracking_page = TrackingPage(self.embedding_db)
        self.stack.addWidget(self.tracking_page)
        
        self.inference_page = InferencePage(self.embedding_db, self)
        self.stack.addWidget(self.inference_page)
        
        main_layout.addWidget(self.stack)
        
        # Connect navigation
        self.menu.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.menu.setCurrentRow(0)  # Start at Home page
        
        # Initially disable tracking and inference pages
        self.menu.item(2).setFlags(Qt.ItemFlag.NoItemFlags)
        self.menu.item(3).setFlags(Qt.ItemFlag.NoItemFlags)

        # Connect page signals
        self.input_page.fileSelected.connect(self.on_file_selected)
        self.input_page.frameIntervalChanged.connect(self.on_frame_interval_changed)
        self.input_page.proceedClicked.connect(self.on_proceed_clicked)
        self.tracking_page.newPersonDetected.connect(self.on_new_person_detected)

    def enable_inference_page(self):
        """Enable the inference page in navigation menu."""
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        self.menu.item(3).setFlags(flags)

    @Slot(str)
    def on_file_selected(self, file_path):
        """Handle video file selection from input page."""
        self.current_video_path = file_path
        
        if file_path:
            self.current_video_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Enable tracking page
            flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            self.menu.item(2).setFlags(flags)
            
            # Enable inference page only if tracking already completed
            if self.tracking_page.processed_frames:
                self.enable_inference_page()
            else:
                self.menu.item(3).setFlags(Qt.ItemFlag.NoItemFlags)
        else:
            # No file selected - reset state
            self.current_video_name = None
            self.menu.item(2).setFlags(Qt.ItemFlag.NoItemFlags)
            self.menu.item(3).setFlags(Qt.ItemFlag.NoItemFlags)
            self.tracking_page.set_video_name(None)
        
        # Reset tracking page state
        self.tracking_page.stop_replay()
        self.tracking_page.processed_frames = []
        self.tracking_page.replay_controls.setVisible(False)
        self.tracking_page.status_label.setVisible(True)
        self.tracking_page.status_label.setText("")
        self.tracking_page.start_button.setVisible(True)
        
        if file_path:
            self.tracking_page.image_display.setText(
                f"Ready for cross-video tracking of '{self.current_video_name}'. Click 'Start' to begin."
            )
        else:
            self.tracking_page.image_display.setText("Select and process a video in the Input tab first.")

    @Slot(int)
    def on_frame_interval_changed(self, interval):
        """Update frame interval for video playback."""
        self.current_frame_interval = interval
        self.tracking_page.replay_timer.setInterval(interval)

    @Slot(str)
    def on_proceed_clicked(self, video_name):
        """Handle proceed button click after video slicing."""
        self.current_video_name = video_name
        
        # Enable tracking page and switch to it
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        self.menu.item(2).setFlags(flags)
        self.menu.setCurrentRow(2)
        
        # Pass video info to other pages
        self.tracking_page.set_video_name(video_name)
        self.inference_page.set_video_name(video_name)
        
        # Get anonymization method from input page
        method = self.input_page.anonymization_combo.currentText().lower()
        self.tracking_page.set_anonymization_method(method)

    @Slot(str)
    def on_new_person_detected(self, person_id):
        """Handle new person detection notification."""
        self.home_page.on_new_person_detected(person_id)
        self.inference_page.update_statistics()

    @Slot(list)
    def on_heatmap_data_received(self, heatmap_data):
        """Forward heatmap data to inference page."""
        self.inference_page.on_heatmap_data_received(heatmap_data)

    @Slot(dict)
    def on_people_statistics_received(self, people_stats):
        """Forward people statistics to inference page."""
        self.inference_page.on_people_statistics_received(people_stats)

    @Slot()
    def on_database_cleared(self):
        """Handle database clear notification."""
        self.home_page.update_display()
        self.inference_page.update_statistics()
        
        # Update tracking page status if visible
        if self.stack.currentWidget() == self.tracking_page:
            self.tracking_page.status_label.setText("Database cleared.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())
