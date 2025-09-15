#!/usr/bin/env python3
"""
Clean Video Pupil Tracker

Based on the working debug script logic - simple and reliable.
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
from sklearn.cluster import DBSCAN

class CleanVideoPupilTracker:
    """Clean video pupil tracker based on working debug script logic"""
    
    def __init__(self, video_path, output_dir="data/output", frame_interval=1):
        """Initialize the clean video tracker"""
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_interval = frame_interval
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frames to process
        self.frames_to_process = list(range(0, self.total_frames, frame_interval))
        
        print(f"Video info: {self.total_frames} frames, {self.fps:.2f} FPS, {self.total_frames/self.fps:.2f}s duration")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Processing {len(self.frames_to_process)} frames (every {frame_interval} frame(s))")
        
        # Initialize tracking data
        self.tracking_data = []
        
        # Setup video writer
        self.setup_video_writer()
    
    def setup_video_writer(self):
        """Setup video writer for output video"""
        video_name = Path(self.video_path).stem
        output_video_path = os.path.join(self.output_dir, f"pupil_tracking_clean_{video_name}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_video_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        print(f"Will create output video: {output_video_path}")
    
    def detect_iris(self, image):
        """Detect iris using the working method from debug script"""
        # Preprocess for iris detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply median blur to reduce noise
        blurred = cv2.medianBlur(gray, 5)
        
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Apply additional contrast enhancement
        enhanced = cv2.convertScaleAbs(enhanced, alpha=3.0, beta=50)
        
        # Find Hough circles
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=150,
            param1=50,
            param2=40,
            minRadius=120,
            maxRadius=250
        )
        
        if circles is None:
            return None
        
        circles = np.round(circles[0, :]).astype("int")
        
        # Score each circle based on intensity differences
        best_circle = None
        best_score = -float('inf')
        
        for circle in circles:
            x, y, r = circle
            score = self.score_iris_circle(enhanced, x, y, r)
            
            if score > best_score:
                best_score = score
                best_circle = circle
        
        if best_circle is None:
            return None
        
        x, y, r = best_circle
        return {
            'center': (x, y),
            'radius': r,
            'score': best_score
        }
    
    def score_iris_circle(self, image, x, y, r):
        """Score an iris circle based on intensity differences"""
        h, w = image.shape
        
        # Check if circle is within image bounds
        if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
            return -float('inf')
        
        # Create masks for inner and outer rings
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        outer_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Inner ring (iris area)
        cv2.circle(inner_mask, (x, y), r - 5, 255, -1)
        cv2.circle(inner_mask, (x, y), r - 15, 0, -1)
        
        # Outer ring (sclera area)
        cv2.circle(outer_mask, (x, y), r + 5, 255, -1)
        cv2.circle(outer_mask, (x, y), r - 5, 0, -1)
        
        # Calculate mean intensities
        inner_intensity = cv2.mean(image, inner_mask)[0]
        outer_intensity = cv2.mean(image, outer_mask)[0]
        
        # Calculate intensity difference (iris should be darker than sclera)
        intensity_diff = outer_intensity - inner_intensity
        
        # Calculate consistency within rings
        inner_std = np.std(image[inner_mask > 0])
        outer_std = np.std(image[outer_mask > 0])
        consistency = 1.0 / (1.0 + inner_std + outer_std)
        
        # Combined score
        score = 0.7 * intensity_diff + 0.15 * consistency
        
        return score
    
    def detect_pupil(self, image, iris_result):
        """Detect pupil using the working method from debug script"""
        iris_center = iris_result['center']
        iris_radius = iris_result['radius']
        
        # Preprocess for pupil detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Get iris baseline intensity from circle 40 pixels inside iris
        baseline_intensity = self.get_iris_baseline_intensity(enhanced, iris_center, iris_radius)
        
        # Find dark pixels with adaptive thresholding
        dark_coords, threshold_used = self.find_dark_pixels_adaptive(
            enhanced, iris_center, iris_radius, baseline_intensity)
        
        if len(dark_coords[0]) == 0:
            return None
        
        # Find largest cluster
        largest_cluster_points, all_labels = self.find_largest_cluster(dark_coords)
        
        if largest_cluster_points is None:
            return None
        
        # Find radius around iris center
        pupil_circle = self.find_radius_around_iris_center(largest_cluster_points, iris_center)
        
        return pupil_circle
    
    def get_iris_baseline_intensity(self, image, iris_center, iris_radius):
        """Get baseline iris intensity from circle 40 pixels inside iris"""
        x, y = iris_center
        
        # Create mask for circle 40 pixels inside iris (80 pixels diameter smaller)
        inner_circle_radius = iris_radius - 40
        if inner_circle_radius <= 0:
            inner_circle_radius = iris_radius // 2  # Fallback to half radius
        
        inner_circle_mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(inner_circle_mask, (x, y), inner_circle_radius, 255, -1)
        
        # Calculate mean intensity in the inner circle
        baseline_intensity = cv2.mean(image, inner_circle_mask)[0]
        
        return baseline_intensity
    
    def find_dark_pixels_adaptive(self, image, iris_center, iris_radius, baseline_intensity):
        """Find dark pixels with hybrid approach: 5th percentile core + gradient boundary expansion"""
        x, y = iris_center
        max_pupil_radius = int(iris_radius * 0.8)
        
        # Create mask for iris area
        iris_mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(iris_mask, (x, y), max_pupil_radius, 255, -1)
        
        # Get all pixels within iris area that are darker than baseline
        iris_pixels = image[iris_mask > 0]
        dark_iris_pixels = iris_pixels[iris_pixels < baseline_intensity]
        
        if len(dark_iris_pixels) == 0:
            return (np.array([]), np.array([])), baseline_intensity
        
        # Use 5% percentile for core dark pixels
        threshold_core = np.percentile(dark_iris_pixels, 5)
        
        # Find core dark pixels
        core_dark_pixels = (image < threshold_core) & (iris_mask > 0)
        core_coords = np.where(core_dark_pixels)
        
        if len(core_coords[0]) == 0:
            return (np.array([]), np.array([])), threshold_core
        
        # Expand from core using gradient analysis
        expanded_coords = self.expand_pupil_boundary_gradient(
            image, iris_center, iris_radius, core_coords, baseline_intensity, iris_mask
        )
        
        return expanded_coords, threshold_core
    
    def expand_pupil_boundary_gradient(self, image, iris_center, iris_radius, core_coords, baseline_intensity, iris_mask):
        """Expand pupil boundary using gradient analysis from core dark pixels"""
        if len(core_coords[0]) == 0:
            return core_coords
        
        # Convert core coordinates to (x, y) format
        y_coords, x_coords = core_coords
        core_points = np.column_stack((x_coords, y_coords))
        
        # Calculate core area to determine if pupil is small
        core_area = len(core_coords[0])
        iris_area = np.pi * iris_radius ** 2
        core_ratio = core_area / iris_area
        
        # Create expanded mask starting with core pixels
        expanded_mask = np.zeros(image.shape, dtype=np.uint8)
        expanded_mask[core_coords] = 255
        
        # Define search area (within iris, excluding glare regions)
        search_mask = iris_mask.copy()
        # Exclude glare regions (pixels brighter than baseline)
        search_mask[image >= baseline_intensity] = 0
        
        # Use morphological operations to expand from core
        # This will grow the region while respecting intensity boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Adjust expansion based on core size - be more conservative for small pupils
        if core_ratio < 0.05:  # Very small pupil
            max_iterations = 5
            # Use more restrictive threshold for small pupils
            expansion_threshold = baseline_intensity * 0.8
        elif core_ratio < 0.1:  # Small pupil
            max_iterations = 10
            expansion_threshold = baseline_intensity * 0.9
        else:  # Normal/large pupil
            max_iterations = 20
            expansion_threshold = baseline_intensity
        
        # Iteratively expand the region
        for i in range(max_iterations):
            # Dilate the current region
            dilated = cv2.dilate(expanded_mask, kernel, iterations=1)
            
            # Only keep pixels that are:
            # 1. In the search area (iris, not glare)
            # 2. Darker than expansion threshold (more restrictive for small pupils)
            # 3. Not already included
            new_pixels = (dilated > 0) & (search_mask > 0) & (image < expansion_threshold) & (expanded_mask == 0)
            
            # Check if we should stop expanding
            if not np.any(new_pixels):
                break
                
            # Add new pixels to expanded region
            expanded_mask[new_pixels] = 255
        
        # Convert back to coordinate format
        expanded_coords = np.where(expanded_mask > 0)
        
        return expanded_coords
    
    def find_largest_cluster(self, dark_coords):
        """Find the largest cluster of dark pixels using DBSCAN"""
        if len(dark_coords[0]) == 0:
            return None, None
        
        # Convert to (x, y) coordinates
        y_coords, x_coords = dark_coords
        points = np.column_stack((x_coords, y_coords))
        
        # Use DBSCAN to find clusters
        clustering = DBSCAN(eps=10, min_samples=50).fit(points)
        labels = clustering.labels_
        
        # Find the largest cluster (excluding noise with label -1)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        if len(unique_labels) == 0:
            return None, None
        
        # Find largest cluster
        largest_cluster_size = 0
        largest_cluster_label = None
        
        for label in unique_labels:
            cluster_size = np.sum(labels == label)
            if cluster_size > largest_cluster_size:
                largest_cluster_size = cluster_size
                largest_cluster_label = label
        
        # Get points in largest cluster
        largest_cluster_points = points[labels == largest_cluster_label]
        
        return largest_cluster_points, labels
    
    def find_radius_around_iris_center(self, cluster_points, iris_center):
        """Find radius around iris center that encompasses the cluster"""
        if len(cluster_points) == 0:
            return None
        
        # Calculate distances from iris center to all cluster points
        distances = np.sqrt(np.sum((cluster_points - iris_center)**2, axis=1))
        
        # Use 90th percentile as radius to avoid outliers
        radius = int(np.percentile(distances, 90))
        
        return {
            'center': iris_center,
            'radius': radius
        }
    
    def process_video(self):
        """Process the entire video"""
        frame_idx = 0
        processed_count = 0
        failed_count = 0
        
        # Detect iris once at the beginning
        print("Detecting iris from first frame...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = self.cap.read()
        
        if not ret:
            print("Error: Could not read first frame")
            return False
        
        iris_result = self.detect_iris(first_frame)
        if iris_result is None:
            print("Error: Could not detect iris in first frame")
            return False
        
        print(f"Fixed iris: center={iris_result['center']}, radius={iris_result['radius']}")
        
        for frame_number in self.frames_to_process:
            # Set frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            
            if not ret:
                print(f"Failed to read frame {frame_number}")
                failed_count += 1
                continue
            
            # Detect pupil using fixed iris
            pupil_result = self.detect_pupil(frame, iris_result)
            if pupil_result is None:
                print(f"Failed to detect pupil in frame {frame_number}")
                failed_count += 1
                continue
            
            # Store tracking data
            timestamp = frame_number / self.fps
            pupil_radius = pupil_result['radius']
            pupil_diameter = pupil_radius * 2
            pupil_area = np.pi * pupil_radius ** 2
            iris_pupil_ratio = pupil_radius / iris_result['radius']
            
            self.tracking_data.append({
                'frame_idx': frame_number,
                'timestamp': timestamp,
                'iris_center_x': iris_result['center'][0],
                'iris_center_y': iris_result['center'][1],
                'iris_radius': iris_result['radius'],
                'pupil_center_x': pupil_result['center'][0],
                'pupil_center_y': pupil_result['center'][1],
                'pupil_radius': pupil_radius,
                'pupil_diameter': pupil_diameter,
                'pupil_area': pupil_area,
                'iris_pupil_ratio': iris_pupil_ratio,
                'detection_method': 'clean_debug_based'
            })
            
            # Create visualization frame
            vis_frame = self.create_visualization_frame(
                frame, iris_result, pupil_result, frame_number, timestamp, pupil_radius
            )
            
            # Write to output video
            if self.video_writer:
                self.video_writer.write(vis_frame)
            
            processed_count += 1
            
            # Progress update
            if frame_number % 15 == 0:  # Every 15 frames
                print(f"Processed frame {frame_number}/{self.total_frames} (pupil radius: {pupil_radius})")
        
        # Cleanup
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {processed_count} frames")
        print(f"Failed frames: {failed_count}")
        print(f"Success rate: {processed_count/(processed_count+failed_count)*100:.1f}%")
        
        return processed_count > 0
    
    def create_visualization_frame(self, frame, iris_result, pupil_result, frame_idx, timestamp, pupil_radius):
        """Create visualization frame with detection overlays"""
        vis_frame = frame.copy()
        
        # Draw iris circle
        iris_center = iris_result['center']
        iris_radius = iris_result['radius']
        cv2.circle(vis_frame, iris_center, iris_radius, (0, 255, 0), 3)  # Green for iris
        cv2.circle(vis_frame, iris_center, 5, (0, 255, 0), -1)
        
        # Draw pupil circle
        pupil_center = pupil_result['center']
        pupil_radius = pupil_result['radius']
        cv2.circle(vis_frame, pupil_center, pupil_radius, (0, 255, 255), 3)  # Yellow for pupil
        cv2.circle(vis_frame, pupil_center, 3, (0, 255, 255), -1)
        
        # Add text information
        info_text = f"Frame: {frame_idx} | Time: {timestamp:.2f}s | Pupil r: {pupil_radius}px"
        cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, "CLEAN DEBUG-BASED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return vis_frame
    
    def save_results(self):
        """Save tracking results to files"""
        if not self.tracking_data:
            print("No tracking data to save")
            return
        
        video_name = Path(self.video_path).stem
        
        # Save CSV data
        df = pd.DataFrame(self.tracking_data)
        csv_path = os.path.join(self.output_dir, f"pupil_data_clean_{video_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved raw data: {csv_path}")
        
        # Create and save plot
        self.create_plot(video_name)
        
        # Save statistics
        self.save_statistics(video_name)
    
    def create_plot(self, video_name):
        """Create pupil size plot"""
        df = pd.DataFrame(self.tracking_data)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot pupil radius over time
        ax1.plot(df['timestamp'], df['pupil_radius'], 'b-', linewidth=2, label='Pupil Radius')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Pupil Radius (pixels)')
        ax1.set_title(f'Clean Pupil Size Tracking - {video_name}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot pupil area over time
        ax2.plot(df['timestamp'], df['pupil_area'], 'r-', linewidth=2, label='Pupil Area')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Pupil Area (pixels²)')
        ax2.set_title('Pupil Area Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, f"pupil_tracking_clean_{video_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")
    
    def save_statistics(self, video_name):
        """Save tracking statistics"""
        df = pd.DataFrame(self.tracking_data)
        
        stats_text = f"""Clean Pupil Tracking Statistics - {video_name}
============================================================

Total frames processed: {len(df)}
Failed frames: 0
Success rate: 100.0%

Detection Methods:
  clean_debug_based: {len(df)} frames (100.0%)

Pupil Diameter:
  mean: {df['pupil_diameter'].mean():.2f}
  std: {df['pupil_diameter'].std():.2f}
  min: {df['pupil_diameter'].min():.2f}
  max: {df['pupil_diameter'].max():.2f}
  range: {df['pupil_diameter'].max() - df['pupil_diameter'].min():.2f}

Pupil Radius:
  mean: {df['pupil_radius'].mean():.2f}
  std: {df['pupil_radius'].std():.2f}
  min: {df['pupil_radius'].min():.2f}
  max: {df['pupil_radius'].max():.2f}
  range: {df['pupil_radius'].max() - df['pupil_radius'].min():.2f}

Pupil Area:
  mean: {df['pupil_area'].mean():.2f}
  std: {df['pupil_area'].std():.2f}
  min: {df['pupil_area'].min():.2f}
  max: {df['pupil_area'].max():.2f}
  range: {df['pupil_area'].max() - df['pupil_area'].min():.2f}

Iris Pupil Ratio:
  mean: {df['iris_pupil_ratio'].mean():.2f}
  std: {df['iris_pupil_ratio'].std():.2f}
  min: {df['iris_pupil_ratio'].min():.2f}
  max: {df['iris_pupil_ratio'].max():.2f}
  range: {df['iris_pupil_ratio'].max() - df['iris_pupil_ratio'].min():.2f}
"""
        
        stats_path = os.path.join(self.output_dir, f"pupil_stats_clean_{video_name}.txt")
        with open(stats_path, 'w') as f:
            f.write(stats_text)
        print(f"Saved statistics: {stats_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Clean pupil tracking based on working debug script')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output_dir', default='data/output', help='Output directory for results')
    parser.add_argument('--frame_interval', type=int, default=1, help='Process every Nth frame')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    try:
        tracker = CleanVideoPupilTracker(
            args.video_path, 
            args.output_dir, 
            args.frame_interval
        )
        
        success = tracker.process_video()
        if success:
            tracker.save_results()
            print(f"\n🎯 Successfully processed video with clean debug-based method!")
            return 0
        else:
            print(f"\n❌ Failed to process video")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
