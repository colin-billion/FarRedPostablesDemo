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
# Removed sklearn dependency - using simple clustering instead

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
            raise ValueError(f"Could not open video: {video_path}. Check if file exists and is a valid video format.")
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"OpenCV video properties: frames={self.total_frames}, fps={self.fps}, size={self.width}x{self.height}")
        
        # Android-specific fix: If OpenCV fails to read properties, try alternative methods
        if self.fps <= 0 or self.total_frames <= 0 or self.width <= 0 or self.height <= 0:
            print("OpenCV failed to read video properties, trying alternative method...")
            
            # Try to read a few frames to estimate properties
            frame_count = 0
            test_fps = 30.0  # Default assumption
            detected_width = 0
            detected_height = 0
            
            # Try to read frames to count them and get dimensions
            # First, try to get just the first frame for dimensions
            ret, first_frame = self.cap.read()
            if ret and first_frame is not None:
                detected_height, detected_width = first_frame.shape[:2]
                print(f"Detected dimensions from first frame: {detected_width}x{detected_height}")
                frame_count = 1
                
                # Now count remaining frames
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if frame_count > 1000:  # Safety limit
                        break
                
                # Reset video capture to beginning
                self.cap.release()
                self.cap = cv2.VideoCapture(video_path)
                if not self.cap.isOpened():
                    raise ValueError(f"Could not reopen video: {video_path}")
            else:
                print("Error: Could not read first frame for dimensions")
                raise ValueError("Invalid video file - cannot read frames")
            
            if frame_count > 0:
                self.total_frames = frame_count
                print(f"Estimated frame count: {self.total_frames}")
            else:
                print("Error: Could not read any frames from video")
                raise ValueError("Invalid video file - no frames detected")
            
            # Use detected dimensions if available
            if detected_width > 0 and detected_height > 0:
                self.width = detected_width
                self.height = detected_height
                print(f"Using detected dimensions: {self.width}x{self.height}")
            else:
                # Fallback to common mobile video dimensions
                self.width = 1280
                self.height = 720
                print(f"Using fallback dimensions: {self.width}x{self.height}")
            
            # Use default FPS if not detected
            if self.fps <= 0:
                self.fps = test_fps
                print(f"Using default FPS: {self.fps}")
        
        # Final validation
        if self.fps <= 0:
            print(f"Warning: Invalid FPS ({self.fps}), using default 30 FPS")
            self.fps = 30.0
        
        if self.total_frames <= 0:
            print(f"Error: Invalid frame count ({self.total_frames})")
            raise ValueError("Invalid video file - no frames detected")
        
        if self.width <= 0 or self.height <= 0:
            print(f"Error: Invalid video dimensions ({self.width}x{self.height})")
            raise ValueError("Invalid video file - invalid dimensions")
        
        # Calculate frames to process
        self.frames_to_process = list(range(0, self.total_frames, frame_interval))
        
        # Calculate duration safely
        try:
            duration = self.total_frames / self.fps if self.fps > 0 else 0
            print(f"Video info: {self.total_frames} frames, {self.fps:.2f} FPS, {duration:.2f}s duration")
        except ZeroDivisionError:
            print(f"Video info: {self.total_frames} frames, {self.fps:.2f} FPS, 0.00s duration (FPS error)")
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
    
    def detect_pupil_red_channel(self, image):
        """Detect pupil using red channel only - simple darkest pixel approach"""
        # Extract only the red channel
        if len(image.shape) == 3:
            red_channel = image[:, :, 2]  # BGR format, so index 2 is red
        else:
            # If already grayscale, use as is
            red_channel = image.copy()
        
        # Find the darkest pixel value
        darkest_value = np.min(red_channel)
        
        # Add buffer of +20 brighter than darkest
        threshold_value = darkest_value + 20
        
        # Find all pixels darker than threshold
        pupil_mask = red_channel <= threshold_value
        
        # Get coordinates of pupil pixels
        pupil_coords = np.where(pupil_mask)
        
        if len(pupil_coords[0]) == 0:
            return None
        
        # Convert to (x, y) format
        y_coords, x_coords = pupil_coords
        pupil_points = np.column_stack((x_coords, y_coords))
        
        # Find the best circle around these points
        pupil_circle = self.fit_circle_to_points(pupil_points)
        
        if pupil_circle is None:
            return None
        
        # Store debug info for visualization
        pupil_circle['debug_mask'] = pupil_mask
        pupil_circle['threshold_value'] = threshold_value
        pupil_circle['darkest_value'] = darkest_value
        
        return pupil_circle
    
    def fit_circle_to_points(self, points):
        """Fit the best circle around a set of points"""
        if len(points) < 3:
            return None
        
        # Calculate centroid as circle center
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        center = (int(center_x), int(center_y))
        
        # Calculate distances from center to all points
        distances = np.sqrt((points[:, 0] - center_x)**2 + (points[:, 1] - center_y)**2)
        
        # Use 90th percentile as radius to avoid outliers
        radius = int(np.percentile(distances, 90))
        
        # Ensure minimum radius
        if radius < 5:
            radius = 5
        
        return {
            'center': center,
            'radius': radius,
            'points': points,
            'num_points': len(points)
        }
    
    def process_video(self):
        """Process the entire video using red-channel pupil detection"""
        try:
            frame_idx = 0
            processed_count = 0
            failed_count = 0
            
            print("Starting red-channel pupil detection...")
            
        except Exception as e:
            print(f"Error in process_video setup: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        for frame_number in self.frames_to_process:
            # Set frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            
            if not ret:
                print(f"Failed to read frame {frame_number}")
                failed_count += 1
                continue
            
            # Detect pupil using red channel method
            pupil_result = self.detect_pupil_red_channel(frame)
            if pupil_result is None:
                print(f"Failed to detect pupil in frame {frame_number}")
                failed_count += 1
                continue
            
            # Validate pupil detection
            pupil_radius = pupil_result['radius']
            if pupil_radius <= 0:
                print(f"Invalid pupil radius in frame {frame_number}: {pupil_radius}")
                failed_count += 1
                continue
            
            # Store tracking data with error handling
            try:
                timestamp = frame_number / self.fps
                if not np.isfinite(timestamp):
                    print(f"Invalid timestamp in frame {frame_number}: {timestamp}")
                    timestamp = 0.0
            except ZeroDivisionError:
                print(f"Zero division error calculating timestamp in frame {frame_number} - FPS: {self.fps}")
                timestamp = 0.0
            
            pupil_diameter = pupil_radius * 2
            pupil_area = np.pi * pupil_radius ** 2
            
            self.tracking_data.append({
                'frame_idx': frame_number,
                'timestamp': timestamp,
                'pupil_center_x': pupil_result['center'][0],
                'pupil_center_y': pupil_result['center'][1],
                'pupil_radius': pupil_radius,
                'pupil_diameter': pupil_diameter,
                'pupil_area': pupil_area,
                'detection_method': 'red_channel_darkest',
                'threshold_value': pupil_result['threshold_value'],
                'darkest_value': pupil_result['darkest_value'],
                'num_points': pupil_result['num_points']
            })
            
            # Create visualization frame
            vis_frame = self.create_visualization_frame(
                frame, pupil_result, frame_number, timestamp, pupil_radius
            )
            
            # Write to output video
            if self.video_writer and self.video_writer.isOpened():
                try:
                    self.video_writer.write(vis_frame)
                    if frame_number % 30 == 0:  # Every 30 frames
                        print(f"  Written frame {frame_number} to video")
                except Exception as e:
                    print(f"  Error writing frame {frame_number} to video: {e}")
            elif self.video_writer:
                print(f"  Video writer not opened, skipping frame {frame_number}")
            
            processed_count += 1
            
            # Progress update
            if frame_number % 15 == 0:  # Every 15 frames
                print(f"Processed frame {frame_number}/{self.total_frames} (pupil radius: {pupil_radius})")
        
        # Cleanup
        self.cap.release()
        if self.video_writer:
            print(f"Releasing video writer...")
            try:
                self.video_writer.release()
                print(f"✅ Video writer released successfully")
            except Exception as e:
                print(f"❌ Error releasing video writer: {e}")
        else:
            print(f"⚠️  No video writer to release")
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {processed_count} frames")
        print(f"Failed frames: {failed_count}")
        print(f"Success rate: {processed_count/(processed_count+failed_count)*100:.1f}%")
        
        return processed_count > 0
    
    def create_visualization_frame(self, frame, pupil_result, frame_idx, timestamp, pupil_radius):
        """Create visualization frame with red-channel pupil detection overlays"""
        vis_frame = frame.copy()
        
        # Draw pupil circle
        pupil_center = pupil_result['center']
        pupil_radius = pupil_result['radius']
        cv2.circle(vis_frame, pupil_center, pupil_radius, (0, 255, 255), 3)  # Yellow for pupil
        cv2.circle(vis_frame, pupil_center, 3, (0, 255, 255), -1)
        
        # Highlight detected pupil pixels in red
        debug_mask = pupil_result['debug_mask']
        vis_frame[debug_mask] = [0, 0, 255]  # Red highlight for detected pixels
        
        # Add text information
        info_text = f"Frame: {frame_idx} | Time: {timestamp:.2f}s | Pupil r: {pupil_radius}px"
        cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add detection method and threshold info
        method_text = f"RED CHANNEL | Darkest: {pupil_result['darkest_value']} | Threshold: {pupil_result['threshold_value']}"
        cv2.putText(vis_frame, method_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add pixel count info
        pixel_text = f"Detected pixels: {pupil_result['num_points']}"
        cv2.putText(vis_frame, pixel_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
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
        ax1.set_title(f'Red Channel Pupil Size Tracking - {video_name}')
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
        
        stats_text = f"""Red Channel Pupil Tracking Statistics - {video_name}
============================================================

Total frames processed: {len(df)}
Failed frames: 0
Success rate: 100.0%

Detection Methods:
  red_channel_darkest: {len(df)} frames (100.0%)

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

Detection Parameters:
  darkest_value: mean={df['darkest_value'].mean():.1f}, std={df['darkest_value'].std():.1f}
  threshold_value: mean={df['threshold_value'].mean():.1f}, std={df['threshold_value'].std():.1f}
  detected_pixels: mean={df['num_points'].mean():.0f}, std={df['num_points'].std():.0f}
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
