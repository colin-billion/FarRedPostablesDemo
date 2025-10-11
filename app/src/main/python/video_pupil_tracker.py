#!/usr/bin/env python3
"""
Red-Channel Adaptive Pupil Tracker

Complete pupil detection using red-channel adaptive histogram-based thresholding.
No iris detection - direct pupil detection from red channel analysis.
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse

class CleanVideoPupilTracker:
    """Red-channel adaptive pupil tracker"""
    
    def __init__(self, video_path, output_dir="data/output", frame_interval=1, debug=False):
        """Initialize the red-channel pupil tracker"""
        self.video_path = video_path
        self.output_dir = output_dir
        self.frame_interval = frame_interval
        self.debug = debug
        
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
    
    def extract_red_channel(self, frame):
        """Extract red channel from frame"""
        if len(frame.shape) == 3:
            return frame[:, :, 2]  # Extract red channel (BGR format)
        else:
            return frame.copy()
    
    def calculate_adaptive_threshold(self, red_channel, debug=False):
        """Calculate adaptive threshold based on darkest pixel + tolerance"""
        # Find the darkest pixel value in the frame
        darkest_pixel = red_channel.min()
        
        # Adaptive tolerance: +30 for dark frames, +20 for bright frames
        if darkest_pixel > 150:
            tolerance = 20  # Use smaller tolerance for bright frames
        else:
            tolerance = 30  # Use larger tolerance for dark frames
        adaptive_threshold = darkest_pixel + tolerance
        
        # Calculate histogram for visualization
        hist = cv2.calcHist([red_channel], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        if debug:
            print(f"  Darkest pixel: {darkest_pixel}")
            print(f"  Tolerance: +{tolerance}")
            print(f"  Adaptive threshold: {adaptive_threshold}")
            print(f"  Red channel range: {red_channel.min()}-{red_channel.max()}")
            print(f"  Mean: {red_channel.mean():.1f}, Std: {red_channel.std():.1f}")
        
        return adaptive_threshold, hist
    
    def create_pupil_mask(self, red_channel, threshold):
        """Create binary mask of pupil pixels"""
        # Create binary mask where pixels darker than threshold are pupil pixels
        mask = red_channel < threshold
        
        # Convert to uint8
        mask = mask.astype(np.uint8) * 255
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def fit_circle_to_mask(self, mask, debug=False):
        """Fit best circle to pupil mask"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find the largest contour (should be the pupil)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if contour is large enough
        area = cv2.contourArea(largest_contour)
        if area < 100:  # Minimum area threshold
            return None
        
        # Fit circle to contour
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Validate circle size (expected radius 40-70 pixels)
        if radius < 20 or radius > 100:
            return None
        
        # Calculate circularity
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter == 0:
            return None
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Only accept reasonably circular shapes
        if circularity < 0.3:
            return None
        
        if debug:
            print(f"  Circle fit: center=({center[0]}, {center[1]}), radius={radius}")
            print(f"  Area={area}, circularity={circularity:.3f}")
        
        return {
            'center': center,
            'radius': radius,
            'area': area,
            'circularity': circularity
        }
    
    def detect_pupil_red_channel_adaptive(self, frame, debug=False, frame_number=0):
        """Detect pupil using red channel with adaptive histogram-based thresholding"""
        if debug:
            print(f"\n🔍 RED-CHANNEL ADAPTIVE PUPIL DETECTION (Frame {frame_number})")
        
        # Extract red channel
        red_channel = self.extract_red_channel(frame)
        
        if debug:
            print(f"  Red channel shape: {red_channel.shape}, range: {red_channel.min()}-{red_channel.max()}")
        
        # Calculate adaptive threshold using histogram analysis
        threshold, hist = self.calculate_adaptive_threshold(red_channel, debug)
        
        # Create pupil mask
        pupil_mask = self.create_pupil_mask(red_channel, threshold)
        
        # Fit circle to mask
        circle_result = self.fit_circle_to_mask(pupil_mask, debug)
        
        # Create debug visualizations
        if debug:
            self.create_red_channel_debug_visualization(frame, red_channel, hist, threshold, 
                                                       pupil_mask, circle_result, frame_number)
        
        return circle_result
    
    def create_red_channel_debug_visualization(self, original_frame, red_channel, hist, threshold, 
                                             pupil_mask, circle_result, frame_number):
        """Create comprehensive debug visualization for red-channel pupil detection"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Create title with detection status
        detection_status = "SUCCESS" if circle_result is not None else "FAILED"
        title = f'Red-Channel Adaptive Pupil Detection - Frame {frame_number} ({detection_status})'
        fig.suptitle(title, fontsize=16, color='green' if circle_result is not None else 'red')
        
        # Original frame
        axes[0, 0].imshow(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Frame')
        axes[0, 0].axis('off')
        
        # Red channel
        axes[0, 1].imshow(red_channel, cmap='gray')
        axes[0, 1].set_title('Red Channel')
        axes[0, 1].axis('off')
        
        # Histogram with threshold
        axes[0, 2].plot(hist)
        axes[0, 2].axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
        axes[0, 2].set_title('Intensity Histogram')
        axes[0, 2].set_xlabel('Intensity')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Add histogram statistics as text
        darkest_pixel = red_channel.min()
        tolerance = threshold - darkest_pixel
        stats_text = f"""Threshold Stats:
Range: {red_channel.min():.0f}-{red_channel.max():.0f}
Mean: {red_channel.mean():.1f} ± {red_channel.std():.1f}
Darkest: {darkest_pixel:.0f}
Tolerance: +{tolerance:.0f}
Threshold: {threshold:.0f}"""
        
        axes[0, 2].text(0.02, 0.98, stats_text, transform=axes[0, 2].transAxes, 
                        verticalalignment='top', fontsize=8, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Binary mask
        axes[1, 0].imshow(pupil_mask, cmap='gray')
        axes[1, 0].set_title('Pupil Binary Mask')
        axes[1, 0].axis('off')
        
        # Circle detection result
        result_image = original_frame.copy()
        if circle_result is not None:
            center = circle_result['center']
            radius = circle_result['radius']
            cv2.circle(result_image, center, radius, (0, 255, 0), 2)
            cv2.circle(result_image, center, 3, (0, 255, 0), -1)
            cv2.putText(result_image, f"R={radius}", (center[0]+10, center[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        axes[1, 1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Circle Detection Result')
        axes[1, 1].axis('off')
        
        # Combined visualization
        combined_image = original_frame.copy()
        if circle_result is not None:
            center = circle_result['center']
            radius = circle_result['radius']
            cv2.circle(combined_image, center, radius, (0, 255, 0), 2)
            cv2.circle(combined_image, center, 3, (0, 255, 0), -1)
        
        # Overlay mask
        mask_colored = cv2.applyColorMap(pupil_mask, cv2.COLORMAP_JET)
        combined_image = cv2.addWeighted(combined_image, 0.7, mask_colored, 0.3, 0)
        
        axes[1, 2].imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('Combined Visualization')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save debug image
        debug_path = os.path.join(self.output_dir, f"red_channel_debug_frame_{frame_number:04d}.png")
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved red-channel debug: {debug_path}")
    
    def process_video(self):
        """Process the entire video using red-channel adaptive pupil detection"""
        print("Processing video with red-channel adaptive pupil detection...")
        
        processed_count = 0
        failed_count = 0
        
        # Process each frame
        for frame_number in self.frames_to_process:
            # Set frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            
            if not ret:
                print(f"Failed to read frame {frame_number}")
                failed_count += 1
                continue
            
            # Enable debug every 20 frames and specifically for frame 70
            debug_enabled = self.debug and (frame_number % 20 == 0 or frame_number == 70)
            
            # Detect pupil using red-channel adaptive method
            pupil_result = self.detect_pupil_red_channel_adaptive(frame, debug=debug_enabled, frame_number=frame_number)
            
            # Prepare output frame
            output_frame = frame.copy()
            
            # Process successful detections
            if pupil_result is not None:
                # Store tracking data
                try:
                    timestamp = frame_number / self.fps
                    if not np.isfinite(timestamp):
                        timestamp = 0.0
                except ZeroDivisionError:
                    timestamp = 0.0
                
                pupil_radius = pupil_result['radius']
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
                    'detection_method': 'red_channel_adaptive',
                    'circularity': pupil_result['circularity']
                })
                
                processed_count += 1
                
                # Add circle overlay for successful detection
                center = pupil_result['center']
                radius = pupil_result['radius']
                cv2.circle(output_frame, center, radius, (0, 255, 0), 2)
                cv2.circle(output_frame, center, 3, (0, 255, 0), -1)
                cv2.putText(output_frame, f"Frame {frame_number}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(output_frame, f"Radius: {radius}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(output_frame, "PUPIL DETECTED", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if frame_number % 30 == 0:
                    print(f"Detected frame {frame_number}/{self.total_frames}")
            else:
                failed_count += 1
                # Add text for failed detection
                cv2.putText(output_frame, f"Frame {frame_number}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(output_frame, "NO PUPIL DETECTED", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Write frame to output video (always write, regardless of detection success)
            self.video_writer.write(output_frame)
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {processed_count} frames")
        print(f"Failed frames: {failed_count}")
        print(f"Success rate: {processed_count/(processed_count+failed_count)*100:.1f}%")
        
        return True
    
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
        
        # Release video writer
        self.video_writer.release()
        print("✅ Video writer released successfully")
    
    def create_plot(self, video_name):
        """Create pupil size plot"""
        df = pd.DataFrame(self.tracking_data)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot pupil radius over time
        ax1.plot(df['timestamp'], df['pupil_radius'], 'b-', linewidth=2, label='Pupil Radius')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Pupil Radius (pixels)')
        ax1.set_title(f'Red-Channel Adaptive Pupil Size Tracking - {video_name}')
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
        """Save basic statistics"""
        df = pd.DataFrame(self.tracking_data)
        
        stats_path = os.path.join(self.output_dir, f"pupil_stats_clean_{video_name}.txt")
        
        with open(stats_path, 'w') as f:
            f.write(f"RED-CHANNEL ADAPTIVE PUPIL TRACKING STATISTICS\n")
            f.write(f"Video: {video_name}\n")
            f.write(f"Total frames processed: {len(self.tracking_data)}\n")
            f.write(f"Success rate: {len(self.tracking_data)/self.total_frames*100:.1f}%\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("PUPIL RADIUS STATISTICS:\n")
            f.write(f"  Mean: {df['pupil_radius'].mean():.2f} pixels\n")
            f.write(f"  Std: {df['pupil_radius'].std():.2f} pixels\n")
            f.write(f"  Min: {df['pupil_radius'].min():.2f} pixels\n")
            f.write(f"  Max: {df['pupil_radius'].max():.2f} pixels\n\n")
            
            f.write("PUPIL AREA STATISTICS:\n")
            f.write(f"  Mean: {df['pupil_area'].mean():.2f} pixels²\n")
            f.write(f"  Std: {df['pupil_area'].std():.2f} pixels²\n")
            f.write(f"  Min: {df['pupil_area'].min():.2f} pixels²\n")
            f.write(f"  Max: {df['pupil_area'].max():.2f} pixels²\n\n")
            
            f.write("CIRCULARITY STATISTICS:\n")
            f.write(f"  Mean: {df['circularity'].mean():.3f}\n")
            f.write(f"  Std: {df['circularity'].std():.3f}\n")
            f.write(f"  Min: {df['circularity'].min():.3f}\n")
            f.write(f"  Max: {df['circularity'].max():.3f}\n")
        
        print(f"Saved statistics: {stats_path}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Red-channel adaptive pupil tracking')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output_dir', default='data/output', help='Output directory')
    parser.add_argument('--frame_interval', type=int, default=1, help='Process every Nth frame')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return 1
    
    try:
        # Initialize tracker
        tracker = CleanVideoPupilTracker(args.video_path, args.output_dir, args.frame_interval)
        
        # Process video
        success = tracker.process_video()
        
        if success:
            # Save results
            tracker.save_results()
            print(f"\n🎯 Red-channel adaptive pupil tracking completed successfully!")
            return 0
        else:
            print(f"\n❌ Pupil tracking failed!")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Tracking interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())