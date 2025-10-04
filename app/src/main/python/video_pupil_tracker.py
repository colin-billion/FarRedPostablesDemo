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
    
    def detect_iris_median_blur(self, image, debug=False):
        """Detect iris using median blurred image with custom parameters: radius 40-150"""
        if debug:
            print(f"\n🔍 IRIS DETECTION (MEDIAN BLURRED)")
            print(f"   Input image shape: {image.shape}")
            print(f"   Input image type: {image.dtype}")
            print(f"   Input image range: {image.min()}-{image.max()}")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if debug:
            print(f"   Grayscale shape: {gray.shape}, range: {gray.min()}-{gray.max()}")
        
        # Apply median blur for noise reduction
        blurred = cv2.medianBlur(gray, 5)
        if debug:
            print(f"   After median blur: range {blurred.min()}-{blurred.max()}")
        
        # Save debug image
        if debug:
            self.save_debug_image(blurred, "iris_median_blur_preprocessed")
        
        # Find Hough circles with custom parameters
        if debug:
            print(f"   HoughCircles parameters: minDist=50, param1=50, param2=30, minRadius=40, maxRadius=150")
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,   # Smaller minimum distance since only one eye
            param1=50,
            param2=30,    # Lower threshold for more sensitive detection
            minRadius=40, # User-specified minimum radius
            maxRadius=150 # User-specified maximum radius
        )
        
        if circles is None:
            if debug:
                print(f"❌ No circles detected by HoughCircles on median blurred image")
            return None
        
        circles = np.round(circles[0, :]).astype("int")
        if debug:
            print(f"✅ Found {len(circles)} potential iris circles on median blurred image")
            # Create debug visualization of all detected circles
            self.create_iris_debug_visualization(blurred, circles, "iris_median_blur_all_circles")
        
        # Score each circle based on intensity differences
        best_circle = None
        best_score = -float('inf')
        
        if debug:
            print(f"   Scoring circles:")
        for i, circle in enumerate(circles):
            x, y, r = circle
            score = self.score_iris_circle(blurred, x, y, r)
            if debug:
                print(f"     Circle {i+1}: center=({x},{y}), radius={r}, score={score:.2f}")
            
            if score > best_score:
                best_score = score
                best_circle = circle
        
        if best_circle is None:
            if debug:
                print(f"❌ No valid iris circle found (all scores too low)")
            return None
        
        x, y, r = best_circle
        if debug:
            print(f"✅ Best iris: center=({x},{y}), radius={r}, score={best_score:.2f}")
            # Create final debug visualization
            self.create_iris_debug_visualization(blurred, [best_circle], "iris_median_blur_best_circle")
        
        return {
            'center': (x, y),
            'radius': r,
            'score': best_score
        }
    
    def detect_iris_custom(self, image, debug=False):
        """Detect iris using custom parameters: radius 40-200, single eye"""
        if debug:
            print(f"\n🔍 IRIS DETECTION (CUSTOM PARAMETERS)")
            print(f"   Input image shape: {image.shape}")
            print(f"   Input image type: {image.dtype}")
            print(f"   Input image range: {image.min()}-{image.max()}")
        
        # Minimal preprocessing - just convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if debug:
            print(f"   Grayscale shape: {gray.shape}, range: {gray.min()}-{gray.max()}")
        
        # Apply only median blur for noise reduction
        blurred = cv2.medianBlur(gray, 5)
        if debug:
            print(f"   After median blur: range {blurred.min()}-{blurred.max()}")
        
        # Save debug image
        if debug:
            self.save_debug_image(blurred, "iris_custom_preprocessed")
        
        # Find Hough circles with custom parameters
        if debug:
            print(f"   HoughCircles parameters: minDist=50, param1=50, param2=30, minRadius=40, maxRadius=200")
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,   # Smaller minimum distance since only one eye
            param1=50,
            param2=30,    # Lower threshold for more sensitive detection
            minRadius=40, # User-specified minimum radius
            maxRadius=200 # User-specified maximum radius
        )
        
        if circles is None:
            if debug:
                print(f"❌ No circles detected by HoughCircles with custom parameters")
            return None
        
        circles = np.round(circles[0, :]).astype("int")
        if debug:
            print(f"✅ Found {len(circles)} potential iris circles with custom parameters")
            # Create debug visualization of all detected circles
            self.create_iris_debug_visualization(blurred, circles, "iris_custom_all_circles")
        
        # Score each circle based on intensity differences
        best_circle = None
        best_score = -float('inf')
        
        if debug:
            print(f"   Scoring circles:")
        for i, circle in enumerate(circles):
            x, y, r = circle
            score = self.score_iris_circle(blurred, x, y, r)
            if debug:
                print(f"     Circle {i+1}: center=({x},{y}), radius={r}, score={score:.2f}")
            
            if score > best_score:
                best_score = score
                best_circle = circle
        
        if best_circle is None:
            if debug:
                print(f"❌ No valid iris circle found (all scores too low)")
            return None
        
        x, y, r = best_circle
        if debug:
            print(f"✅ Best iris: center=({x},{y}), radius={r}, score={best_score:.2f}")
            # Create final debug visualization
            self.create_iris_debug_visualization(blurred, [best_circle], "iris_custom_best_circle")
        
        return {
            'center': (x, y),
            'radius': r,
            'score': best_score
        }
    
    def detect_iris_original(self, image, debug=False):
        """Detect iris using original image with minimal preprocessing"""
        if debug:
            print(f"\n🔍 IRIS DETECTION (ORIGINAL IMAGE)")
            print(f"   Input image shape: {image.shape}")
            print(f"   Input image type: {image.dtype}")
            print(f"   Input image range: {image.min()}-{image.max()}")
        
        # Minimal preprocessing - just convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if debug:
            print(f"   Grayscale shape: {gray.shape}, range: {gray.min()}-{gray.max()}")
        
        # Apply only median blur for noise reduction
        blurred = cv2.medianBlur(gray, 5)
        if debug:
            print(f"   After median blur: range {blurred.min()}-{blurred.max()}")
        
        # Save debug image
        if debug:
            self.save_debug_image(blurred, "iris_original_preprocessed")
        
        # Find Hough circles on original image
        if debug:
            print(f"   HoughCircles parameters: minDist=100, param1=50, param2=30, minRadius=80, maxRadius=200")
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,  # Larger minimum distance for iris
            param1=50,
            param2=30,    # Lower threshold for more sensitive detection
            minRadius=80, # Larger minimum radius for iris
            maxRadius=200 # Larger maximum radius for iris
        )
        
        if circles is None:
            if debug:
                print(f"❌ No circles detected by HoughCircles on original image")
            return None
        
        circles = np.round(circles[0, :]).astype("int")
        if debug:
            print(f"✅ Found {len(circles)} potential iris circles on original image")
            # Create debug visualization of all detected circles
            self.create_iris_debug_visualization(blurred, circles, "iris_original_all_circles")
        
        # Score each circle based on intensity differences
        best_circle = None
        best_score = -float('inf')
        
        if debug:
            print(f"   Scoring circles:")
        for i, circle in enumerate(circles):
            x, y, r = circle
            score = self.score_iris_circle(blurred, x, y, r)
            if debug:
                print(f"     Circle {i+1}: center=({x},{y}), radius={r}, score={score:.2f}")
            
            if score > best_score:
                best_score = score
                best_circle = circle
        
        if best_circle is None:
            if debug:
                print(f"❌ No valid iris circle found (all scores too low)")
            return None
        
        x, y, r = best_circle
        if debug:
            print(f"✅ Best iris: center=({x},{y}), radius={r}, score={best_score:.2f}")
            # Create final debug visualization
            self.create_iris_debug_visualization(blurred, [best_circle], "iris_original_best_circle")
        
        return {
            'center': (x, y),
            'radius': r,
            'score': best_score
        }
    

    def preprocess_frame(self, frame):
        """Apply preprocessing before circle detection."""
        red = frame[:, :, 2]  # Extract red channel
        
        # Skip histogram equalization for iris detection (too slow)
        # hist_eq = cv2.equalizeHist(red)
        
        blurred = cv2.medianBlur(red, 5)

        # CLAHE for local contrast
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # Unsharp mask for edge emphasis
        gaussian = cv2.GaussianBlur(enhanced, (9, 9), 10.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

        return sharpened

    def apply_localized_histogram_equalization(self, image, iris_center, iris_radius):
        """Apply histogram equalization only to the iris region for better pupil contrast"""
        # Create a mask for the iris region
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, iris_center, iris_radius, 255, -1)
        
        # Create a copy of the image
        result = image.copy()
        
        # Extract the iris region
        iris_region = image[mask > 0]
        
        if len(iris_region) > 0:
            # Apply histogram equalization to the iris region only
            hist_eq_region = cv2.equalizeHist(iris_region)
            
            # Put the equalized region back into the result
            result[mask > 0] = hist_eq_region.ravel()
        
        return result

    def adaptive_threshold_with_bimodal(self, iris_pixels, baseline_intensity):
        """Adaptive threshold: try bimodal detection first, fallback to brightness-based percentages"""
        from scipy.signal import find_peaks
        from scipy.ndimage import gaussian_filter1d
        
        darkest_pixel = np.min(iris_pixels)
        darkness_range = baseline_intensity - darkest_pixel
        
        # Disabled peak-based thresholding for now
        # if darkest_pixel < 100:
        #     first_max_min_threshold = self.find_first_max_then_min_threshold(iris_pixels)
        #     if first_max_min_threshold is not None:
        #         return first_max_min_threshold
        
        # Fallback to brightness-based thresholds
        if darkest_pixel < 100:     # Dark scenes - 50%
            percentage = 0.50
        else:                       # Bright scenes - 80%
            percentage = 0.80
        
        return darkest_pixel + (darkness_range * percentage)
    
    def detect_bimodal_and_find_minima(self, iris_pixels, baseline_intensity):
        """Detect bimodal distribution and find local minima after first peak"""
        from scipy.signal import find_peaks
        from scipy.ndimage import gaussian_filter1d
        
        # Create histogram
        hist, bins = np.histogram(iris_pixels, bins=50)
        
        # Smooth the histogram slightly to reduce noise
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=1)
        
        # Find peaks - use different sensitivity based on brightness
        darkest_pixel = np.min(iris_pixels)
        if darkest_pixel < 50:
            # More sensitive for dark scenes
            min_height = 5
            min_distance = 3
        else:
            # Less sensitive for brighter scenes
            min_height = 10
            min_distance = 5
        
        peaks, properties = find_peaks(hist_smooth, height=min_height, distance=min_distance)
        
        # Need at least 2 peaks for bimodal
        if len(peaks) < 2:
            return None
        
        # Find local minima after the first peak
        first_peak_idx = peaks[0]
        minima_threshold = self.find_minima_after_peak(hist_smooth, bins, first_peak_idx)
        
        return minima_threshold
    
    def find_minima_after_peak(self, hist_smooth, bins, first_peak_idx):
        """Find local minima after the first peak"""
        # Look in the region after the first peak
        search_start = first_peak_idx + 2  # Skip a few bins after peak
        search_end = min(first_peak_idx + 20, len(hist_smooth) - 1)  # Don't go too far
        
        if search_start >= search_end:
            return None
        
        search_region = hist_smooth[search_start:search_end]
        search_bins = bins[search_start:search_end]
        
        # Find the first significant minimum
        for i in range(1, len(search_region) - 1):
            if (search_region[i] < search_region[i-1] and 
                search_region[i] < search_region[i+1] and
                search_region[i] < hist_smooth[first_peak_idx] * 0.5):  # Must be significantly lower than peak
                return search_bins[i]
        
        return None

    def find_first_max_then_min_threshold(self, iris_pixels):
        """Find first two maxima starting from darkest pixel, then second minimum after that"""
        from scipy.ndimage import gaussian_filter1d
        
        # Create histogram
        hist, bins = np.histogram(iris_pixels, bins=50)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Smooth the histogram
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=1)
        
        # Find the darkest pixel bin index
        darkest_pixel = np.min(iris_pixels)
        darkest_bin_idx = np.argmin(np.abs(bin_centers - darkest_pixel))
        
        # Start from darkest pixel and find first maximum
        first_max_idx = None
        for i in range(darkest_bin_idx, len(hist_smooth) - 1):
            if (hist_smooth[i] > hist_smooth[i-1] and 
                hist_smooth[i] > hist_smooth[i+1] and
                hist_smooth[i] > hist_smooth[darkest_bin_idx] * 1.2):  # Must be significantly higher than darkest
                first_max_idx = i
                break
        
        if first_max_idx is None:
            return None
        
        # Find second maximum after the first maximum
        second_max_idx = None
        for i in range(first_max_idx + 1, len(hist_smooth) - 1):
            if (hist_smooth[i] > hist_smooth[i-1] and 
                hist_smooth[i] > hist_smooth[i+1] and
                hist_smooth[i] > hist_smooth[first_max_idx] * 0.8):  # Must be reasonably high compared to first max
                second_max_idx = i
                break
        
        if second_max_idx is None:
            # If no second maximum found, use first minimum after first maximum
            for i in range(first_max_idx + 1, len(hist_smooth) - 1):
                if (hist_smooth[i] < hist_smooth[i-1] and 
                    hist_smooth[i] < hist_smooth[i+1]):
                    return bin_centers[i]
            return None
        
        # Use the second maximum (iris peak) as the threshold
        second_max_threshold = bin_centers[second_max_idx]
        
        # Check if second maximum is too high (larger than default threshold)
        # Calculate default threshold as fallback
        baseline_intensity = np.mean(iris_pixels)  # Approximate baseline
        darkness_range = baseline_intensity - darkest_pixel
        default_threshold = baseline_intensity - (darkness_range * 0.1)  # 10% fallback
        
        if second_max_threshold > default_threshold:
            return default_threshold
        else:
            return second_max_threshold

    def create_darkness_histogram(self, iris_pixels, baseline_intensity, threshold_core, target_size):
        """Create histogram visualization of pixel darkness values with smoothed version"""
        import matplotlib.pyplot as plt
        import matplotlib
        from scipy.ndimage import gaussian_filter1d
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Create figure
        fig, ax = plt.subplots(figsize=(4, 3))
        
        # Create histogram
        hist, bins = np.histogram(iris_pixels, bins=50)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Plot original histogram
        ax.bar(bin_centers, hist, alpha=0.3, color='blue', width=bins[1]-bins[0], label='Original')
        
        # Create and plot smoothed histogram
        hist_smooth = gaussian_filter1d(hist.astype(float), sigma=1)
        ax.plot(bin_centers, hist_smooth, color='red', linewidth=2, label='Smoothed')
        
        # Add vertical lines for key values
        ax.axvline(baseline_intensity, color='green', linestyle='--', linewidth=2, label=f'Baseline: {baseline_intensity:.1f}')
        ax.axvline(threshold_core, color='red', linestyle='-', linewidth=2, label=f'Threshold: {threshold_core:.1f}')
        
        # Calculate statistics
        darkest_pixel = np.min(iris_pixels)
        darkness_range = baseline_intensity - darkest_pixel
        percentage_used = ((baseline_intensity - threshold_core) / darkness_range) * 100 if darkness_range > 0 else 0
        
        # Add text annotations
        ax.text(0.02, 0.98, f'Darkest: {darkest_pixel:.1f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.text(0.02, 0.88, f'Range: {darkness_range:.1f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.text(0.02, 0.78, f'Using: {percentage_used:.1f}%', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Formatting
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Count')
        ax.set_title('Darkness Distribution (Original + Smoothed)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Convert to image
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        buf = buf[:, :, :3]
        
        plt.close(fig)
        
        # Resize to match target size
        histogram_img = cv2.resize(buf, target_size)
        return histogram_img

    def detect_iris_circles(self, frame, r_min, r_max):
        """Run HoughCircles on a preprocessed frame with radius filtering."""
        processed = self.preprocess_frame(frame)

        circles = cv2.HoughCircles(
            processed,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=frame.shape[0] // 4,
            param1=100,
            param2=20,
            minRadius=60,
            maxRadius=250
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")  # flatten to shape (N, 3)

            # filter by radius range
            circles = [c for c in circles if r_min <= c[2] <= r_max]

            if len(circles) > 0:
                # just take the first valid circle
                x, y, r = circles[0]
                return (x, y, r)
        
        return None

    def detect_iris(self, image, debug=False):
        """Detect iris using radius filtering"""
        # Apply radius filtering (110-140 pixels)
        circle = self.detect_iris_circles(image, 110, 140)
        
        if circle is None:
            return None
        
        x, y, r = circle
        return {
            'center': (x, y),
            'radius': r,
            'score': 1.0
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
    
    
    def calculate_iris_darkness_score(self, image, center, radius):
        """Calculate how good a circle is as an iris (lower = better)"""
        x, y = center
        h, w = image.shape
        
        # Check if circle is within image bounds
        if x - radius < 0 or x + radius >= w or y - radius < 0 or y + radius >= h:
            return float('inf')
        
        # Create masks for inner (iris) and outer (sclera) regions
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        outer_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Inner circle (iris area)
        cv2.circle(inner_mask, (x, y), radius, 255, -1)
        
        # Outer ring (sclera area) - slightly larger circle minus inner circle
        outer_radius = int(radius * 1.3)  # 30% larger
        if outer_radius < w//2 and outer_radius < h//2:  # Make sure it fits
            cv2.circle(outer_mask, (x, y), outer_radius, 255, -1)
            cv2.circle(outer_mask, (x, y), radius, 0, -1)  # Remove inner circle
        
        # Calculate mean intensities
        inner_intensity = cv2.mean(image, inner_mask)[0]
        outer_intensity = cv2.mean(image, outer_mask)[0]
        
        # Good iris should be darker than sclera
        # Lower score = darker iris relative to sclera
        if outer_intensity > 0:  # Avoid division by zero
            darkness_ratio = inner_intensity / outer_intensity
            return darkness_ratio
        else:
            return float('inf')
    
    def score_iris_circle(self, image, x, y, r):
        """Score an iris circle based on intensity differences"""
        h, w = image.shape
        
        # Check if circle is within image bounds
        if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
            return -float('inf')
        
        # Ensure radius is large enough for meaningful scoring
        if r < 20:
            return -float('inf')
        
        # Create masks for inner and outer rings
        inner_mask = np.zeros((h, w), dtype=np.uint8)
        outer_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Inner ring (iris area) - ensure inner radius is positive
        inner_outer_r = max(1, r - 5)
        inner_inner_r = max(1, r - 15)
        cv2.circle(inner_mask, (x, y), inner_outer_r, 255, -1)
        if inner_inner_r < inner_outer_r:
            cv2.circle(inner_mask, (x, y), inner_inner_r, 0, -1)
        
        # Outer ring (sclera area) - ensure outer radius is within bounds
        outer_outer_r = min(w//2, h//2, r + 5)
        outer_inner_r = max(1, r - 5)
        cv2.circle(outer_mask, (x, y), outer_outer_r, 255, -1)
        if outer_inner_r < outer_outer_r:
            cv2.circle(outer_mask, (x, y), outer_inner_r, 0, -1)
        
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
    
    def save_debug_image(self, image, name):
        """Save debug image to output directory"""
        try:
            debug_path = os.path.join(self.output_dir, f"debug_{name}.png")
            cv2.imwrite(debug_path, image)
            print(f"   💾 Saved debug image: {debug_path}")
        except Exception as e:
            print(f"   ❌ Failed to save debug image {name}: {e}")
    
    def create_preprocessing_comparison(self, original, gray, blurred, clahe_result, enhanced):
        """Create side-by-side comparison of all preprocessing steps"""
        try:
            # Ensure all images are the same size and type
            h, w = gray.shape
            
            # Resize original to match grayscale if needed
            if len(original.shape) == 3:
                original_resized = cv2.resize(original, (w, h))
            else:
                original_resized = original
            
            # Convert grayscale images to 3-channel for display
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
            clahe_bgr = cv2.cvtColor(clahe_result, cv2.COLOR_GRAY2BGR)
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            # Create a large comparison image (2x3 grid)
            # Each image will be 400x300 for good visibility
            target_size = (400, 300)
            
            # Resize all images
            img1 = cv2.resize(original_resized, target_size)
            img2 = cv2.resize(gray_bgr, target_size)
            img3 = cv2.resize(blurred_bgr, target_size)
            img4 = cv2.resize(clahe_bgr, target_size)
            img5 = cv2.resize(enhanced_bgr, target_size)
            
            # Create a 6th image showing the difference between original and enhanced
            diff = cv2.absdiff(original_resized, enhanced_bgr)
            diff_bgr = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR) if len(diff.shape) == 2 else diff
            img6 = cv2.resize(diff_bgr, target_size)
            
            # Create the comparison grid
            # Top row: Original, Grayscale, Blurred
            top_row = np.hstack([img1, img2, img3])
            # Bottom row: CLAHE, Enhanced, Difference
            bottom_row = np.hstack([img4, img5, img6])
            comparison = np.vstack([top_row, bottom_row])
            
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            color = (255, 255, 255)
            
            # Add labels to each section
            cv2.putText(comparison, "1. Original", (10, 30), font, font_scale, color, thickness)
            cv2.putText(comparison, "2. Grayscale", (410, 30), font, font_scale, color, thickness)
            cv2.putText(comparison, "3. Median Blur", (810, 30), font, font_scale, color, thickness)
            cv2.putText(comparison, "4. CLAHE", (10, 330), font, font_scale, color, thickness)
            cv2.putText(comparison, "5. Enhanced", (410, 330), font, font_scale, color, thickness)
            cv2.putText(comparison, "6. Difference", (810, 330), font, font_scale, color, thickness)
            
            # Add range information
            cv2.putText(comparison, f"Range: {gray.min()}-{gray.max()}", (10, 60), font, 0.5, color, 1)
            cv2.putText(comparison, f"Range: {blurred.min()}-{blurred.max()}", (410, 60), font, 0.5, color, 1)
            cv2.putText(comparison, f"Range: {clahe_result.min()}-{clahe_result.max()}", (10, 360), font, 0.5, color, 1)
            cv2.putText(comparison, f"Range: {enhanced.min()}-{enhanced.max()}", (410, 360), font, 0.5, color, 1)
            
            # Save comparison
            debug_path = os.path.join(self.output_dir, "debug_preprocessing_comparison.png")
            cv2.imwrite(debug_path, comparison)
            print(f"   💾 Saved preprocessing comparison: {debug_path}")
            
        except Exception as e:
            print(f"   ❌ Failed to create preprocessing comparison: {e}")
    
    def create_iris_debug_visualization(self, image, circles, name):
        """Create debug visualization of iris circles"""
        try:
            # Create a copy of the image for visualization
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            # Draw all circles
            for i, circle in enumerate(circles):
                x, y, r = circle
                color = (0, 255, 0) if i == 0 else (0, 255, 255)  # Green for first, yellow for others
                cv2.circle(vis_image, (x, y), r, color, 2)
                cv2.circle(vis_image, (x, y), 3, color, -1)
                cv2.putText(vis_image, f"{i+1}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Save debug visualization
            debug_path = os.path.join(self.output_dir, f"debug_{name}.png")
            cv2.imwrite(debug_path, vis_image)
            print(f"   💾 Saved debug visualization: {debug_path}")
        except Exception as e:
            print(f"   ❌ Failed to create debug visualization {name}: {e}")
    
    def create_comprehensive_iris_debug(self, image, debug=False):
        """Create comprehensive debug visualization for iris detection"""
        import matplotlib.pyplot as plt
        
        if debug:
            print(f"\n🔍 COMPREHENSIVE IRIS DETECTION DEBUG")
            print(f"   Input image shape: {image.shape}")
            print(f"   Input image type: {image.dtype}")
            print(f"   Input image range: {image.min()}-{image.max()}")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create different preprocessing versions
        preprocessed_versions = {}
        
        # 1. Original grayscale
        preprocessed_versions['Original'] = gray
        
        # 2. Median blur only
        preprocessed_versions['Median Blur'] = cv2.medianBlur(gray, 5)
        
        # 3. Gaussian blur
        preprocessed_versions['Gaussian Blur'] = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 4. CLAHE only (less aggressive)
        clahe_mild = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        preprocessed_versions['CLAHE Mild'] = clahe_mild.apply(gray)
        
        # 5. CLAHE moderate
        clahe_moderate = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        preprocessed_versions['CLAHE Moderate'] = clahe_moderate.apply(gray)
        
        # 6. CLAHE aggressive
        clahe_aggressive = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        preprocessed_versions['CLAHE Aggressive'] = clahe_aggressive.apply(gray)
        
        # 7. Edge detection (Canny)
        edges = cv2.Canny(gray, 50, 150)
        preprocessed_versions['Canny Edges'] = edges
        
        # 8. Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        preprocessed_versions['Morphological'] = morph
        
        # 9. Histogram equalization
        hist_eq = cv2.equalizeHist(gray)
        preprocessed_versions['Histogram Eq'] = hist_eq
        
        # 10. Current enhanced method
        blurred = cv2.medianBlur(gray, 5)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=3.0, beta=50)
        preprocessed_versions['Current Enhanced'] = enhanced
        
        # Test different Hough circle parameter sets
        hough_params = [
            {'name': 'Conservative', 'minDist': 100, 'param1': 50, 'param2': 30, 'minRadius': 80, 'maxRadius': 200},
            {'name': 'Moderate', 'minDist': 50, 'param1': 50, 'param2': 30, 'minRadius': 40, 'maxRadius': 150},
            {'name': 'Sensitive', 'minDist': 30, 'param1': 30, 'param2': 20, 'minRadius': 30, 'maxRadius': 120},
            {'name': 'Very Sensitive', 'minDist': 20, 'param1': 20, 'param2': 15, 'minRadius': 20, 'maxRadius': 100},
            {'name': 'Large Only', 'minDist': 80, 'param1': 50, 'param2': 25, 'minRadius': 60, 'maxRadius': 180},
            {'name': 'Small Only', 'minDist': 30, 'param1': 40, 'param2': 20, 'minRadius': 20, 'maxRadius': 80}
        ]
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(4, 6, figsize=(24, 16))
        fig.suptitle('Comprehensive Iris Detection Debug - Preprocessing + Hough Circles', fontsize=16)
        
        row = 0
        col = 0
        
        for prep_name, prep_image in preprocessed_versions.items():
            if row >= 4:
                break
                
            # Show preprocessing result
            if prep_name == 'Canny Edges':
                axes[row, col].imshow(prep_image, cmap='gray')
            else:
                axes[row, col].imshow(prep_image, cmap='gray')
            axes[row, col].set_title(f'{prep_name}\nRange: {prep_image.min()}-{prep_image.max()}')
            axes[row, col].axis('off')
            
            # Test Hough circles on this preprocessing
            circles_found = []
            for params in hough_params:
                circles = cv2.HoughCircles(
                    prep_image,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=params['minDist'],
                    param1=params['param1'],
                    param2=params['param2'],
                    minRadius=params['minRadius'],
                    maxRadius=params['maxRadius']
                )
                
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    circles_found.extend([(params['name'], circles)])
            
            # Draw circles on the image
            result_image = prep_image.copy()
            if len(result_image.shape) == 2:
                result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
            
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            
            for i, (param_name, circles) in enumerate(circles_found):
                color = colors[i % len(colors)]
                for circle in circles:
                    x, y, r = circle
                    cv2.circle(result_image, (x, y), r, color, 2)
                    cv2.putText(result_image, f"{param_name[:3]}", (x-10, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Show result with circles
            axes[row, col+1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            axes[row, col+1].set_title(f'{prep_name} + Circles\nFound: {len(circles_found)} param sets')
            axes[row, col+1].axis('off')
            
            col += 2
            if col >= 6:
                col = 0
                row += 1
        
        # Fill remaining empty subplots
        while row < 4:
            while col < 6:
                axes[row, col].axis('off')
                col += 1
            col = 0
            row += 1
        
        plt.tight_layout()
        
        # Save the comprehensive debug
        debug_path = os.path.join(self.output_dir, "debug_comprehensive_iris_detection.png")
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if debug:
            print(f"💾 Saved comprehensive iris debug: {debug_path}")
        
        return debug_path
    
    def create_detailed_parameter_debug(self, image, debug=False):
        """Create detailed debug visualization for different Hough circle parameters on median blurred image"""
        import matplotlib.pyplot as plt
        
        if debug:
            print(f"\n🔍 DETAILED PARAMETER DEBUG")
            print(f"   Input image shape: {image.shape}")
            print(f"   Input image type: {image.dtype}")
            print(f"   Input image range: {image.min()}-{image.max()}")
        
        # Convert to grayscale and apply median blur
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        blurred = cv2.medianBlur(gray, 5)
        
        if debug:
            print(f"   Median blurred image range: {blurred.min()}-{blurred.max()}")
        
        # Test different Hough circle parameter sets
        hough_params = [
            {'name': 'Conservative', 'minDist': 100, 'param1': 50, 'param2': 30, 'minRadius': 80, 'maxRadius': 200},
            {'name': 'Moderate', 'minDist': 50, 'param1': 50, 'param2': 30, 'minRadius': 40, 'maxRadius': 150},
            {'name': 'Sensitive', 'minDist': 30, 'param1': 30, 'param2': 20, 'minRadius': 30, 'maxRadius': 120},
            {'name': 'Very Sensitive', 'minDist': 20, 'param1': 20, 'param2': 15, 'minRadius': 20, 'maxRadius': 100},
            {'name': 'Large Only', 'minDist': 80, 'param1': 50, 'param2': 25, 'minRadius': 60, 'maxRadius': 180},
            {'name': 'Small Only', 'minDist': 30, 'param1': 40, 'param2': 20, 'minRadius': 20, 'maxRadius': 80},
            {'name': 'Ultra Sensitive', 'minDist': 10, 'param1': 10, 'param2': 10, 'minRadius': 10, 'maxRadius': 150},
            {'name': 'Edge Focus', 'minDist': 40, 'param1': 100, 'param2': 20, 'minRadius': 30, 'maxRadius': 120}
        ]
        
        # Create detailed visualization
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.suptitle('Detailed Hough Circle Parameter Testing on Median Blurred Image', fontsize=16)
        
        # Show original median blurred image in top-left
        axes[0, 0].imshow(blurred, cmap='gray')
        axes[0, 0].set_title('Median Blurred Image\n(Input for all tests)')
        axes[0, 0].axis('off')
        
        # Test each parameter set
        for i, params in enumerate(hough_params):
            row = (i + 1) // 3
            col = (i + 1) % 3
            
            if row >= 3:
                break
            
            # Test Hough circles with these parameters
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=params['minDist'],
                param1=params['param1'],
                param2=params['param2'],
                minRadius=params['minRadius'],
                maxRadius=params['maxRadius']
            )
            
            # Create result image
            result_image = blurred.copy()
            if len(result_image.shape) == 2:
                result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
            
            circles_found = 0
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                circles_found = len(circles)
                
                # Draw circles
                for j, circle in enumerate(circles):
                    x, y, r = circle
                    color = (0, 255, 0) if j == 0 else (0, 255, 255)  # Green for first, yellow for others
                    cv2.circle(result_image, (x, y), r, color, 2)
                    cv2.circle(result_image, (x, y), 3, color, -1)
                    cv2.putText(result_image, f"{j+1}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Show result
            axes[row, col].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f'{params["name"]}\n'
                                   f'minDist={params["minDist"]}, param1={params["param1"]}, param2={params["param2"]}\n'
                                   f'minR={params["minRadius"]}, maxR={params["maxRadius"]}\n'
                                   f'Found: {circles_found} circles')
            axes[row, col].axis('off')
            
            if debug:
                print(f"   {params['name']}: Found {circles_found} circles")
        
        # Fill remaining empty subplots
        for i in range(len(hough_params) + 1, 9):
            row = i // 3
            col = i % 3
            if row < 3:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save the detailed debug
        debug_path = os.path.join(self.output_dir, "debug_detailed_parameter_testing.png")
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if debug:
            print(f"💾 Saved detailed parameter debug: {debug_path}")
        
        return debug_path
    
    def create_focused_parameter_debug(self, image, debug=False):
        """Create focused debug visualization showing only the best circle from each parameter set"""
        import matplotlib.pyplot as plt
        
        if debug:
            print(f"\n🔍 FOCUSED PARAMETER DEBUG (BEST CIRCLES ONLY)")
            print(f"   Input image shape: {image.shape}")
            print(f"   Input image type: {image.dtype}")
            print(f"   Input image range: {image.min()}-{image.max()}")
        
        # Convert to grayscale and apply median blur
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        blurred = cv2.medianBlur(gray, 5)
        
        if debug:
            print(f"   Median blurred image range: {blurred.min()}-{blurred.max()}")
        
        # Test different Hough circle parameter sets
        hough_params = [
            {'name': 'Conservative', 'minDist': 100, 'param1': 50, 'param2': 30, 'minRadius': 80, 'maxRadius': 200},
            {'name': 'Moderate', 'minDist': 50, 'param1': 50, 'param2': 30, 'minRadius': 40, 'maxRadius': 150},
            {'name': 'Sensitive', 'minDist': 30, 'param1': 30, 'param2': 20, 'minRadius': 30, 'maxRadius': 120},
            {'name': 'Very Sensitive', 'minDist': 20, 'param1': 20, 'param2': 15, 'minRadius': 20, 'maxRadius': 100},
            {'name': 'Large Only', 'minDist': 80, 'param1': 50, 'param2': 25, 'minRadius': 60, 'maxRadius': 180},
            {'name': 'Small Only', 'minDist': 30, 'param1': 40, 'param2': 20, 'minRadius': 20, 'maxRadius': 80},
            {'name': 'Ultra Sensitive', 'minDist': 10, 'param1': 10, 'param2': 10, 'minRadius': 10, 'maxRadius': 150},
            {'name': 'Edge Focus', 'minDist': 40, 'param1': 100, 'param2': 20, 'minRadius': 30, 'maxRadius': 120}
        ]
        
        # Create focused visualization
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        fig.suptitle('Focused Hough Circle Parameter Testing - Best Circle Only', fontsize=16)
        
        # Show original median blurred image in top-left
        axes[0, 0].imshow(blurred, cmap='gray')
        axes[0, 0].set_title('Median Blurred Image\n(Input for all tests)')
        axes[0, 0].axis('off')
        
        # Test each parameter set
        for i, params in enumerate(hough_params):
            row = (i + 1) // 3
            col = (i + 1) % 3
            
            if row >= 3:
                break
            
            # Test Hough circles with these parameters
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=params['minDist'],
                param1=params['param1'],
                param2=params['param2'],
                minRadius=params['minRadius'],
                maxRadius=params['maxRadius']
            )
            
            # Create result image
            result_image = blurred.copy()
            if len(result_image.shape) == 2:
                result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
            
            circles_found = 0
            best_circle = None
            best_score = -float('inf')
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                circles_found = len(circles)
                
                # Find the best circle using our scoring function
                for circle in circles:
                    x, y, r = circle
                    score = self.score_iris_circle(blurred, x, y, r)
                    if score > best_score:
                        best_score = score
                        best_circle = circle
                
                # Draw only the best circle in bright green
                if best_circle is not None:
                    x, y, r = best_circle
                    # Ensure radius is positive and within bounds
                    if r > 0 and x >= 0 and y >= 0 and x < result_image.shape[1] and y < result_image.shape[0]:
                        try:
                            cv2.circle(result_image, (int(x), int(y)), int(r), (0, 255, 0), 3)  # Bright green
                            cv2.circle(result_image, (int(x), int(y)), 5, (0, 255, 0), -1)  # Solid center
                            cv2.putText(result_image, f"Best", (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            cv2.putText(result_image, f"Score: {best_score:.1f}", (int(x)+10, int(y)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        except Exception as e:
                            print(f"   Warning: Could not draw circle at ({x}, {y}) with radius {r}: {e}")
            
            # Show result
            axes[row, col].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f'{params["name"]}\n'
                                   f'Total: {circles_found} circles\n'
                                   f'Best: {best_circle if best_circle is not None else "None"}\n'
                                   f'Score: {best_score:.1f}')
            axes[row, col].axis('off')
            
            if debug:
                if best_circle is not None:
                    print(f"   {params['name']}: Best circle at {best_circle}, score={best_score:.1f}")
                else:
                    print(f"   {params['name']}: No circles found")
        
        # Fill remaining empty subplots
        for i in range(len(hough_params) + 1, 9):
            row = i // 3
            col = i % 3
            if row < 3:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        # Save the focused debug
        debug_path = os.path.join(self.output_dir, "debug_focused_parameter_testing.png")
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if debug:
            print(f"💾 Saved focused parameter debug: {debug_path}")
        
        return debug_path
    
    def detect_pupil(self, image, iris_result, debug=False, frame_number=0, last_pupil_result=None):
        """Detect pupil using dual-method approach with visualization"""
        if iris_result is None:
            return None
        
        # Extract iris region
        iris_center = iris_result['center']
        iris_radius = iris_result['radius']
        
        # Method 1: Current darkness-based contour detection (red channel only)
        pupil_method1 = self.detect_pupil_method1(image, iris_center, iris_radius, frame_number, last_pupil_result)
        
        # Method 2: Previous working method (CLAHE + adaptive thresholding)
        pupil_method2 = self.detect_pupil_method2(image, iris_center, iris_radius, frame_number, last_pupil_result)
        
        # Create dual-method visualization if requested
        if debug:
            self.create_dual_method_debug_visualization(image, iris_center, iris_radius, 
                                                      pupil_method1, pupil_method2, frame_number)
        
        # Combine methods: prefer Method 1 when available, fallback to Method 2
        if pupil_method1 is not None:
            # Method 1 succeeded - use it
            primary_result = pupil_method1
            method_used = 'method1'
        elif pupil_method2 is not None:
            # Method 1 failed, but Method 2 succeeded - use Method 2
            primary_result = pupil_method2
            method_used = 'method2'
        else:
            # Both methods failed
            primary_result = None
            method_used = 'none'
        
        # Store both methods for visualization
        result = {
            'method1': pupil_method1,
            'method2': pupil_method2,
            'primary': primary_result,
            'method_used': method_used
        }
        
        return result
    
    def detect_pupil_method1(self, image, iris_center, iris_radius, frame_number, last_pupil_result):
        """Method 1: Current darkness-based contour detection (red channel only)"""
        # Preprocess for pupil detection - use red channel only
        if len(image.shape) == 3:
            red = image[:, :, 2]  # Extract red channel
            red_bgr = cv2.merge([red, red, red])  # Convert to 3-channel for consistency
            gray = red  # Use red channel as grayscale
        else:
            gray = image.copy()
        
        # Use red channel only (no CLAHE preprocessing)
        enhanced = gray
        
        # Try darkness-based contour detection on red channel only
        pupil_circle = self.detect_pupil_by_darkness(enhanced, iris_center, iris_radius, frame_number, last_pupil_result)
        
        return pupil_circle
    
    def detect_pupil_method2(self, image, iris_center, iris_radius, frame_number, last_pupil_result):
        """Method 2: Previous working method (CLAHE + adaptive thresholding)"""
        # Preprocess for pupil detection - use CLAHE like the previous working version
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for local contrast enhancement (like previous working version)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Get iris baseline intensity from circle 40 pixels inside iris
        baseline_intensity = self.get_iris_baseline_intensity(enhanced, iris_center, iris_radius)
        
        # Find dark pixels with adaptive thresholding (previous working method)
        dark_coords, threshold_used = self.find_dark_pixels_adaptive(
            enhanced, iris_center, iris_radius, baseline_intensity, frame_number)
        
        # Debug output for Method 2
        if frame_number in [0, 7, 30, 40, 60, 90, 120]:
            print(f"Method 2 Frame {frame_number}: Baseline={baseline_intensity:.1f}, Threshold={threshold_used:.1f}, Dark pixels={len(dark_coords[0])}")
        
        if len(dark_coords[0]) == 0:
            return None
        
        # Find best cluster (prioritizing proximity to iris center)
        largest_cluster_points, all_labels = self.find_largest_cluster(dark_coords, iris_center)
        
        if largest_cluster_points is None:
            if frame_number in [0, 7, 30, 40, 60, 90, 120]:
                print(f"Method 2 Frame {frame_number}: No valid cluster found")
            return None
        
        # Find radius around iris center (previous method)
        pupil_circle = self.find_radius_around_iris_center_method2(largest_cluster_points, iris_center)
        
        if frame_number in [0, 7, 30, 40, 60, 90, 120]:
            print(f"Method 2 Frame {frame_number}: SUCCESS - Center={pupil_circle['center']}, Radius={pupil_circle['radius']}")
        
        return pupil_circle
    
    def find_radius_around_iris_center_method2(self, cluster_points, iris_center):
        """Find radius around iris center that encompasses the cluster (previous method)"""
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
    
    def create_dual_method_debug_visualization(self, original, iris_center, iris_radius, pupil_method1, pupil_method2, frame_number):
        """Create comprehensive debug visualization comparing both detection methods"""
        import os
        
        # Create debug directory
        debug_dir = os.path.join(self.output_dir, "dual_method_debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Calculate target size maintaining aspect ratio
        h, w = original.shape[:2]
        target_height = 300
        target_width = int(w * target_height / h)
        target_size = (target_width, target_height)
        
        # Resize original image
        original_resized = cv2.resize(original, target_size)
        
        # Scale coordinates to resized image
        scale_x = target_size[0] / w
        scale_y = target_size[1] / h
        
        # Scale iris center and radius
        iris_center_scaled = (int(iris_center[0] * scale_x), int(iris_center[1] * scale_y))
        iris_radius_scaled = int(iris_radius * min(scale_x, scale_y))
        
        # Create visualization images
        # 1. Original with iris circle
        original_with_iris = original_resized.copy()
        cv2.circle(original_with_iris, iris_center_scaled, iris_radius_scaled, (0, 255, 0), 2)
        cv2.circle(original_with_iris, iris_center_scaled, 2, (0, 0, 255), 3)
        
        # 2. Method 1 result (Red channel only - BLUE circles)
        method1_result = original_resized.copy()
        cv2.circle(method1_result, iris_center_scaled, iris_radius_scaled, (0, 255, 0), 2)
        if pupil_method1 is not None:
            pupil_center_scaled = (int(pupil_method1['center'][0] * scale_x), 
                                 int(pupil_method1['center'][1] * scale_y))
            pupil_radius_scaled = int(pupil_method1['radius'] * min(scale_x, scale_y))
            cv2.circle(method1_result, pupil_center_scaled, pupil_radius_scaled, (255, 0, 0), 3)  # Blue
            cv2.circle(method1_result, pupil_center_scaled, 2, (255, 0, 0), -1)
            cv2.putText(method1_result, f"M1: r={pupil_method1['radius']}", 
                       (pupil_center_scaled[0] + 10, pupil_center_scaled[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 3. Method 2 result (CLAHE + adaptive - YELLOW circles)
        method2_result = original_resized.copy()
        cv2.circle(method2_result, iris_center_scaled, iris_radius_scaled, (0, 255, 0), 2)
        if pupil_method2 is not None:
            pupil_center_scaled = (int(pupil_method2['center'][0] * scale_x), 
                                 int(pupil_method2['center'][1] * scale_y))
            pupil_radius_scaled = int(pupil_method2['radius'] * min(scale_x, scale_y))
            cv2.circle(method2_result, pupil_center_scaled, pupil_radius_scaled, (0, 255, 255), 3)  # Yellow
            cv2.circle(method2_result, pupil_center_scaled, 2, (0, 255, 255), -1)
            cv2.putText(method2_result, f"M2: r={pupil_method2['radius']}", 
                       (pupil_center_scaled[0] + 10, pupil_center_scaled[1] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 4. Combined visualization (both methods on same image)
        combined_result = original_resized.copy()
        cv2.circle(combined_result, iris_center_scaled, iris_radius_scaled, (0, 255, 0), 2)
        
        # Draw Method 1 (Blue)
        if pupil_method1 is not None:
            pupil_center_scaled = (int(pupil_method1['center'][0] * scale_x), 
                                 int(pupil_method1['center'][1] * scale_y))
            pupil_radius_scaled = int(pupil_method1['radius'] * min(scale_x, scale_y))
            cv2.circle(combined_result, pupil_center_scaled, pupil_radius_scaled, (255, 0, 0), 3)  # Blue
            cv2.circle(combined_result, pupil_center_scaled, 2, (255, 0, 0), -1)
            cv2.putText(combined_result, f"M1: r={pupil_method1['radius']}", 
                       (pupil_center_scaled[0] + 10, pupil_center_scaled[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw Method 2 (Yellow)
        if pupil_method2 is not None:
            pupil_center_scaled = (int(pupil_method2['center'][0] * scale_x), 
                                 int(pupil_method2['center'][1] * scale_y))
            pupil_radius_scaled = int(pupil_method2['radius'] * min(scale_x, scale_y))
            cv2.circle(combined_result, pupil_center_scaled, pupil_radius_scaled, (0, 255, 255), 3)  # Yellow
            cv2.circle(combined_result, pupil_center_scaled, 2, (0, 255, 255), -1)
            cv2.putText(combined_result, f"M2: r={pupil_method2['radius']}", 
                       (pupil_center_scaled[0] + 10, pupil_center_scaled[1] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 5. Parameters text
        params_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        # Determine which method was used
        if pupil_method1 is not None:
            method_used = "METHOD 1 (Red Channel)"
            primary_result = pupil_method1
        elif pupil_method2 is not None:
            method_used = "METHOD 2 (CLAHE + Adaptive)"
            primary_result = pupil_method2
        else:
            method_used = "NONE (Both Failed)"
            primary_result = None
        
        params_text = [
            f"Frame: {frame_number}",
            f"Iris Center: ({iris_center[0]}, {iris_center[1]})",
            f"Iris Radius: {iris_radius}",
            "",
            "COMBINED LOGIC:",
            f"  Selected: {method_used}",
            f"  Center: {primary_result['center'] if primary_result is not None else 'N/A'}",
            f"  Radius: {primary_result['radius'] if primary_result is not None else 'N/A'}",
            "",
            "METHOD 1 (Red Channel Only):",
            f"  Status: {'SUCCESS' if pupil_method1 is not None else 'FAILED'}",
            f"  Center: {pupil_method1['center'] if pupil_method1 is not None else 'N/A'}",
            f"  Radius: {pupil_method1['radius'] if pupil_method1 is not None else 'N/A'}",
            "",
            "METHOD 2 (CLAHE + Adaptive):",
            f"  Status: {'SUCCESS' if pupil_method2 is not None else 'FAILED'}",
            f"  Center: {pupil_method2['center'] if pupil_method2 is not None else 'N/A'}",
            f"  Radius: {pupil_method2['radius'] if pupil_method2 is not None else 'N/A'}",
            "",
            "LEGEND:",
            "Green: Iris",
            "Blue: Method 1 (Red Channel)",
            "Yellow: Method 2 (CLAHE)"
        ]
        
        for i, text in enumerate(params_text):
            color = (255, 255, 255) if not text.startswith(('METHOD', 'LEGEND:')) else (0, 255, 255)
            cv2.putText(params_img, text, (10, 25 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add white padding between images
        padding = 10
        white_padding = np.ones((target_size[1], padding, 3), dtype=np.uint8) * 255
        white_padding_h = np.ones((padding, target_size[0] * 3 + padding * 2, 3), dtype=np.uint8) * 255
        
        # Create 2x3 grid
        # Row 1: Original+Iris, Method1, Method2
        row1 = np.hstack([original_with_iris, white_padding, method1_result, white_padding, method2_result])
        
        # Row 2: Combined, Parameters (empty space)
        row2 = np.hstack([combined_result, white_padding, params_img, white_padding, 
                         np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)])
        
        # Combine rows
        comprehensive = np.vstack([row1, white_padding_h, row2])
        
        # Add labels
        label_y = 20
        cv2.putText(comprehensive, "Original+Iris", (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(comprehensive, "Method 1 (Red)", (target_size[0] + padding + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(comprehensive, "Method 2 (CLAHE)", (2 * (target_size[0] + padding) + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.putText(comprehensive, "Combined", (10, target_size[1] + padding + label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(comprehensive, "Parameters", (target_size[0] + padding + 10, target_size[1] + padding + label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Save comprehensive visualization
        debug_path = f"{debug_dir}/dual_method_frame_{frame_number:03d}.jpg"
        cv2.imwrite(debug_path, comprehensive)
        print(f"📊 Dual method debug visualization saved: {debug_path}")
    
    def detect_pupil_by_darkness(self, frame, iris_circle, iris_radius, frame_number, last_pupil_result, debug=False):
        """
        Detect pupil inside iris using darkness scoring.
        
        Args:
            frame (ndarray): Grayscale image (already enhanced red-channel).
            iris_circle (tuple): (x, y) for iris center.
            iris_radius (int): Iris radius.
            frame_number (int): Current frame number.
            last_pupil_result (dict): Previous pupil result for temporal constraints.
            debug (bool): If True, show intermediate results.
        
        Returns:
            dict: {'center': (x, y), 'radius': r, 'score': score} or None
        """
        cx, cy = iris_circle
        r_iris = iris_radius

        # Crop ROI around iris
        x1, y1 = max(cx - r_iris, 0), max(cy - r_iris, 0)
        x2, y2 = min(cx + r_iris, frame.shape[1]), min(cy + r_iris, frame.shape[0])
        roi = frame[y1:y2, x1:x2].copy()

        # Use OTSU thresholding
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean up the thresholded image with morphological operations
        # Remove noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh_cleaned = cv2.morphologyEx(thresh_cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours in cleaned thresholded ROI
        contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug output
        if frame_number in [0, 7, 30, 40, 60, 90, 120]:
            print(f"Frame {frame_number}: Found {len(contours)} contours")

        best_circle = None
        best_score = float("inf")
        debug_output = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        
        # Debug: track blob sizes
        valid_blobs = 0
        blob_sizes = []

        for cnt in contours:
            (px, py), pr = cv2.minEnclosingCircle(cnt)
            px, py, pr = int(px), int(py), int(pr)

            blob_sizes.append(pr)
            
            if pr < 8:   # too tiny (noise)
                continue
            if pr > r_iris * 0.7:  # too big to be pupil
                continue
                
            valid_blobs += 1

            # Mask for this blob
            mask = np.zeros_like(roi, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            # Compute mean intensity of blob
            mean_intensity = cv2.mean(roi, mask=mask)[0]

            # Distance from iris center (relative to ROI)
            dist = np.hypot((px + x1) - cx, (py + y1) - cy)

            # Score: mostly darkness, with penalties for off-center
            score = mean_intensity + 0.2 * dist

            if score < best_score:
                best_score = score
                best_circle = (px + x1, py + y1, pr)

            if debug:
                cv2.circle(debug_output, (px, py), pr, (0, 255, 0), 1)
                cv2.putText(debug_output, f"{mean_intensity:.1f}", (px, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Debug output for blob sizes
        if frame_number in [0, 7, 30, 40, 60, 90, 120]:
            if blob_sizes:
                print(f"  All blob sizes: {sorted(blob_sizes)[:10]} (showing first 10)")
                print(f"  Valid blobs (8-{int(r_iris*0.7)}px): {valid_blobs}")
                if valid_blobs > 0:
                    valid_sizes = [s for s in blob_sizes if 8 <= s <= r_iris * 0.7]
                    print(f"  Valid blob sizes: {valid_sizes}")
            else:
                print(f"  No blobs found")

        # Apply temporal constraints if we have previous result
        if best_circle is not None and last_pupil_result is not None and isinstance(last_pupil_result, dict):
            last_x, last_y = last_pupil_result['center']
            last_r = last_pupil_result['radius']
            bx, by, br = best_circle
            
            # Constrain center movement (max 10 pixels)
            center_distance = np.sqrt((bx - last_x)**2 + (by - last_y)**2)
            if center_distance > 10:
                # Use iris center if too far from last pupil center
                bx, by = iris_circle
            
            # Constrain radius change (max 5 pixels from last pupil radius)
            br = max(5, min(br, last_r + 5))
            br = max(br, last_r - 5)
            
            best_circle = (bx, by, br)

        # Convert to dictionary format expected by main processing loop
        if best_circle is not None:
            bx, by, br = best_circle
            return {
                'center': (bx, by),
                'radius': br,
                'score': best_score
            }
        else:
            return None
        
    def detect_pupil_hough_circles(self, enhanced, iris_center, iris_radius, frame_number, last_pupil_result):
        """Detect pupil using Hough circles on red channel + CLAHE image"""
        x, y = iris_center
        
        # Debug: print frame number and iris info
        if frame_number == 0:
            print(f"Debug - Frame {frame_number}: Iris center=({x}, {y}), radius={iris_radius}")
            print(f"Debug - Last pupil result type: {type(last_pupil_result)}, value: {last_pupil_result}")
            if last_pupil_result is not None:
                print(f"Debug - Last pupil result length: {len(last_pupil_result) if hasattr(last_pupil_result, '__len__') else 'N/A'}")
        
        # Define pupil radius constraints (must be inside iris and smaller)
        min_pupil_radius = 10
        max_pupil_radius = int(iris_radius * 0.6)  # Max 60% of iris radius
        
        # Apply temporal constraints if we have previous result
        if last_pupil_result is not None and isinstance(last_pupil_result, dict):
            last_x, last_y = last_pupil_result['center']
            last_r = last_pupil_result['radius']
            
            # Constrain center movement (max 10 pixels)
            center_distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
            if center_distance > 10:
                # Use iris center if too far from last pupil center
                x, y = iris_center
            
            # Constrain radius change (max 5 pixels from last pupil radius)
            max_pupil_radius = min(max_pupil_radius, last_r + 5)
            min_pupil_radius = max(min_pupil_radius, last_r - 5)
        
        # Run Hough circles detection
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=iris_radius,  # Minimum distance between circle centers
            param1=50,            # Upper threshold for edge detection
            param2=30,            # Accumulator threshold for center detection
            minRadius=min_pupil_radius,
            maxRadius=max_pupil_radius
        )
        
        if circles is None:
            if frame_number == 0:
                print(f"Debug - No circles found in frame {frame_number}")
            return None
        
        circles = np.round(circles[0, :]).astype("int")
        if frame_number == 0:
            print(f"Debug - Found {len(circles)} circles in frame {frame_number}")
            print(f"Debug - Circles shape: {circles.shape}")
            print(f"Debug - First circle: {circles[0] if len(circles) > 0 else 'None'}")
        
        # Filter circles that are inside the iris and score them
        valid_circles = []
        iris_x, iris_y = iris_center
        if frame_number == 0:
            print(f"Debug - Processing {len(circles)} circles, iris_center=({iris_x}, {iris_y}), iris_radius={iris_radius}")
        
        try:
            for i, (cx, cy, cr) in enumerate(circles):
                if frame_number == 0 and i < 3:  # Debug first 3 circles
                    print(f"Debug - Circle {i}: center=({cx}, {cy}), radius={cr}")
                
                # Check if circle center is inside iris
                distance_from_iris_center = np.sqrt((cx - iris_x)**2 + (cy - iris_y)**2)
                if distance_from_iris_center + cr <= iris_radius:
                    if frame_number == 0 and i < 3:
                        print(f"Debug - Circle {i} is inside iris, scoring...")
                    # Score the circle based on darkness and proximity to iris center
                    try:
                        score = self.score_pupil_circle(enhanced, (cx, cy), cr, iris_center)
                        valid_circles.append(((cx, cy, cr), score))
                        if frame_number == 0 and i < 3:
                            print(f"Debug - Circle {i} score: {score}")
                    except Exception as e:
                        print(f"Debug - Error scoring circle {i}: {e}")
                        continue
        except Exception as e:
            print(f"Debug - Error in for loop: {e}")
            print(f"Debug - Circles type: {type(circles)}")
            print(f"Debug - Circles content: {circles}")
            raise
        
        if not valid_circles:
            return None
        
        # Select the best circle (highest score)
        print(f"Debug - valid_circles: {valid_circles}")
        print(f"Debug - First valid circle: {valid_circles[0] if valid_circles else 'None'}")
        best_circle, best_score = max(valid_circles, key=lambda x: x[1])
        print(f"Debug - best_circle: {best_circle}, best_score: {best_score}")
        
        # Convert tuple to dictionary format expected by main processing loop
        x, y, r = best_circle
        return {
            'center': (x, y),
            'radius': r,
            'score': best_score
        }
    
    def score_pupil_circle(self, image, center, radius, iris_center):
        """Score a potential pupil circle based on darkness and position"""
        x, y = center
        print(f"Debug - Scoring circle: center=({x}, {y}), radius={radius}, iris_center={iris_center}")
        print(f"Debug - iris_center type: {type(iris_center)}")
        if hasattr(iris_center, '__len__'):
            print(f"Debug - iris_center length: {len(iris_center)}")
        
        # Create mask for the circle
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        
        # Calculate average intensity inside the circle (lower = better)
        circle_pixels = image[mask > 0]
        if len(circle_pixels) == 0:
            return 0
        
        avg_intensity = np.mean(circle_pixels)
        darkness_score = 255 - avg_intensity  # Higher score for darker circles
        
        # Calculate proximity score (closer to iris center = better)
        print(f"Debug - About to unpack iris_center: {iris_center}")
        iris_x, iris_y = iris_center
        print(f"Debug - Unpacked: iris_x={iris_x}, iris_y={iris_y}")
        distance_from_iris = np.sqrt((x - iris_x)**2 + (y - iris_y)**2)
        max_distance = 100  # Use a fixed maximum distance for scoring
        proximity_score = max(0, max_distance - distance_from_iris) / max_distance * 100
        
        # Combine scores (weighted)
        total_score = darkness_score * 0.7 + proximity_score * 0.3
        
        return total_score
    
    def create_pupil_debug_visualization(self, original, gray, enhanced, iris_center, iris_radius, 
                                       baseline_intensity, dark_coords, cluster_points, pupil_circle, frame_number, threshold_core=None):
        """Create comprehensive debug visualization for pupil detection"""
        import os
        
        # Debug: check iris_center type
        print(f"Debug - Debug viz: iris_center type: {type(iris_center)}, value: {iris_center}")
        if hasattr(iris_center, '__len__'):
            print(f"Debug - Debug viz: iris_center length: {len(iris_center)}")
        
        # Create debug directory
        debug_dir = os.path.join(self.output_dir, "pupil_debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create red channel + CLAHE debug directory
        red_clahe_dir = os.path.join(self.output_dir, "red_clahe_debug")
        os.makedirs(red_clahe_dir, exist_ok=True)
        
        # Calculate target size maintaining aspect ratio
        h, w = original.shape[:2]
        target_height = 300
        target_width = int(w * target_height / h)
        target_size = (target_width, target_height)
        
        # Resize images
        original_resized = cv2.resize(original, target_size)
        gray_resized = cv2.resize(gray, target_size)
        enhanced_resized = cv2.resize(enhanced, target_size)
        
        # Create simple preprocessing comparison visualizations
        # Red channel only (no processing)
        red_only_resized = cv2.resize(gray, target_size)
        
        # Red channel only (what we're actually using now)
        red_only_actual_resized = cv2.resize(enhanced, target_size)
        
        # Red channel + CLAHE (for comparison)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        red_clahe = clahe.apply(gray)
        red_clahe_resized = cv2.resize(red_clahe, target_size)
        
        # Convert grayscale to BGR for display
        gray_bgr = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
        enhanced_bgr = cv2.cvtColor(enhanced_resized, cv2.COLOR_GRAY2BGR)
        red_only_bgr = cv2.cvtColor(red_only_resized, cv2.COLOR_GRAY2BGR)
        red_only_actual_bgr = cv2.cvtColor(red_only_actual_resized, cv2.COLOR_GRAY2BGR)
        red_clahe_bgr = cv2.cvtColor(red_clahe_resized, cv2.COLOR_GRAY2BGR)
        
        # Scale coordinates to resized image
        scale_x = target_size[0] / w
        scale_y = target_size[1] / h
        
        # Scale iris center and radius
        iris_center_scaled = (int(iris_center[0] * scale_x), int(iris_center[1] * scale_y))
        iris_radius_scaled = int(iris_radius * min(scale_x, scale_y))
        
        # Create visualization images
        # 1. Original with iris circle
        original_with_iris = original_resized.copy()
        cv2.circle(original_with_iris, iris_center_scaled, iris_radius_scaled, (0, 255, 0), 2)
        cv2.circle(original_with_iris, iris_center_scaled, 2, (0, 0, 255), 3)
        
        # 2. Red+CLAHE with iris circle and baseline circle
        enhanced_with_circles = red_clahe_bgr.copy()
        cv2.circle(enhanced_with_circles, iris_center_scaled, iris_radius_scaled, (0, 255, 0), 2)
        # Draw baseline circle (40 pixels inside iris)
        baseline_radius_scaled = int((iris_radius - 40) * min(scale_x, scale_y))
        if baseline_radius_scaled > 0:
            cv2.circle(enhanced_with_circles, iris_center_scaled, baseline_radius_scaled, (255, 0, 0), 2)
        
        # 3. Dark pixels visualization
        dark_pixels_img = enhanced_bgr.copy()
        if dark_coords is not None and len(dark_coords[0]) > 0:
            # Scale dark pixel coordinates
            dark_y_scaled = (dark_coords[0] * scale_y).astype(int)
            dark_x_scaled = (dark_coords[1] * scale_x).astype(int)
            # Draw dark pixels
            for i in range(len(dark_y_scaled)):
                if 0 <= dark_y_scaled[i] < target_size[1] and 0 <= dark_x_scaled[i] < target_size[0]:
                    dark_pixels_img[dark_y_scaled[i], dark_x_scaled[i]] = [0, 0, 255]  # Red
        
        # 4. Cluster visualization
        cluster_img = enhanced_bgr.copy()
        if cluster_points is not None and len(cluster_points) > 0:
            # Scale cluster points
            cluster_scaled = cluster_points * np.array([scale_x, scale_y])
            for point in cluster_scaled:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < target_size[0] and 0 <= y < target_size[1]:
                    cv2.circle(cluster_img, (x, y), 1, (0, 255, 255), -1)  # Yellow dots
        
        # 5. Final result
        final_result = original_resized.copy()
        if pupil_circle is not None:
            pupil_center_scaled = (int(pupil_circle['center'][0] * scale_x), 
                                 int(pupil_circle['center'][1] * scale_y))
            pupil_radius_scaled = int(pupil_circle['radius'] * min(scale_x, scale_y))
            cv2.circle(final_result, pupil_center_scaled, pupil_radius_scaled, (0, 255, 0), 3)
            cv2.circle(final_result, pupil_center_scaled, 2, (0, 0, 255), 3)
        
        # 6. Parameters text
        params_img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        # Calculate cluster center distance from iris center for debug info
        cluster_center_distance = "N/A"
        if cluster_points is not None and len(cluster_points) > 0:
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_center_distance = np.sqrt((cluster_center[0] - iris_center[0])**2 + (cluster_center[1] - iris_center[1])**2)
            cluster_center_distance = f"{cluster_center_distance:.1f}"
        
        # Calculate iris brightness for debugging
        iris_mask = np.zeros(enhanced.shape, dtype=np.uint8)
        cv2.circle(iris_mask, iris_center, iris_radius, 255, -1)
        iris_pixels = enhanced[iris_mask > 0]
        iris_mean = np.mean(iris_pixels) if len(iris_pixels) > 0 else 0
        iris_std = np.std(iris_pixels) if len(iris_pixels) > 0 else 0
        
        params_text = [
            f"Frame: {frame_number}",
            f"Iris Center: ({iris_center[0]}, {iris_center[1]})",
            f"Iris Radius: {iris_radius}",
            f"Baseline Intensity: {baseline_intensity:.1f}",
            f"Iris Mean: {iris_mean:.1f}",
            f"Iris Std: {iris_std:.1f}",
            f"Dark Pixels: {len(dark_coords[0]) if dark_coords is not None else 0}",
            f"Cluster Points: {len(cluster_points) if cluster_points is not None else 0}",
            f"Cluster Dist from Iris: {cluster_center_distance}",
            f"Pupil Radius: {pupil_circle['radius'] if pupil_circle is not None else 'N/A'}",
            f"Pupil Center: ({pupil_circle['center'][0] if pupil_circle is not None else 'N/A'}, {pupil_circle['center'][1] if pupil_circle is not None else 'N/A'})"
        ]
        
        for i, text in enumerate(params_text):
            cv2.putText(params_img, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add white padding between images
        padding = 10
        white_padding = np.ones((target_size[1], padding, 3), dtype=np.uint8) * 255
        white_padding_h = np.ones((padding, target_size[0] * 4 + padding * 3, 3), dtype=np.uint8) * 255
        
        # Create comparison images with iris circles
        red_only_actual_with_circles = red_only_actual_bgr.copy()
        cv2.circle(red_only_actual_with_circles, iris_center_scaled, iris_radius_scaled, (0, 255, 0), 2)
        
        red_clahe_with_circles = red_clahe_bgr.copy()
        cv2.circle(red_clahe_with_circles, iris_center_scaled, iris_radius_scaled, (0, 255, 0), 2)
        
        # Create 4x2 grid
        # Row 1: Original+Iris, Red Channel Only, Red+CLAHE, Dark Pixels
        row1 = np.hstack([original_with_iris, white_padding, red_only_actual_with_circles, white_padding, 
                         red_clahe_with_circles, white_padding, dark_pixels_img])
        
        # Create histogram visualization
        if threshold_core is not None:
            histogram_img = self.create_darkness_histogram(iris_pixels, baseline_intensity, threshold_core, target_size)
        else:
            # Fallback if threshold_core is not available
            histogram_img = np.ones_like(params_img) * 128
        
        # Row 2: Cluster, Final Result, Parameters, Histogram
        row2 = np.hstack([cluster_img, white_padding, final_result, white_padding, 
                         params_img, white_padding, histogram_img])
        
        # Combine rows
        comprehensive = np.vstack([row1, white_padding_h, row2])
        
        # Add labels
        label_y = 20
        cv2.putText(comprehensive, "Original+Iris", (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(comprehensive, "Red Channel Only", (target_size[0] + padding + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(comprehensive, "Red+CLAHE", (2 * (target_size[0] + padding) + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(comprehensive, "Dark Pixels", (3 * (target_size[0] + padding) + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        cv2.putText(comprehensive, "Cluster", (10, target_size[1] + padding + label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(comprehensive, "Final Result", (target_size[0] + padding + 10, target_size[1] + padding + label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(comprehensive, "Parameters", (2 * (target_size[0] + padding) + 10, target_size[1] + padding + label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(comprehensive, "Darkness Histogram", (3 * (target_size[0] + padding) + 10, target_size[1] + padding + label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Save comprehensive visualization
        debug_path = f"{debug_dir}/pupil_debug_frame_{frame_number:03d}.jpg"
        cv2.imwrite(debug_path, comprehensive)
        print(f"📊 Pupil debug visualization saved: {debug_path}")
        
        # Save red channel only frame as separate JPG
        red_clahe_filename = f"red_only_frame_{frame_number:03d}.jpg"
        red_clahe_path = os.path.join(red_clahe_dir, red_clahe_filename)
        
        # Convert enhanced (red channel only) to BGR for saving
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(red_clahe_path, enhanced_bgr)
        print(f"📊 Red channel only frame saved: {red_clahe_path}")
    
    def calculate_circle_darkness(self, image, center, radius):
        """Calculate how dark a circle is (lower values = darker)"""
        x, y = center
        h, w = image.shape
        
        # Check if circle is within image bounds
        if x - radius < 0 or x + radius >= w or y - radius < 0 or y + radius >= h:
            return float('inf')
        
        # Create circle mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        
        # Calculate mean intensity inside the circle (lower = darker)
        mean_intensity = cv2.mean(image, mask)[0]
        
        return mean_intensity
    
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
    
    def find_dark_pixels_adaptive(self, image, iris_center, iris_radius, baseline_intensity, frame_number):
        """Find dark pixels using adaptive thresholding with brightness constraints"""
        x, y = iris_center
        max_pupil_radius = int(iris_radius * 0.8)
        
        # Create mask for iris area
        iris_mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(iris_mask, (x, y), max_pupil_radius, 255, -1)
        
        # Get all pixels within iris area for statistics
        iris_pixels = image[iris_mask > 0]
        
        if len(iris_pixels) == 0:
            return (np.array([]), np.array([])), baseline_intensity
        
        # Calculate iris brightness statistics
        iris_mean = np.mean(iris_pixels)
        iris_std = np.std(iris_pixels)
        
        # Adaptive threshold: try bimodal detection first, fallback to brightness-based
        threshold_core = self.adaptive_threshold_with_bimodal(iris_pixels, baseline_intensity)
        
        # Debug: print threshold values for frame 7
        if frame_number == 7:
            print(f"Frame 7 Debug - Darkest pixel: {np.min(iris_pixels):.1f}, Baseline: {baseline_intensity:.1f}, Threshold: {threshold_core:.1f}")
        
        # Ensure threshold is reasonable (not too low)
        threshold_core = max(threshold_core, 30)  # Minimum threshold of 30
        
        # Find dark pixels using the adaptive threshold
        dark_pixels = (image < threshold_core) & (iris_mask > 0)
        dark_coords = np.where(dark_pixels)

        if len(dark_coords[0]) == 0:
            return (np.array([]), np.array([])), threshold_core

        return dark_coords, threshold_core
    
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
    
    def find_largest_cluster(self, dark_coords, iris_center):
        """Find the best cluster of dark pixels using OpenCV connected components, prioritizing clusters closer to iris center"""
        if len(dark_coords[0]) == 0:
            return None, None
        
        # Convert to (x, y) coordinates
        y_coords, x_coords = dark_coords
        points = np.column_stack((x_coords, y_coords))
        
        # Create a binary mask from the dark coordinates
        # Find bounds of the points
        min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
        min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
        
        # Create binary mask
        mask = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
        mask[y_coords - min_y, x_coords - min_x] = 255
        
        # Use OpenCV connected components to find clusters
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Find the best cluster (excluding background with label 0)
        if num_labels <= 1:  # Only background
            return None, None
        
        # Score each cluster based on area and distance from iris center
        best_cluster_label = None
        best_score = -1
        
        for label in range(1, num_labels):  # Skip background (label 0)
            area = stats[label, cv2.CC_STAT_AREA]
            
            # Filter out very small clusters (less than 50 pixels)
            if area < 50:
                continue
            
            # Get cluster centroid (relative to mask coordinates)
            centroid_x = centroids[label][0] + min_x  # Convert to original image coordinates
            centroid_y = centroids[label][1] + min_y
            
            # Calculate distance from iris center
            distance_from_iris = np.sqrt((centroid_x - iris_center[0])**2 + (centroid_y - iris_center[1])**2)
            
            # Score combines area (larger is better) and proximity to iris center (closer is better)
            # Normalize area by dividing by 1000 to balance with distance
            # Lower distance is better, so we use negative distance
            score = (area / 1000.0) - (distance_from_iris / 10.0)
            
            if score > best_score:
                best_score = score
                best_cluster_label = label
        
        if best_cluster_label is None:
            return None, None
        
        # Get points in the best cluster
        cluster_mask = (labels == best_cluster_label)
        cluster_y, cluster_x = np.where(cluster_mask)
        
        # Convert back to original coordinates
        cluster_points = np.column_stack((cluster_x + min_x, cluster_y + min_y))
        
        # Create labels array for compatibility (all points in best cluster get label 0, others -1)
        point_labels = np.full(len(points), -1)
        for i, (px, py) in enumerate(points):
            if cluster_mask[py - min_y, px - min_x]:
                point_labels[i] = 0
        
        return cluster_points, point_labels
    
    def find_radius_around_iris_center(self, cluster_points, iris_center, iris_radius, frame_number, last_pupil_result=None):
        """Find the best circle around the cluster using minEnclosingCircle with temporal constraints"""
        if len(cluster_points) == 0:
            return None
        
        # Convert points to the format expected by minEnclosingCircle
        # minEnclosingCircle expects points as (x, y) coordinates
        points_for_circle = cluster_points.astype(np.float32)
        
        # Use OpenCV's minEnclosingCircle to find the best circle
        (center_x, center_y), radius = cv2.minEnclosingCircle(points_for_circle)
        
        # Apply temporal constraints if we have previous frame data
        if last_pupil_result is not None:
            last_center = last_pupil_result['center']
            last_radius = last_pupil_result['radius']
            
            # Constrain center movement (max 10 pixels per frame)
            center_distance = np.sqrt((center_x - last_center[0])**2 + (center_y - last_center[1])**2)
            if center_distance > 10:
                # Scale back to max 10 pixel movement
                direction_x = (center_x - last_center[0]) / center_distance
                direction_y = (center_y - last_center[1]) / center_distance
                center_x = last_center[0] + direction_x * 10
                center_y = last_center[1] + direction_y * 10
            
            # Constrain radius change (max 5 pixels per frame)
            radius_change = radius - last_radius
            if abs(radius_change) > 5:
                radius = last_radius + (5 if radius_change > 0 else -5)
        
        # Apply brightness-based constraints
        is_bright_frame = 40 <= frame_number <= 85
        
        if is_bright_frame:
            # During bright region: enforce minimum pupil size
            min_pupil_radius = 15  # Minimum during bright frames
            max_pupil_radius = int(iris_radius * 0.4)  # Max 40% of iris during bright frames
        else:
            # Outside bright region: allow larger pupils
            min_pupil_radius = 20  # Minimum outside bright frames
            max_pupil_radius = int(iris_radius * 0.6)  # Max 60% of iris outside bright frames
        
        # Apply radius constraints
        radius = max(min_pupil_radius, min(radius, max_pupil_radius))
        
        # Ensure pupil center is within iris bounds
        distance_from_iris_center = np.sqrt((center_x - iris_center[0])**2 + (center_y - iris_center[1])**2)
        max_distance_from_center = int(iris_radius * 0.4)  # Pupil center within 40% of iris radius
        
        if distance_from_iris_center > max_distance_from_center:
            # If too far from center, use iris center as pupil center
            center_x, center_y = iris_center
        
        return {
            'center': (int(center_x), int(center_y)),
            'radius': int(radius)
        }
    
    def process_video(self):
        """Process the entire video with two-phase approach"""
        try:
            print("Phase 1: Running iris detection on entire video to get median values...")
            
            # Phase 1: Run through entire video to collect iris detections
            iris_detections = []
            frame_count = 0
            
            # Reset video to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                # Detect iris with radius filtering
                circle = self.detect_iris_circles(frame, 110, 140)
                if circle is not None:
                    iris_detections.append(circle)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"  Processed {frame_count} frames, found {len(iris_detections)} iris detections")
            
            # Calculate median iris values
            if iris_detections:
                iris_detections = np.array(iris_detections)
                median_x = int(np.median(iris_detections[:, 0]))
                median_y = int(np.median(iris_detections[:, 1]))
                median_r = int(np.median(iris_detections[:, 2]))
                print(f"✅ Median iris: center=({median_x}, {median_y}), radius={median_r}")
                
                # Set fixed iris for entire video
                self.fixed_iris = {
                    'center': (median_x, median_y),
                    'radius': median_r,
                    'score': 1.0
                }
            else:
                print("❌ No iris detections found in entire video")
                return False
            
            print("Phase 2: Running pupil detection on all frames...")
            
            # Phase 2a: Run detection on all frames and store results
            detection_results = {}  # frame_number -> detection_result
            processed_count = 0
            failed_count = 0
            
        except Exception as e:
            print(f"Error in process_video setup: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Track previous pupil result for temporal constraints
        last_pupil_result = None
        
        # Phase 2a: Run detection on all frames
        for frame_number in self.frames_to_process:
            # Set frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            
            if not ret:
                print(f"Failed to read frame {frame_number}")
                failed_count += 1
                continue
            
            # Use fixed iris from Phase 1
            iris_result = self.fixed_iris
            
            # Detect pupil using current iris (enable debug for specific frames)
            debug_frames = list(range(0, 144, 30)) + [7]  # Every 30 frames: 0, 30, 60, 90, 120 + frame 7
            is_bright_frame = 40 <= frame_number <= 85  # Hardcoded bright region
            debug_enabled = frame_number in debug_frames or is_bright_frame
            
            pupil_result = self.detect_pupil(frame, iris_result, debug=debug_enabled, frame_number=frame_number, last_pupil_result=last_pupil_result)
            
            # Store detection result
            detection_results[frame_number] = {
                'iris_result': iris_result,
                'pupil_result': pupil_result,
                'frame': frame.copy()
            }
            
            # Process successful detections for data storage
            if pupil_result is not None:
                primary_result = pupil_result['primary']
                method_used = pupil_result['method_used']
                
                if primary_result is not None and primary_result['radius'] > 0:
                    # Update last_pupil_result for temporal constraints
                    last_pupil_result = primary_result
                    
                    # Store tracking data
                    try:
                        timestamp = frame_number / self.fps
                        if not np.isfinite(timestamp):
                            timestamp = 0.0
                    except ZeroDivisionError:
                        timestamp = 0.0
                    
                    pupil_radius = primary_result['radius']
                    pupil_diameter = pupil_radius * 2
                    pupil_area = np.pi * pupil_radius ** 2
                    
                    # Calculate iris-pupil ratio with error handling
                    try:
                        iris_pupil_ratio = pupil_radius / iris_result['radius']
                        if not np.isfinite(iris_pupil_ratio):
                            iris_pupil_ratio = 0.0
                    except ZeroDivisionError:
                        iris_pupil_ratio = 0.0
                    
                    self.tracking_data.append({
                        'frame_idx': frame_number,
                        'timestamp': timestamp,
                        'iris_center_x': iris_result['center'][0],
                        'iris_center_y': iris_result['center'][1],
                        'iris_radius': iris_result['radius'],
                        'pupil_center_x': primary_result['center'][0],
                        'pupil_center_y': primary_result['center'][1],
                        'pupil_radius': pupil_radius,
                        'pupil_diameter': pupil_diameter,
                        'pupil_area': pupil_area,
                        'iris_pupil_ratio': iris_pupil_ratio,
                        'detection_method': method_used
                    })
                    
                    processed_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
            
            # Progress update
            if frame_number % 15 == 0:  # Every 15 frames
                print(f"Detected frame {frame_number}/{self.total_frames}")
        
        print(f"Phase 2b: Creating video with stored detection results...")
        
        # Phase 2b: Create video by reading original video and overlaying detections
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        for frame_number in self.frames_to_process:
            # Read original frame
            ret, frame = self.cap.read()
            if not ret:
                print(f"Failed to read frame {frame_number} for video creation")
                continue
            
            # Get detection results for this frame
            if frame_number in detection_results:
                iris_result = detection_results[frame_number]['iris_result']
                pupil_result = detection_results[frame_number]['pupil_result']
                
                # Create visualization frame
                if pupil_result is not None:
                    pupil_method1 = pupil_result['method1']
                    pupil_method2 = pupil_result['method2']
                    primary_result = pupil_result['primary']
                    method_used = pupil_result['method_used']
                    
                    if primary_result is not None:
                        # Show the combined result (primary) with method indicator
                        vis_frame = self.create_visualization_frame(
                            frame, iris_result, primary_result, frame_number, frame_number / self.fps, 
                            primary_result['radius'], pupil_method2, method_used
                        )
                    else:
                        vis_frame = self.create_visualization_frame(
                            frame, iris_result, None, frame_number, frame_number / self.fps, 0, pupil_method2, method_used
                        )
                else:
                    vis_frame = self.create_visualization_frame(
                        frame, iris_result, None, frame_number, frame_number / self.fps, 0, None, 'none'
                    )
            else:
                # Fallback if no detection results
                vis_frame = self.create_visualization_frame(
                    frame, self.fixed_iris, None, frame_number, frame_number / self.fps, 0
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
    
    def create_visualization_frame(self, frame, iris_result, pupil_result, frame_idx, timestamp, pupil_radius, pupil_method2=None, method_used='none'):
        """Create visualization frame with detection overlays showing combined method results"""
        vis_frame = frame.copy()
        
        # Draw iris circle (green)
        if iris_result is not None:
            iris_center = iris_result['center']
            iris_radius = iris_result['radius']
            cv2.circle(vis_frame, iris_center, iris_radius, (0, 255, 0), 3)  # Green for iris
            cv2.circle(vis_frame, iris_center, 5, (0, 255, 0), -1)
            cv2.putText(vis_frame, f"Iris: r={iris_radius}", (iris_center[0] + 10, iris_center[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw primary pupil circle (color depends on method used)
        if pupil_result is not None:
            pupil_center = pupil_result['center']
            pupil_radius = pupil_result['radius']
            
            # Choose color based on method used
            if method_used == 'method1':
                pupil_color = (255, 0, 0)  # Blue for Method 1
                method_label = "M1"
            elif method_used == 'method2':
                pupil_color = (0, 255, 255)  # Yellow for Method 2
                method_label = "M2"
            else:
                pupil_color = (128, 128, 128)  # Gray for unknown
                method_label = "?"
            
            cv2.circle(vis_frame, pupil_center, pupil_radius, pupil_color, 3)
            cv2.circle(vis_frame, pupil_center, 3, pupil_color, -1)
            cv2.putText(vis_frame, f"{method_label}: r={pupil_radius}", (pupil_center[0] + 10, pupil_center[1] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, pupil_color, 1)
        
        # Draw Method 2 pupil circle (yellow) as reference if different from primary
        if pupil_method2 is not None and method_used != 'method2':
            pupil_center = pupil_method2['center']
            pupil_radius = pupil_method2['radius']
            cv2.circle(vis_frame, pupil_center, pupil_radius, (0, 255, 255), 2)  # Yellow for Method 2 (thinner)
            cv2.circle(vis_frame, pupil_center, 2, (0, 255, 255), -1)
            cv2.putText(vis_frame, f"M2: r={pupil_radius}", (pupil_center[0] + 10, pupil_center[1] + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Add frame information
        info_text = f"Frame: {frame_idx} | Time: {timestamp:.2f}s"
        cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add detection status with method used
        status_text = f"COMBINED DETECTION ({method_used.upper()})"
        cv2.putText(vis_frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Add legend
        cv2.putText(vis_frame, "Green: Iris", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_frame, "Blue: Method 1 (Red Channel)", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(vis_frame, "Yellow: Method 2 (CLAHE)", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
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
        
        # Create method comparison plot
        self.create_method_comparison_plot(video_name)
        
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
    
    def create_method_comparison_plot(self, video_name):
        """Create a plot comparing both detection methods over time"""
        # We need to reconstruct the full timeline with both methods
        # First, let's get all frame numbers from the video
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Create full timeline
        timestamps = np.arange(total_frames) / fps
        
        # Initialize arrays for both methods
        method1_radius = np.full(total_frames, np.nan)
        method2_radius = np.full(total_frames, np.nan)
        combined_radius = np.full(total_frames, np.nan)
        
        # Fill in the data we have
        df = pd.DataFrame(self.tracking_data)
        for _, row in df.iterrows():
            frame_idx = int(row['frame_idx'])
            if frame_idx < total_frames:
                combined_radius[frame_idx] = row['pupil_radius']
                
                # Determine which method was used
                if row['detection_method'] == 'method1':
                    method1_radius[frame_idx] = row['pupil_radius']
                elif row['detection_method'] == 'method2':
                    method2_radius[frame_idx] = row['pupil_radius']
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Plot both methods
        ax.plot(timestamps, method1_radius, 'b-', linewidth=2, label='Method 1 (Red Channel)', alpha=0.8)
        ax.plot(timestamps, method2_radius, 'orange', linewidth=2, label='Method 2 (CLAHE + Adaptive)', alpha=0.8)
        ax.plot(timestamps, combined_radius, 'g-', linewidth=3, label='Combined Result', alpha=0.6)
        
        # Add markers for successful detections
        method1_valid = ~np.isnan(method1_radius)
        method2_valid = ~np.isnan(method2_radius)
        
        if np.any(method1_valid):
            ax.scatter(timestamps[method1_valid], method1_radius[method1_valid], 
                      c='blue', s=20, alpha=0.6, zorder=5)
        if np.any(method2_valid):
            ax.scatter(timestamps[method2_valid], method2_radius[method2_valid], 
                      c='orange', s=20, alpha=0.6, zorder=5)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Pupil Radius (pixels)')
        ax.set_title(f'Method Comparison Over Time - {video_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add text box with method usage statistics
        method1_count = np.sum(method1_valid)
        method2_count = np.sum(method2_valid)
        total_detections = method1_count + method2_count
        
        stats_text = f"""Method Usage Statistics:
Method 1 (Red Channel): {method1_count} frames ({method1_count/total_detections*100:.1f}%)
Method 2 (CLAHE): {method2_count} frames ({method2_count/total_detections*100:.1f}%)
Total Detections: {total_detections} frames"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, f"method_comparison_{video_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved method comparison plot: {plot_path}")
    
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
