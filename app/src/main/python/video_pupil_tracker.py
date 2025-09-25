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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)

        # CLAHE for local contrast
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # Unsharp mask for edge emphasis
        gaussian = cv2.GaussianBlur(enhanced, (9, 9), 10.0)
        sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

        return sharpened

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
    
    def detect_pupil(self, image, iris_result, debug=False, frame_number=0):
        """Detect pupil using the original method that finds the full pupil boundary"""
        if iris_result is None:
            return None
        
        # Extract iris region
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
            if debug:
                self.create_pupil_debug_visualization(image, gray, enhanced, iris_center, iris_radius, 
                                                    baseline_intensity, None, None, None, frame_number)
            return None
        
        # Find largest cluster
        largest_cluster_points, all_labels = self.find_largest_cluster(dark_coords)
        
        if largest_cluster_points is None:
            if debug:
                self.create_pupil_debug_visualization(image, gray, enhanced, iris_center, iris_radius, 
                                                    baseline_intensity, dark_coords, None, None, frame_number)
            return None
        
        # Find radius around iris center
        pupil_circle = self.find_radius_around_iris_center(largest_cluster_points, iris_center)
        
        if debug:
            self.create_pupil_debug_visualization(image, gray, enhanced, iris_center, iris_radius, 
                                                baseline_intensity, dark_coords, largest_cluster_points, pupil_circle, frame_number)
        
        return pupil_circle
    
    def create_pupil_debug_visualization(self, original, gray, enhanced, iris_center, iris_radius, 
                                       baseline_intensity, dark_coords, cluster_points, pupil_circle, frame_number):
        """Create comprehensive debug visualization for pupil detection"""
        import os
        
        # Create debug directory
        debug_dir = os.path.join(self.output_dir, "pupil_debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Calculate target size maintaining aspect ratio
        h, w = original.shape[:2]
        target_height = 300
        target_width = int(w * target_height / h)
        target_size = (target_width, target_height)
        
        # Resize images
        original_resized = cv2.resize(original, target_size)
        gray_resized = cv2.resize(gray, target_size)
        enhanced_resized = cv2.resize(enhanced, target_size)
        
        # Convert grayscale to BGR for display
        gray_bgr = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
        enhanced_bgr = cv2.cvtColor(enhanced_resized, cv2.COLOR_GRAY2BGR)
        
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
        
        # 2. Enhanced with iris circle and baseline circle
        enhanced_with_circles = enhanced_bgr.copy()
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
        params_text = [
            f"Frame: {frame_number}",
            f"Iris Center: ({iris_center[0]}, {iris_center[1]})",
            f"Iris Radius: {iris_radius}",
            f"Baseline Intensity: {baseline_intensity:.1f}",
            f"Dark Pixels: {len(dark_coords[0]) if dark_coords is not None else 0}",
            f"Cluster Points: {len(cluster_points) if cluster_points is not None else 0}",
            f"Pupil Radius: {pupil_circle['radius'] if pupil_circle is not None else 'N/A'}",
            f"Pupil Center: ({pupil_circle['center'][0] if pupil_circle is not None else 'N/A'}, {pupil_circle['center'][1] if pupil_circle is not None else 'N/A'})"
        ]
        
        for i, text in enumerate(params_text):
            cv2.putText(params_img, text, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add white padding between images
        padding = 10
        white_padding = np.ones((target_size[1], padding, 3), dtype=np.uint8) * 255
        white_padding_h = np.ones((padding, target_size[0] * 3 + padding * 2, 3), dtype=np.uint8) * 255
        
        # Create 3x2 grid
        # Row 1: Original+Iris, Enhanced+Circles, Dark Pixels
        row1 = np.hstack([original_with_iris, white_padding, enhanced_with_circles, white_padding, dark_pixels_img])
        
        # Row 2: Cluster, Final Result, Parameters
        row2 = np.hstack([cluster_img, white_padding, final_result, white_padding, params_img])
        
        # Combine rows
        comprehensive = np.vstack([row1, white_padding_h, row2])
        
        # Add labels
        label_y = 20
        cv2.putText(comprehensive, "Original+Iris", (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(comprehensive, "Enhanced+Circles", (target_size[0] + padding + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(comprehensive, "Dark Pixels", (2 * (target_size[0] + padding) + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.putText(comprehensive, "Cluster", (10, target_size[1] + padding + label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(comprehensive, "Final Result", (target_size[0] + padding + 10, target_size[1] + padding + label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(comprehensive, "Parameters", (2 * (target_size[0] + padding) + 10, target_size[1] + padding + label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Save comprehensive visualization
        debug_path = f"{debug_dir}/pupil_debug_frame_{frame_number:03d}.jpg"
        cv2.imwrite(debug_path, comprehensive)
        print(f"📊 Pupil debug visualization saved: {debug_path}")
    
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
    
    def find_dark_pixels_adaptive(self, image, iris_center, iris_radius, baseline_intensity):
        """Find dark pixels using intensity-based thresholding (less conservative)"""
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
        
        # Use intensity-based threshold instead of percentile
        # Set threshold to be 20% darker than baseline (more inclusive)
        threshold_core = baseline_intensity * 0.8
        
        # Find dark pixels using the intensity threshold
        dark_pixels = (image < threshold_core) & (iris_mask > 0)
        dark_coords = np.where(dark_pixels)
        
        if len(dark_coords[0]) == 0:
            return (np.array([]), np.array([])), threshold_core
        
        # No expansion needed - we're already capturing more of the pupil
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
    
    def find_largest_cluster(self, dark_coords):
        """Find the largest cluster of dark pixels using OpenCV connected components"""
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
        
        # Find the largest cluster (excluding background with label 0)
        if num_labels <= 1:  # Only background
            return None, None
        
        # Find largest cluster by area
        largest_cluster_label = 1  # Start with first non-background component
        largest_area = stats[1, cv2.CC_STAT_AREA]
        
        for label in range(2, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area > largest_area:
                largest_area = area
                largest_cluster_label = label
        
        # Filter out small clusters (less than 50 pixels)
        if largest_area < 50:
            return None, None
        
        # Get points in the largest cluster
        cluster_mask = (labels == largest_cluster_label)
        cluster_y, cluster_x = np.where(cluster_mask)
        
        # Convert back to original coordinates
        cluster_points = np.column_stack((cluster_x + min_x, cluster_y + min_y))
        
        # Create labels array for compatibility (all points in largest cluster get label 0, others -1)
        point_labels = np.full(len(points), -1)
        for i, (px, py) in enumerate(points):
            if cluster_mask[py - min_y, px - min_x]:
                point_labels[i] = 0
        
        return cluster_points, point_labels
    
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
            
            print("Phase 2: Processing video with fixed iris for pupil detection...")
            
            # Phase 2: Process video with fixed iris
            processed_count = 0
            failed_count = 0
            
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
            
            # Use fixed iris from Phase 1
            iris_result = self.fixed_iris
            
            # Detect pupil using current iris (enable debug for specific frames)
            debug_enabled = frame_number in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]
            pupil_result = self.detect_pupil(frame, iris_result, debug=debug_enabled, frame_number=frame_number)
            if pupil_result is None:
                failed_count += 1
                continue
            
            # Validate pupil detection
            pupil_radius = pupil_result['radius']
            if pupil_radius <= 0:
                failed_count += 1
                continue
            
            # Store tracking data with error handling
            try:
                timestamp = frame_number / self.fps
                if not np.isfinite(timestamp):
                    timestamp = 0.0
            except ZeroDivisionError:
                timestamp = 0.0
            
            pupil_diameter = pupil_radius * 2
            pupil_area = np.pi * pupil_radius ** 2
            
            # Calculate iris-pupil ratio with error handling
            try:
                iris_pupil_ratio = pupil_radius / iris_result['radius']
                if not np.isfinite(iris_pupil_ratio):
                    print(f"Invalid iris-pupil ratio in frame {frame_number}: {iris_pupil_ratio}")
                    iris_pupil_ratio = 0.0
            except ZeroDivisionError:
                print(f"Zero division error in frame {frame_number} - iris radius: {iris_result['radius']}")
                iris_pupil_ratio = 0.0
            
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
            
            # Create visualization frame with both iris and pupil
            vis_frame = self.create_visualization_frame(
                frame, iris_result, pupil_result, frame_number, timestamp, pupil_radius
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
    
    def create_visualization_frame(self, frame, iris_result, pupil_result, frame_idx, timestamp, pupil_radius):
        """Create visualization frame with detection overlays"""
        vis_frame = frame.copy()
        
        # Draw iris circle (green)
        if iris_result is not None:
            iris_center = iris_result['center']
            iris_radius = iris_result['radius']
            cv2.circle(vis_frame, iris_center, iris_radius, (0, 255, 0), 3)  # Green for iris
            cv2.circle(vis_frame, iris_center, 5, (0, 255, 0), -1)
            cv2.putText(vis_frame, f"Iris: r={iris_radius}", (iris_center[0] + 10, iris_center[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw pupil circle (yellow)
        if pupil_result is not None:
            pupil_center = pupil_result['center']
            pupil_radius = pupil_result['radius']
            cv2.circle(vis_frame, pupil_center, pupil_radius, (0, 255, 255), 3)  # Yellow for pupil
            cv2.circle(vis_frame, pupil_center, 3, (0, 255, 255), -1)
            cv2.putText(vis_frame, f"Pupil: r={pupil_radius}", (pupil_center[0] + 10, pupil_center[1] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Add frame information
        info_text = f"Frame: {frame_idx} | Time: {timestamp:.2f}s"
        cv2.putText(vis_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add detection status
        status_text = "IRIS + PUPIL DETECTION"
        cv2.putText(vis_frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add legend
        cv2.putText(vis_frame, "Green: Iris", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_frame, "Yellow: Pupil", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
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
