#!/usr/bin/env python3
"""
Simple Pupil Size Filter

This module provides focused post-processing for pupil tracking data:
- Load pupil tracking CSV data
- Apply filtering techniques to pupil size measurements
- Create one clean, filtered graph of pupil size over time
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy import signal
from scipy.stats import zscore
from filterpy.kalman import KalmanFilter
import warnings
warnings.filterwarnings('ignore')

class PupilPostProcessor:
    """Simple filter for pupil size data with clean visualization"""
    
    def __init__(self, csv_path, output_dir=None):
        """Initialize the pupil filter with tracking data"""
        self.csv_path = csv_path
        self.output_dir = output_dir or os.path.dirname(csv_path)
        self.video_name = Path(csv_path).stem.replace('pupil_data_clean_', '')
        
        # Load data
        self.df = self.load_data()
        self.original_df = self.df.copy()
        
        print(f"Loaded pupil data: {len(self.df)} frames")
        print(f"Video: {self.video_name}")
        print(f"Duration: {self.df['timestamp'].iloc[-1]:.2f} seconds")
    
    def load_data(self):
        """Load pupil tracking data from CSV"""
        try:
            df = pd.read_csv(self.csv_path)
            print(f"Successfully loaded data from {self.csv_path}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    
    def detect_outliers(self, method='pixel_threshold', threshold=20):
        """Detect outliers in pupil radius measurements"""
        print(f"\nDetecting outliers using {method} method...")
        
        if method == 'pixel_threshold':
            # Calculate differences from neighboring points
            radius_values = self.df['pupil_radius'].values
            outliers = np.zeros(len(radius_values), dtype=bool)
            
            for i in range(len(radius_values)):
                # Check difference from previous point
                if i > 0:
                    diff_prev = abs(radius_values[i] - radius_values[i-1])
                    if diff_prev > threshold:
                        outliers[i] = True
                        continue
                
                # Check difference from next point
                if i < len(radius_values) - 1:
                    diff_next = abs(radius_values[i] - radius_values[i+1])
                    if diff_next > threshold:
                        outliers[i] = True
                        continue
            
        elif method == 'iqr':
            Q1 = self.df['pupil_radius'].quantile(0.25)
            Q3 = self.df['pupil_radius'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (self.df['pupil_radius'] < lower_bound) | (self.df['pupil_radius'] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(zscore(self.df['pupil_radius']))
            outliers = z_scores > threshold
            
        elif method == 'modified_zscore':
            median = np.median(self.df['pupil_radius'])
            mad = np.median(np.abs(self.df['pupil_radius'] - median))
            modified_z_scores = 0.6745 * (self.df['pupil_radius'] - median) / mad
            outliers = np.abs(modified_z_scores) > threshold
        
        self.outlier_mask = outliers
        print(f"Detected {outliers.sum()} outliers ({outliers.sum()/len(self.df)*100:.1f}%)")
        if method == 'pixel_threshold':
            print(f"  Using {threshold} pixel threshold from neighboring points")
        
        return outliers
    
    def filter_outliers(self, method='pixel_threshold', threshold=20, replace_with_interpolation=True):
        """Filter outliers from pupil radius data"""
        if not hasattr(self, 'outlier_mask'):
            self.detect_outliers(method, threshold)
        
        if replace_with_interpolation:
            # Replace outliers with interpolated values
            self.df.loc[self.outlier_mask, 'pupil_radius'] = np.nan
            self.df['pupil_radius'] = self.df['pupil_radius'].interpolate(method='linear')
            print("Outliers replaced with interpolated values")
        else:
            # Remove outlier rows entirely
            self.df = self.df[~self.outlier_mask].reset_index(drop=True)
            print(f"Removed {self.outlier_mask.sum()} outlier frames")
    
    def smooth_data(self, method='savgol', window_length=None, polyorder=3):
        """Apply smoothing to pupil radius data"""
        print(f"\nApplying {method} smoothing...")
        
        if window_length is None:
            # Auto-determine window length
            window_length = min(21, len(self.df) // 10 * 2 + 1)
            if window_length < 5:
                window_length = 5
        
        if method == 'savgol':
            # Savitzky-Golay filter
            if window_length >= len(self.df):
                window_length = len(self.df) - 1 if len(self.df) % 2 == 0 else len(self.df)
            if window_length < polyorder + 1:
                polyorder = max(1, window_length - 1)
            
            self.df['pupil_radius_smoothed'] = signal.savgol_filter(
                self.df['pupil_radius'], window_length, polyorder
            )
            
        elif method == 'moving_average':
            # Moving average
            self.df['pupil_radius_smoothed'] = self.df['pupil_radius'].rolling(
                window=window_length, center=True
            ).mean()
            
        elif method == 'gaussian':
            # Gaussian filter
            self.df['pupil_radius_smoothed'] = signal.gaussian_filter1d(
                self.df['pupil_radius'], sigma=window_length/3
            )
            
        elif method == 'kalman':
            # Kalman filter
            self.df['pupil_radius_smoothed'] = self.apply_kalman_filter()
        
        print(f"Applied {method} smoothing with window length {window_length}")
    
    def apply_kalman_filter(self):
        """Apply Kalman filter to pupil radius data"""
        print("  Setting up Kalman filter...")
        
        # Initialize Kalman filter
        kf = KalmanFilter(dim_x=3, dim_z=1)  # 3 states: [radius, velocity, acceleration], 1 measurement: radius
        
        # State transition matrix (how state evolves over time)
        dt = 1.0 / 30.0  # Assuming 30 FPS
        kf.F = np.array([
            [1, dt, 0.5*dt*dt],  # radius = radius + velocity*dt + 0.5*acceleration*dt^2
            [0, 1, dt],           # velocity = velocity + acceleration*dt
            [0, 0, 1]             # acceleration = acceleration (constant)
        ])
        
        # Measurement matrix (what we observe)
        kf.H = np.array([[1, 0, 0]])  # We only observe radius
        
        # Process noise covariance (how much we trust our model)
        process_noise = 0.1
        kf.Q = np.array([
            [dt**4/4, dt**3/2, dt**2/2],
            [dt**3/2, dt**2, dt],
            [dt**2/2, dt, 1]
        ]) * process_noise
        
        # Measurement noise covariance (how noisy our measurements are)
        measurement_noise = 5.0  # pixels
        kf.R = np.array([[measurement_noise]])
        
        # Initial state and covariance
        initial_radius = self.df['pupil_radius'].iloc[0]
        kf.x = np.array([initial_radius, 0, 0])  # [radius, velocity, acceleration]
        kf.P = np.eye(3) * 100  # Initial uncertainty
        
        # Apply Kalman filter
        filtered_data = []
        for radius in self.df['pupil_radius']:
            kf.predict()
            kf.update(radius)
            filtered_data.append(kf.x[0])  # Extract radius from state
        
        print(f"  Kalman filter applied with process noise {process_noise}, measurement noise {measurement_noise}")
        return np.array(filtered_data)
    
    def create_filtered_plot(self, save_path=None):
        """Create a clean, filtered plot of pupil size over time"""
        print("\nCreating filtered pupil size plot...")
        
        # Set up the plot
        plt.figure(figsize=(12, 6))
        
        # Plot original data
        plt.plot(self.df['timestamp'], self.df['pupil_radius'], 
                color='lightblue', alpha=0.6, linewidth=1, label='Original')
        
        # Plot filtered data if available
        if 'pupil_radius_smoothed' in self.df.columns:
            plt.plot(self.df['timestamp'], self.df['pupil_radius_smoothed'], 
                    color='darkblue', linewidth=2, label='Filtered')
        
        # Highlight outliers if they exist
        if hasattr(self, 'outlier_mask') and self.outlier_mask.any():
            outlier_times = self.original_df.loc[self.outlier_mask, 'timestamp']
            outlier_radii = self.original_df.loc[self.outlier_mask, 'pupil_radius']
            plt.scatter(outlier_times, outlier_radii, color='red', s=20, alpha=0.7, 
                       label=f'Outliers ({self.outlier_mask.sum()})', zorder=5)
        
        # Formatting
        plt.xlabel('Time (seconds)')
        plt.ylabel('Pupil Radius (pixels)')
        plt.title(f'Filtered Pupil Size - {self.video_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"pupil_size_filtered_{self.video_name}.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Filtered plot saved: {save_path}")
        
        return save_path
    
    def save_filtered_data(self, save_path=None):
        """Save filtered data to CSV"""
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"pupil_data_filtered_{self.video_name}.csv")
        
        self.df.to_csv(save_path, index=False)
        print(f"Filtered data saved: {save_path}")
        return save_path
    
    def run_filtering(self, outlier_method='pixel_threshold', outlier_threshold=20, 
                     smoothing_method='kalman', smoothing_window=None):
        """Run complete filtering process"""
        print(f"\n{'='*50}")
        print(f"RUNNING PUPIL SIZE FILTERING")
        print(f"{'='*50}")
        
        # 1. Detect and filter outliers
        self.detect_outliers(method=outlier_method, threshold=outlier_threshold)
        self.filter_outliers(method=outlier_method, threshold=outlier_threshold)
        
        # 2. Apply smoothing
        self.smooth_data(method=smoothing_method, window_length=smoothing_window)
        
        # 3. Create filtered plot
        self.create_filtered_plot()
        
        # 4. Save filtered data
        self.save_filtered_data()
        
        print(f"\n{'='*50}")
        print(f"FILTERING COMPLETE!")
        print(f"{'='*50}")
        print(f"Results saved to: {self.output_dir}")
        print(f"Video: {self.video_name}")
        print(f"Frames processed: {len(self.df)}")
        
        return self.df

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Filter pupil size data and create clean plot')
    parser.add_argument('csv_path', help='Path to pupil tracking CSV file')
    parser.add_argument('--output_dir', help='Output directory for results')
    parser.add_argument('--outlier_method', choices=['pixel_threshold', 'iqr', 'zscore', 'modified_zscore'], 
                       default='pixel_threshold', help='Outlier detection method')
    parser.add_argument('--outlier_threshold', type=float, default=20, 
                       help='Outlier detection threshold (pixels for pixel_threshold, multiplier for others)')
    parser.add_argument('--smoothing_method', choices=['savgol', 'moving_average', 'gaussian', 'kalman'], 
                       default='kalman', help='Smoothing method')
    parser.add_argument('--smoothing_window', type=int, help='Smoothing window length')
    parser.add_argument('--no_filtering', action='store_true', help='Skip outlier filtering')
    parser.add_argument('--no_smoothing', action='store_true', help='Skip smoothing')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return 1
    
    try:
        # Initialize processor
        processor = PupilPostProcessor(args.csv_path, args.output_dir)
        
        if args.no_filtering and args.no_smoothing:
            # Just create basic plot
            processor.create_filtered_plot()
        else:
            # Run full filtering
            processor.run_filtering(
                outlier_method=args.outlier_method,
                outlier_threshold=args.outlier_threshold,
                smoothing_method=args.smoothing_method,
                smoothing_window=args.smoothing_window
            )
        
        print(f"\n🎯 Filtering complete!")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
