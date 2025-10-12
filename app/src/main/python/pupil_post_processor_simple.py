#!/usr/bin/env python3
"""
Simple Pupil Size Filter (Chaquopy Compatible)

This module provides focused post-processing for pupil tracking data using only
numpy, pandas, and matplotlib - compatible with Chaquopy on Android.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SimplePupilPostProcessor:
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
            # Simple z-score calculation without scipy
            mean_val = self.df['pupil_radius'].mean()
            std_val = self.df['pupil_radius'].std()
            z_scores = np.abs((self.df['pupil_radius'] - mean_val) / std_val)
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
        # Detect outliers
        outliers = self.detect_outliers(method, threshold)
        
        if not outliers.any():
            print("No outliers detected - data is clean!")
            return self.df
        
        print(f"\nFiltering {outliers.sum()} outliers...")
        
        if replace_with_interpolation:
            # Replace outliers with interpolated values
            self.df.loc[outliers, 'pupil_radius'] = np.nan
            
            # Interpolate missing values
            self.df['pupil_radius'] = self.df['pupil_radius'].interpolate(method='linear')
            
            # Handle edge cases (first/last values)
            self.df['pupil_radius'] = self.df['pupil_radius'].fillna(method='bfill').fillna(method='ffill')
            
            print("Outliers replaced with interpolated values")
        else:
            # Remove outliers entirely
            self.df = self.df[~outliers].reset_index(drop=True)
            print(f"Outliers removed - {len(self.df)} frames remaining")
        
        return self.df
    
    def smooth_data(self, method='moving_average', window_size=None):
        """Apply smoothing to pupil radius data"""
        if window_size is None:
            # Auto-determine window size (5% of data length, minimum 3, maximum 21)
            window_size = max(3, min(21, len(self.df) // 20))
            if window_size % 2 == 0:
                window_size += 1  # Make odd for better smoothing
        
        print(f"\nApplying {method} smoothing with window size {window_size}...")
        
        if method == 'moving_average':
            # Simple moving average
            self.df['pupil_radius_smoothed'] = self.df['pupil_radius'].rolling(
                window=window_size, center=True, min_periods=1
            ).mean()
            
        elif method == 'gaussian':
            # Simple Gaussian-like smoothing using weighted moving average
            weights = np.exp(-0.5 * ((np.arange(window_size) - window_size//2) / (window_size/6))**2)
            weights = weights / weights.sum()
            
            self.df['pupil_radius_smoothed'] = self.df['pupil_radius'].rolling(
                window=window_size, center=True, min_periods=1
            ).apply(lambda x: np.average(x, weights=weights), raw=True)
            
        elif method == 'savgol':
            # Savitzky-Golay filter approximation using polynomial fitting
            def savgol_approx(data, window, order=2):
                if len(data) < window:
                    return data
                
                smoothed = np.zeros_like(data)
                half_window = window // 2
                
                for i in range(len(data)):
                    start = max(0, i - half_window)
                    end = min(len(data), i + half_window + 1)
                    window_data = data[start:end]
                    
                    if len(window_data) >= order + 1:
                        # Simple polynomial fit
                        x = np.arange(len(window_data))
                        coeffs = np.polyfit(x, window_data, min(order, len(window_data)-1))
                        smoothed[i] = np.polyval(coeffs, len(window_data)//2)
                    else:
                        smoothed[i] = data[i]
                
                return smoothed
            
            self.df['pupil_radius_smoothed'] = savgol_approx(
                self.df['pupil_radius'].values, window_size, order=2
            )
        
        # Fill any remaining NaN values
        self.df['pupil_radius_smoothed'] = self.df['pupil_radius_smoothed'].fillna(self.df['pupil_radius'])
        
        print(f"Smoothing completed")
        return self.df
    
    def create_pupil_size_plot(self, save_plot=True):
        """Create a clean plot of pupil size over time"""
        print(f"\nCreating pupil size plot...")
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        
        # Plot original data (light)
        plt.plot(self.df['timestamp'], self.df['pupil_radius'], 
                alpha=0.3, color='lightblue', linewidth=1, label='Original')
        
        # Plot smoothed data (bold)
        if 'pupil_radius_smoothed' in self.df.columns:
            plt.plot(self.df['timestamp'], self.df['pupil_radius_smoothed'], 
                    color='darkblue', linewidth=2, label='Filtered')
        
        # Styling
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Pupil Radius (pixels)', fontsize=12)
        plt.title(f'Pupil Size Over Time - {self.video_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        if 'pupil_radius_smoothed' in self.df.columns:
            mean_radius = self.df['pupil_radius_smoothed'].mean()
            std_radius = self.df['pupil_radius_smoothed'].std()
            min_radius = self.df['pupil_radius_smoothed'].min()
            max_radius = self.df['pupil_radius_smoothed'].max()
        else:
            mean_radius = self.df['pupil_radius'].mean()
            std_radius = self.df['pupil_radius'].std()
            min_radius = self.df['pupil_radius'].min()
            max_radius = self.df['pupil_radius'].max()
        
        stats_text = f'Mean: {mean_radius:.1f} ± {std_radius:.1f}\nRange: {min_radius:.1f} - {max_radius:.1f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.output_dir, f'pupil_size_filtered_{self.video_name}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {plot_path}")
        
        plt.show()
        return plt.gcf()
    
    def save_filtered_data(self):
        """Save the filtered data to CSV"""
        output_path = os.path.join(self.output_dir, f'pupil_data_filtered_{self.video_name}.csv')
        
        # Prepare data for saving
        save_df = self.df.copy()
        
        # Add pupil diameter column
        if 'pupil_radius_smoothed' in save_df.columns:
            save_df['pupil_diameter'] = save_df['pupil_radius_smoothed'] * 2
        else:
            save_df['pupil_diameter'] = save_df['pupil_radius'] * 2
        
        # Reorder columns
        column_order = ['frame_idx', 'timestamp', 'pupil_radius', 'pupil_radius_smoothed', 'pupil_diameter']
        available_columns = [col for col in column_order if col in save_df.columns]
        other_columns = [col for col in save_df.columns if col not in available_columns]
        save_df = save_df[available_columns + other_columns]
        
        save_df.to_csv(output_path, index=False)
        print(f"Filtered data saved: {output_path}")
        
        return output_path
    
    def run_filtering(self, outlier_method='pixel_threshold', outlier_threshold=20,
                     smoothing_method='moving_average', smoothing_window=None):
        """Run the complete filtering pipeline"""
        print(f"\n{'='*50}")
        print(f"PUPIL DATA FILTERING")
        print(f"{'='*50}")
        
        # Step 1: Filter outliers
        self.filter_outliers(outlier_method, outlier_threshold)
        
        # Step 2: Apply smoothing
        self.smooth_data(smoothing_method, smoothing_window)
        
        # Step 3: Create visualization
        self.create_pupil_size_plot()
        
        # Step 4: Save results
        output_path = self.save_filtered_data()
        
        print(f"\n✅ Filtering completed successfully!")
        print(f"   Output: {output_path}")
        
        return output_path

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple pupil size filtering')
    parser.add_argument('csv_path', help='Path to pupil tracking CSV file')
    parser.add_argument('--output_dir', help='Output directory (default: same as CSV)')
    parser.add_argument('--outlier_method', choices=['pixel_threshold', 'iqr', 'zscore', 'modified_zscore'], 
                       default='pixel_threshold', help='Outlier detection method')
    parser.add_argument('--outlier_threshold', type=float, default=20,
                       help='Outlier detection threshold')
    parser.add_argument('--smoothing_method', choices=['moving_average', 'gaussian', 'savgol'],
                       default='moving_average', help='Smoothing method')
    parser.add_argument('--smoothing_window', type=int,
                       help='Smoothing window size (auto if not specified)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return 1
    
    try:
        processor = SimplePupilPostProcessor(args.csv_path, args.output_dir)
        processor.run_filtering(
            args.outlier_method, args.outlier_threshold,
            args.smoothing_method, args.smoothing_window
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

