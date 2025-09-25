#!/usr/bin/env python3
"""
Complete Pupil Tracking Pipeline

This script runs the complete pupil tracking pipeline:
1. Video processing with pupil detection
2. Post-processing with filtering and analysis
3. Comprehensive visualization and reporting

Usage:
    python pupil_tracking_pipeline.py input_video.mp4 [options]
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Import from the same directory
from video_pupil_tracker import CleanVideoPupilTracker
from pupil_post_processor import PupilPostProcessor

class PupilTrackingPipeline:
    """Complete pipeline for pupil tracking and analysis"""
    
    def __init__(self, video_path, output_dir="../data/output", frame_interval=1,
                 outlier_method='pixel_threshold', outlier_threshold=20,
                 smoothing_method='kalman', smoothing_window=None):
        """Initialize the complete pipeline"""
        self.video_path = video_path
        self.base_output_dir = output_dir
        self.frame_interval = frame_interval
        
        # Post-processing parameters
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.smoothing_method = smoothing_method
        self.smoothing_window = smoothing_window
        
        # Create video-specific output directory
        video_name = Path(video_path).stem
        self.output_dir = os.path.join(output_dir, video_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Pipeline state
        self.tracking_complete = False
        self.post_processing_complete = False
        self.csv_path = None
        
        print(f"🎯 Pupil Tracking Pipeline Initialized")
        print(f"   Video: {Path(video_path).name}")
        print(f"   Output: {self.output_dir}")
        print(f"   Frame interval: {frame_interval}")
    
    def check_video_orientation(self):
        """Check video orientation and print dimension info for debugging"""
        import cv2
        
        # Open video to check dimensions
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("❌ Could not open video for orientation check")
            return
        
        # Get OpenCV reported dimensions
        reported_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        reported_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Read first frame to get actual dimensions
        ret, frame = cap.read()
        if ret and frame is not None:
            actual_height, actual_width = frame.shape[:2]
            
            print(f"📐 Video dimension check:")
            print(f"   OpenCV reported: {reported_width}x{reported_height} (width x height)")
            print(f"   Actual frame: {actual_width}x{actual_height} (width x height)")
            
            if actual_width != reported_width or actual_height != reported_height:
                print(f"⚠️  Dimension mismatch detected!")
                print(f"   This may cause iris detection issues.")
                print(f"   Consider rotating the video or adjusting detection parameters.")
            else:
                print(f"✅ Dimensions match - no rotation issues detected")
            
            # Check if iris detection parameters might be too restrictive
            min_dimension = min(actual_width, actual_height)
            # Calculate dynamic parameters (same as in video_pupil_tracker.py)
            min_radius = max(20, min_dimension // 20)
            max_radius = min_dimension // 4
            min_dist = min_radius * 2
            
            print(f"📏 Iris detection parameter analysis:")
            print(f"   Video dimensions: {actual_width}x{actual_height}")
            print(f"   Dynamic minRadius={min_radius}, maxRadius={max_radius}, minDist={min_dist}")
            print(f"   Min dimension: {min_dimension}")
            
            if min_dimension < 400:
                print(f"⚠️  Video has very small dimensions - iris detection may be challenging")
            elif min_dimension < 600:
                print(f"ℹ️  Video has small dimensions - using scaled parameters")
            else:
                print(f"ℹ️  Video has good dimensions - parameters should work well")
        else:
            print("❌ Could not read first frame for dimension check")
        
        cap.release()
    
    def run_tracking(self):
        """Run the pupil tracking phase"""
        print(f"\n{'='*60}")
        print(f"PHASE 1: PUPIL TRACKING")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Check video orientation and dimensions before initializing tracker
            self.check_video_orientation()
            
            # Initialize tracker
            tracker = CleanVideoPupilTracker(
                self.video_path,
                self.output_dir,
                self.frame_interval
            )
            
            # Process video
            success = tracker.process_video()
            
            if not success:
                print("❌ Pupil tracking failed!")
                return False
            
            # Save results
            tracker.save_results()
            
            # Set CSV path for post-processing
            video_name = Path(self.video_path).stem
            self.csv_path = os.path.join(self.output_dir, f"pupil_data_clean_{video_name}.csv")
            
            elapsed_time = time.time() - start_time
            print(f"✅ Pupil tracking completed in {elapsed_time:.1f} seconds")
            print(f"   Data saved: {self.csv_path}")
            
            self.tracking_complete = True
            return True
            
        except Exception as e:
            print(f"❌ Error during pupil tracking: {e}")
            return False
    
    def run_post_processing(self):
        """Run the post-processing phase"""
        if not self.tracking_complete:
            print("❌ Must run tracking first!")
            return False
        
        if not os.path.exists(self.csv_path):
            print(f"❌ Tracking data not found: {self.csv_path}")
            return False
        
        print(f"\n{'='*60}")
        print(f"PHASE 2: POST-PROCESSING")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Initialize post-processor
            processor = PupilPostProcessor(self.csv_path, self.output_dir)
            
            # Run filtering
            processor.run_filtering(
                outlier_method=self.outlier_method,
                outlier_threshold=self.outlier_threshold,
                smoothing_method=self.smoothing_method,
                smoothing_window=self.smoothing_window
            )
            
            elapsed_time = time.time() - start_time
            print(f"✅ Post-processing completed in {elapsed_time:.1f} seconds")
            
            self.post_processing_complete = True
            return True
            
        except Exception as e:
            print(f"❌ Error during post-processing: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete pipeline from start to finish"""
        print(f"\n🚀 Starting Complete Pupil Tracking Pipeline")
        print(f"   Video: {Path(self.video_path).name}")
        print(f"   Output: {self.output_dir}")
        
        total_start_time = time.time()
        
        # Phase 1: Tracking
        if not self.run_tracking():
            print("❌ Pipeline failed at tracking phase")
            return False
        
        # Phase 2: Post-processing
        if not self.run_post_processing():
            print("❌ Pipeline failed at post-processing phase")
            return False
        
        total_elapsed_time = time.time() - total_start_time
        
        print(f"\n{'='*60}")
        print(f"🎉 PIPELINE COMPLETE!")
        print(f"{'='*60}")
        print(f"Total time: {total_elapsed_time:.1f} seconds")
        print(f"Results saved to: {self.output_dir}")
        
        # List output files
        self.list_output_files()
        
        return True
    
    def list_output_files(self):
        """List all output files created by the pipeline"""
        print(f"\n📁 OUTPUT FILES:")
        
        video_name = Path(self.video_path).stem
        
        # Expected output files
        expected_files = [
            f"pupil_data_clean_{video_name}.csv",
            f"pupil_tracking_clean_{video_name}.mp4",
            f"pupil_tracking_clean_{video_name}.png",
            f"pupil_stats_clean_{video_name}.txt",
            f"pupil_data_filtered_{video_name}.csv",
            f"pupil_size_filtered_{video_name}.png"
        ]
        
        for filename in expected_files:
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                if size > 1024*1024:
                    size_str = f"{size/(1024*1024):.1f} MB"
                elif size > 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size} bytes"
                print(f"   ✅ {filename} ({size_str})")
            else:
                print(f"   ❌ {filename} (missing)")
    
    def create_summary_report(self):
        """Create a simple summary report"""
        if not self.post_processing_complete:
            print("❌ Must complete post-processing first!")
            return False
        
        try:
            # Create simple summary
            video_name = Path(self.video_path).stem
            summary_path = os.path.join(self.output_dir, f"pipeline_summary_{video_name}.txt")
            
            with open(summary_path, 'w') as f:
                f.write(f"PUPIL TRACKING PIPELINE SUMMARY\n")
                f.write(f"Video: {video_name}\n")
                f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("PROCESSING:\n")
                f.write(f"  Outlier detection: {self.outlier_method}\n")
                f.write(f"  Outlier threshold: {self.outlier_threshold}\n")
                f.write(f"  Smoothing: {self.smoothing_method}\n")
                f.write(f"  Output: Filtered pupil size data and plot\n")
            
            print(f"📊 Summary report saved: {summary_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating summary report: {e}")
            return False

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Complete pupil tracking pipeline with post-processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python pupil_tracking_pipeline.py ../data/input/input_video.mp4
  
  # Custom output directory and frame interval
  python pupil_tracking_pipeline.py ../data/input/input_video.mp4 --output_dir ../data/output --frame_interval 2
  
  # Custom post-processing parameters
  python pupil_tracking_pipeline.py ../data/input/input_video.mp4 --outlier_method zscore --smoothing_method gaussian
  
  # Skip post-processing (tracking only)
  python pupil_tracking_pipeline.py ../data/input/input_video.mp4 --tracking_only
        """
    )
    
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output_dir', default='../data/output', 
                       help='Output directory for results (default: ../data/output)')
    parser.add_argument('--frame_interval', type=int, default=1,
                       help='Process every Nth frame (default: 1)')
    
    # Post-processing options
    parser.add_argument('--outlier_method', choices=['pixel_threshold', 'iqr', 'zscore', 'modified_zscore'], 
                       default='pixel_threshold', help='Outlier detection method (default: pixel_threshold)')
    parser.add_argument('--outlier_threshold', type=float, default=20,
                       help='Outlier detection threshold (pixels for pixel_threshold, multiplier for others) (default: 20)')
    parser.add_argument('--smoothing_method', choices=['savgol', 'moving_average', 'gaussian', 'kalman'],
                       default='kalman', help='Smoothing method (default: kalman)')
    parser.add_argument('--smoothing_window', type=int,
                       help='Smoothing window length (auto-determined if not specified)')
    
    # Pipeline control
    parser.add_argument('--tracking_only', action='store_true',
                       help='Run only pupil tracking (skip post-processing)')
    parser.add_argument('--post_processing_only', action='store_true',
                       help='Run only post-processing (assumes tracking data exists)')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        return 1
    
    if args.post_processing_only:
        # Check if tracking data exists
        video_name = Path(args.video_path).stem
        video_output_dir = os.path.join(args.output_dir, video_name)
        csv_path = os.path.join(video_output_dir, f"pupil_data_clean_{video_name}.csv")
        if not os.path.exists(csv_path):
            print(f"❌ Error: Tracking data not found: {csv_path}")
            print("   Run tracking first or use --tracking_only")
            return 1
    
    try:
        # Initialize pipeline
        pipeline = PupilTrackingPipeline(
            args.video_path,
            args.output_dir,
            args.frame_interval,
            args.outlier_method,
            args.outlier_threshold,
            args.smoothing_method,
            args.smoothing_window
        )
        
        if args.tracking_only:
            # Run only tracking
            success = pipeline.run_tracking()
        elif args.post_processing_only:
            # Run only post-processing
            pipeline.tracking_complete = True
            video_name = Path(args.video_path).stem
            pipeline.csv_path = os.path.join(pipeline.output_dir, f"pupil_data_clean_{video_name}.csv")
            success = pipeline.run_post_processing()
        else:
            # Run complete pipeline
            success = pipeline.run_complete_pipeline()
        
        if success:
            # Create summary report
            pipeline.create_summary_report()
            print(f"\n🎯 Pipeline completed successfully!")
            return 0
        else:
            print(f"\n❌ Pipeline failed!")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
