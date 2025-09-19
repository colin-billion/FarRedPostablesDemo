#!/usr/bin/env python3
"""
Simple Pupil Tracking Pipeline (Chaquopy Compatible)

This script runs a simplified pupil tracking pipeline using only packages
compatible with Chaquopy on Android.
"""

import os
import sys
import time
from pathlib import Path

# Import from the same directory
from video_pupil_tracker import CleanVideoPupilTracker
from pupil_post_processor_simple import SimplePupilPostProcessor

class SimplePupilTrackingPipeline:
    """Simplified pipeline for pupil tracking and analysis"""
    
    def __init__(self, video_path, output_dir=None, frame_interval=1):
        """Initialize the simplified pipeline"""
        self.video_path = video_path
        self.frame_interval = frame_interval
        
        # Use Android app directory if no output_dir specified
        if output_dir is None:
            # Use Android app's internal directory
            # This is the standard path for Chaquopy apps
            app_files_dir = "/data/data/co.billionlabs.farredpostablesdemo/files"
            self.base_output_dir = os.path.join(app_files_dir, "pupil_output")
        else:
            self.base_output_dir = output_dir
        
        # Create video-specific output directory
        video_name = Path(video_path).stem
        self.output_dir = os.path.join(self.base_output_dir, video_name)
        
        print(f"🔧 Debug: base_output_dir = {self.base_output_dir}")
        print(f"🔧 Debug: output_dir = {self.output_dir}")
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"✅ Successfully created output directory: {self.output_dir}")
        except Exception as e:
            print(f"❌ Failed to create output directory: {e}")
            # Try alternative directory
            alt_dir = "/data/data/co.billionlabs.farredpostablesdemo/cache/pupil_output"
            self.output_dir = os.path.join(alt_dir, video_name)
            print(f"🔧 Trying alternative directory: {self.output_dir}")
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Pipeline state
        self.tracking_complete = False
        self.post_processing_complete = False
        self.csv_path = None
        
        print(f"🎯 Simple Pupil Tracking Pipeline Initialized")
        print(f"   Video: {Path(video_path).name}")
        print(f"   Output: {self.output_dir}")
        print(f"   Frame interval: {frame_interval}")
    
    def run_tracking(self):
        """Run the pupil tracking phase"""
        print(f"\n{'='*60}")
        print(f"PHASE 1: PUPIL TRACKING")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
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
        """Run the simplified post-processing phase"""
        if not self.tracking_complete:
            print("❌ Must run tracking first!")
            return False
        
        if not os.path.exists(self.csv_path):
            print(f"❌ Tracking data not found: {self.csv_path}")
            return False
        
        print(f"\n{'='*60}")
        print(f"PHASE 2: SIMPLE POST-PROCESSING")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Initialize simple post-processor
            processor = SimplePupilPostProcessor(self.csv_path, self.output_dir)
            
            # Run filtering with default parameters
            processor.run_filtering(
                outlier_method='pixel_threshold',
                outlier_threshold=20,
                smoothing_method='moving_average',
                smoothing_window=None
            )
            
            elapsed_time = time.time() - start_time
            print(f"✅ Post-processing completed in {elapsed_time:.1f} seconds")
            
            self.post_processing_complete = True
            return True
            
        except Exception as e:
            print(f"❌ Error during post-processing: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete simplified pipeline from start to finish"""
        print(f"\n🚀 Starting Simple Pupil Tracking Pipeline")
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
        print(f"🎉 SIMPLE PIPELINE COMPLETE!")
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

def main(video_path, output_dir=None):
    """Main function for Chaquopy integration"""
    try:
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file not found: {video_path}")
            return None
        
        # Check if video file is readable
        if not os.access(video_path, os.R_OK):
            print(f"❌ Error: Video file is not readable: {video_path}")
            return None
        
        print(f"📹 Processing video: {video_path}")
        
        # Initialize pipeline with Android-compatible output directory
        pipeline = SimplePupilTrackingPipeline(video_path, output_dir)
        
        print(f"📁 Output directory: {pipeline.output_dir}")
        
        # Run complete pipeline
        success = pipeline.run_complete_pipeline()
        
        if success:
            print(f"\n🎯 Simple pipeline completed successfully!")
            print(f"📁 Results saved to: {pipeline.output_dir}")
            return pipeline.output_dir
        else:
            print(f"\n❌ Simple pipeline failed!")
            return None
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pupil_tracking_pipeline_simple.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    result = main(video_path)
    sys.exit(0 if result else 1)

