# Pupil Tracking System

Complete end-to-end system for pupil detection, tracking, and analysis from video input to filtered results.

## 🎯 System Overview

This system takes a video file and produces clean, filtered pupil size measurements with comprehensive analysis.

### **Input:** Video file (MP4)
### **Output:** 
- **Organized by video**: Each video gets its own subdirectory
- Raw pupil detection data (CSV)
- Annotated video with detection overlays (MP4)
- Basic analysis plots (PNG)
- **Filtered pupil size data (CSV)**
- **Clean, filtered visualization (PNG)**

## 📁 Files in This Directory

### **Core System Files:**

#### **1. `video_pupil_tracker.py`** - Pupil Detection Engine
- **Purpose**: Detects and tracks pupil size in video frames
- **Input**: Video file
- **Output**: 4 files per video (saved in video-specific subdirectory):
  - `pupil_data_clean_[video_name].csv` - Raw tracking data
  - `pupil_tracking_clean_[video_name].mp4` - Annotated video
  - `pupil_tracking_clean_[video_name].png` - Basic plot
  - `pupil_stats_clean_[video_name].txt` - Statistics

#### **2. `pupil_post_processor.py`** - Data Filtering & Analysis
- **Purpose**: Filters noise and creates clean visualizations
- **Input**: CSV data from pupil tracker
- **Output**: 2 additional files per video (saved in video-specific subdirectory):
  - `pupil_data_filtered_[video_name].csv` - Cleaned data
  - `pupil_size_filtered_[video_name].png` - Filtered graph

#### **3. `pupil_tracking_pipeline.py`** - Complete End-to-End Pipeline
- **Purpose**: Runs detection + post-processing in one command
- **Input**: Video file
- **Output**: All 6 files (4 from detection + 2 from filtering) organized in video-specific subdirectory


## 📂 Directory Structure

The system automatically organizes output files by video:

```
data/
├── input/                          # Input videos
│   └── your_video.mp4
└── output/                         # Output directory
    └── your_video/                 # Video-specific subdirectory
        ├── pupil_data_clean_your_video.csv
        ├── pupil_tracking_clean_your_video.mp4
        ├── pupil_tracking_clean_your_video.png
        ├── pupil_stats_clean_your_video.txt
        ├── pupil_data_filtered_your_video.csv
        ├── pupil_size_filtered_your_video.png
        └── pipeline_summary_your_video.txt
```

**Benefits:**
- ✅ Clean organization - each video has its own folder
- ✅ No file conflicts when processing multiple videos
- ✅ Easy to find results for specific videos
- ✅ Simple cleanup - just delete the video folder

## 🚀 Usage

### **Complete End-to-End Pipeline (Recommended)**
```bash
# Run everything: detection + filtering
python pupil_tracking_pipeline.py ../data/input/your_video.mp4

# With custom parameters
python pupil_tracking_pipeline.py ../data/input/your_video.mp4 --outlier_threshold 15 --smoothing_method kalman
```

### **Individual Components**
```bash
# Step 1: Pupil detection only
python video_pupil_tracker.py ../data/input/your_video.mp4

# Step 2: Post-processing only (if CSV exists)
python pupil_post_processor.py ../data/output/your_video/pupil_data_clean_your_video.csv
```


## 🔧 Filtering Options

### **Outlier Detection Methods:**
- **`pixel_threshold`** (default): Detects points that differ by >20 pixels from neighbors
- **`iqr`**: Statistical method using interquartile range
- **`zscore`**: Statistical method using z-scores
- **`modified_zscore`**: Robust statistical method

### **Smoothing Methods:**
- **`kalman`** (default): Kalman filter - physics-based, adaptive smoothing
- **`savgol`**: Savitzky-Golay filter - preserves peaks and valleys
- **`moving_average`**: Simple moving average
- **`gaussian`**: Gaussian-weighted smoothing

### **Example Commands:**
```bash
# More sensitive outlier detection
python pupil_post_processor.py data.csv --outlier_threshold 15

# Use Savitzky-Golay filter for smoothing
python pupil_post_processor.py data.csv --smoothing_method savgol

# Conservative filtering
python pupil_post_processor.py data.csv --outlier_threshold 30 --smoothing_method moving_average
```

## 📊 Output Files Explained

### **From Pupil Detection (`video_pupil_tracker.py`):**
1. **`pupil_data_clean_[video].csv`** - Raw tracking data with columns:
   - `frame_idx`, `timestamp` - Frame and time information
   - `iris_center_x/y`, `iris_radius` - Iris detection results
   - `pupil_center_x/y`, `pupil_radius` - Pupil detection results
   - `pupil_diameter`, `pupil_area` - Calculated measurements
   - `iris_pupil_ratio` - Ratio of pupil to iris size

2. **`pupil_tracking_clean_[video].mp4`** - Annotated video showing:
   - Green circle: Iris boundary
   - Yellow circle: Pupil boundary
   - Frame number and timestamp overlay

3. **`pupil_tracking_clean_[video].png`** - Basic plot showing:
   - Pupil radius over time
   - Pupil area over time

4. **`pupil_stats_clean_[video].txt`** - Basic statistics:
   - Mean, std, min, max values
   - Success rate and frame counts

### **From Post-Processing (`pupil_post_processor.py`):**
5. **`pupil_data_filtered_[video].csv`** - Cleaned data with:
   - All original columns
   - `pupil_radius_smoothed` - Filtered pupil radius values
   - Outliers replaced with interpolated values

6. **`pupil_size_filtered_[video].png`** - Clean visualization showing:
   - Light blue line: Original data
   - Dark blue line: Filtered data
   - Red dots: Outlier points that were removed

## 🎛️ System Parameters

### **Detection Parameters:**
- **`--frame_interval`**: Process every Nth frame (default: 1)
- **`--output_dir`**: Output directory (default: ../data/output)

### **Filtering Parameters:**
- **`--outlier_method`**: Detection method (default: pixel_threshold)
- **`--outlier_threshold`**: Sensitivity (default: 20 pixels)
- **`--smoothing_method`**: Smoothing algorithm (default: savgol)
- **`--smoothing_window`**: Window size (auto-determined)

## 🔬 Technical Details

### **Pupil Detection Algorithm:**
1. **Iris Detection**: Hough circles with intensity scoring
2. **Pupil Detection**: Dark pixel clustering within iris
3. **Quality Control**: Scoring and validation of detections

### **Post-Processing Pipeline:**
1. **Outlier Detection**: Compare each point to its neighbors
2. **Outlier Removal**: Replace with linear interpolation
3. **Smoothing**: Apply chosen smoothing algorithm
4. **Visualization**: Create clean, filtered plots

### **Kalman Filter Details:**
- **State**: [pupil_radius, velocity, acceleration]
- **Process Model**: Physics-based motion equations
- **Measurement Model**: Handles noisy pupil detections
- **Adaptive**: Learns from data as it processes

## 📈 Example Workflow

```bash
# 1. Process a video end-to-end
python pupil_tracking_pipeline.py ../data/input/FarRed_713.mp4

# 2. Check results (organized in video-specific folder)
ls ../data/output/FarRed_713/

# 3. Try different filtering parameters
python pupil_post_processor.py ../data/output/FarRed_713/pupil_data_clean_FarRed_713.csv --smoothing_method kalman

# 4. Compare results
# Look at ../data/output/FarRed_713/pupil_size_filtered_FarRed_713.png
```

## 🛠️ Requirements

```bash
pip install opencv-python numpy pandas matplotlib scipy scikit-learn filterpy
```

## 📝 Notes

- **Input videos** should be in `../data/input/`
- **All outputs** go to `../data/output/[video_name]/` (organized by video)
- **System works** with any MP4 video file
- **Filtering is optional** - you can use just the detection if needed
- **Parameters are tunable** - experiment with different settings for your data
- **Clean organization** - each video gets its own subdirectory automatically