# Smartphone Pupillometry Application

A complete Android application for recording and analyzing pupil size data using the front-facing camera. The app records a 5-second video, processes it using Python-based computer vision algorithms, and displays real-time pupil size measurements.

## 🎯 Features

- **Front-facing Camera Recording**: 5-second video recording with live preview
- **On-device Processing**: Python-based pupil detection and tracking using Chaquopy
- **Real-time Visualization**: Interactive charts showing pupil size over time
- **Modern UI**: Clean, intuitive interface with Material Design 3
- **Statistics Display**: Comprehensive analysis including mean, std, range, and duration

## 🏗️ Architecture

### Android Components
- **MainActivity**: Main entry point with permission handling
- **PupilTrackingViewModel**: Business logic and Python integration
- **CameraPreview**: Camera2 API integration for video recording
- **PupilSizeChart**: Custom chart component for data visualization
- **ProcessingIndicator**: Animated loading indicator

### Python Pipeline
- **pupil_processor_android.py**: Android-compatible wrapper for the existing pipeline
- **video_pupil_tracker.py**: Core pupil detection algorithm
- **pupil_post_processor.py**: Data filtering and smoothing
- **pupil_tracking_pipeline.py**: Complete end-to-end processing

## 📱 User Interface

The app features a clean, modern interface with:

- **Top Third**: White background area displaying:
  - Processing indicator during analysis
  - Interactive pupil size chart with statistics
  - Real-time data visualization

- **Bottom Two Thirds**: Camera preview with:
  - Live front-facing camera feed
  - Large red record button
  - 5-second countdown timer
  - Automatic recording stop

## 🔧 Technical Implementation

### Camera Integration
- Uses CameraX for modern camera API
- Front-facing camera with HD quality recording
- Automatic 5-second recording duration
- MP4 file output to device storage

### Python Integration (Chaquopy)
- **Python 3.11** runtime embedded in Android
- **OpenCV 4.8.1** for computer vision
- **NumPy, Pandas** for data processing
- **Matplotlib, SciPy** for analysis
- **scikit-learn, filterpy** for advanced filtering

### Pupil Detection Algorithm
1. **Iris Detection**: Hough circle detection with intensity scoring
2. **Pupil Detection**: Dark pixel clustering within iris boundaries
3. **Quality Control**: Adaptive thresholding and outlier detection
4. **Post-processing**: Kalman filtering and smoothing

## 📊 Data Processing Pipeline

1. **Video Recording**: 5-second MP4 file creation
2. **Frame Processing**: Extract and analyze each frame
3. **Pupil Detection**: Identify iris and pupil boundaries
4. **Data Filtering**: Remove outliers and apply smoothing
5. **Visualization**: Generate interactive charts and statistics

## 🚀 Getting Started

### Prerequisites
- Android Studio Arctic Fox or later
- Android SDK 35+ (Android 15+)
- Device with front-facing camera

### Installation
1. Clone the repository
2. Open in Android Studio
3. Sync Gradle dependencies
4. Build and run on device

### First Run
1. Grant camera and storage permissions
2. Position face in front of camera
3. Tap the red record button
4. Wait for 5-second recording
5. View pupil size analysis results

## 📈 Output Data

The app provides comprehensive pupil analysis:

- **Pupil Radius**: Smoothed radius measurements in pixels
- **Pupil Diameter**: Calculated diameter values
- **Pupil Area**: Area calculations in square pixels
- **Statistics**: Mean, standard deviation, min/max values
- **Duration**: Total recording time
- **Frame Count**: Number of processed frames

## 🔬 Scientific Applications

This app is designed for:
- **Pupillometry Research**: Academic and clinical studies
- **Cognitive Load Assessment**: Mental workload measurement
- **Medical Diagnostics**: Neurological condition monitoring
- **Human-Computer Interaction**: Attention and engagement analysis

## 🛠️ Customization

### Python Parameters
Modify `pupil_processor_android.py` to adjust:
- Frame processing interval
- Outlier detection thresholds
- Smoothing algorithms
- Quality control parameters

### UI Customization
- Chart colors and styling in `PupilSizeChart.kt`
- Recording duration in `CameraPreview.kt`
- Layout proportions in `MainActivity.kt`

## 📱 Device Requirements

- **Android 15+** (API level 35+)
- **Front-facing camera** with HD recording capability
- **4GB+ RAM** recommended for smooth processing
- **Storage space** for video files and Python libraries

## 🔒 Privacy & Security

- **Local Processing**: All analysis performed on-device
- **No Network**: No data transmitted to external servers
- **Temporary Storage**: Video files stored locally
- **User Control**: Full control over recorded data

## 🐛 Troubleshooting

### Common Issues
1. **Camera Permission Denied**: Grant camera permissions in device settings
2. **Python Initialization Failed**: Ensure device has sufficient RAM
3. **Processing Timeout**: Reduce video quality or frame rate
4. **No Pupil Detected**: Ensure good lighting and clear eye visibility

### Performance Optimization
- Close other apps before recording
- Ensure good lighting conditions
- Keep device stable during recording
- Use devices with 4GB+ RAM for best performance

## 📚 Dependencies

### Android
- CameraX 1.3.1
- Compose BOM 2024.09.00
- Material Design 3
- MPAndroidChart 3.1.0

### Python (via Chaquopy)
- opencv-python 4.8.1.78
- numpy 1.24.3
- pandas 2.0.3
- matplotlib 3.7.2
- scipy 1.11.1
- scikit-learn 1.3.0
- filterpy 1.4.5

## 🤝 Contributing

This project is designed for research and educational purposes. Contributions are welcome for:
- Algorithm improvements
- UI/UX enhancements
- Performance optimizations
- Additional analysis features

## 📄 License

This project is intended for research and educational use. Please ensure compliance with local regulations regarding medical device software and data privacy.

## 🔬 Research Applications

The app is particularly suitable for:
- **Pupillometry Studies**: Academic research on pupil dynamics
- **Cognitive Load Assessment**: Measuring mental workload
- **Medical Research**: Neurological condition monitoring
- **HCI Research**: Attention and engagement analysis
- **Educational Demos**: Teaching computer vision concepts

## 📞 Support

For technical support or questions about the implementation, please refer to the code documentation and inline comments throughout the codebase.
