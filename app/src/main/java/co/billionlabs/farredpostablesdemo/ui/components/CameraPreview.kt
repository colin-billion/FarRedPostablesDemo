package co.billionlabs.farredpostablesdemo.ui.components

import android.content.Context
import android.os.Environment
import android.util.Log
import android.view.ViewGroup
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import kotlinx.coroutines.delay
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

@Composable
fun CameraPreview(
    onVideoRecorded: (String) -> Unit,
    onRecordingStateChanged: (Boolean) -> Unit = {},
    useFrontCamera: Boolean = true,
    onCameraChanged: (Boolean) -> Unit = {},
    onCameraReady: (Camera?) -> Unit = {},
    shouldStartActualRecording: Boolean = false,
    onRecordButtonPressed: () -> Unit = {},
    onRecordingCompleted: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    
    var isRecording by remember { mutableStateOf(false) }
    var recordingTime by remember { mutableStateOf(0) }
    var cameraProvider by remember { mutableStateOf<ProcessCameraProvider?>(null) }
    var videoCapture by remember { mutableStateOf<VideoCapture<Recorder>?>(null) }
    var recording by remember { mutableStateOf<Recording?>(null) }
    var shouldStartRecording by remember { mutableStateOf(false) }
    var camera by remember { mutableStateOf<Camera?>(null) }
    var hasProcessedRecording by remember { mutableStateOf(false) }
    
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    
    DisposableEffect(Unit) {
        onDispose {
            // Ensure recording state is reset when component is disposed
            if (isRecording) {
                isRecording = false
                recordingTime = 0
                onRecordingStateChanged(false)
            }
            cameraExecutor.shutdown()
        }
    }
    
    // Handle actual recording start
    LaunchedEffect(shouldStartActualRecording) {
        if (shouldStartActualRecording && videoCapture != null && !hasProcessedRecording) {
            hasProcessedRecording = true
            
            // Start the actual recording
            startRecording(
                context,
                videoCapture!!,
                onVideoRecorded
            ) { recordingRef ->
                recording = recordingRef
                isRecording = true
                recordingTime = 0
                onRecordingStateChanged(true)
            }
            
            // Record for 5 seconds (1s + 1s + 3s sequence)
            repeat(5) {
                delay(1000)
                recordingTime++
            }
            
            // Stop recording after 5 seconds
            // Reset recording state immediately to prevent UI getting stuck
            isRecording = false
            recordingTime = 0
            onRecordingStateChanged(false)
            
            // Stop the actual recording
            stopRecording(recording) {
                onRecordingCompleted()
            }
            
            // Safety timeout: Force reset recording state after 2 additional seconds
            // This prevents the app from getting stuck if recording doesn't stop properly
            delay(2000)
            if (isRecording) {
                android.util.Log.w("CameraPreview", "Safety timeout: Forcing recording state reset")
                isRecording = false
                recordingTime = 0
                onRecordingStateChanged(false)
            }
        }
    }
    
    // Reset the processed flag when shouldStartActualRecording becomes false
    LaunchedEffect(shouldStartActualRecording) {
        if (!shouldStartActualRecording) {
            hasProcessedRecording = false
        }
    }
    
    // Reset recording state when camera changes
    LaunchedEffect(useFrontCamera) {
        if (isRecording) {
            android.util.Log.d("CameraPreview", "Camera changed during recording - resetting state")
            isRecording = false
            recordingTime = 0
            onRecordingStateChanged(false)
        }
        hasProcessedRecording = false
    }
    
    // Initialize camera when component is first created
    LaunchedEffect(Unit) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
        }, ContextCompat.getMainExecutor(context))
    }
    
    Box(modifier = modifier) {
        // Camera preview
        AndroidView(
            factory = { ctx ->
                PreviewView(ctx).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                    layoutParams = ViewGroup.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                }
            },
            modifier = Modifier.fillMaxSize(),
            update = { previewView ->
                // Start camera when preview is ready and camera provider is available
                if (cameraProvider != null) {
                    startCamera(context, lifecycleOwner, previewView, cameraProvider!!, cameraExecutor, useFrontCamera) { videoCap, cam ->
                        videoCapture = videoCap
                        camera = cam
                        onCameraReady(cam)
                    }
                }
            }
        )
        
        // Camera toggle button (top right)
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            contentAlignment = Alignment.TopEnd
        ) {
            IconButton(
                onClick = {
                    onCameraChanged(!useFrontCamera)
                },
                modifier = Modifier
                    .background(
                        Color.Black.copy(alpha = 0.5f),
                        CircleShape
                    )
                    .padding(8.dp)
            ) {
                Text(
                    text = if (useFrontCamera) "🔄" else "🔄",
                    style = MaterialTheme.typography.titleLarge,
                    color = Color.White
                )
            }
        }
        
        // Recording button overlay
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            contentAlignment = Alignment.BottomCenter
        ) {
            if (isRecording) {
                // Recording indicator
                Box(
                    modifier = Modifier
                        .size(80.dp)
                        .background(
                            Color.Red,
                            CircleShape
                        )
                        .border(
                            4.dp,
                            Color.White,
                            CircleShape
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = "${5 - recordingTime}s",
                        color = Color.White,
                        style = MaterialTheme.typography.titleMedium
                    )
                }
            } else {
                // Record button
                Button(
                    onClick = {
                        if (!isRecording && !shouldStartActualRecording && videoCapture != null) {
                            android.util.Log.d("CameraPreview", "Recording button pressed - preparing for recording")
                            onRecordButtonPressed()
                        }
                    },
                    modifier = Modifier
                        .size(80.dp)
                        .background(
                            Color.Red,
                            CircleShape
                        ),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color.Red
                    )
                ) {
                    Text(
                        text = "●",
                        color = Color.White,
                        style = MaterialTheme.typography.headlineLarge
                    )
                }
            }
        }
    }
}

private fun startCamera(
    context: Context,
    lifecycleOwner: LifecycleOwner,
    previewView: PreviewView,
    cameraProvider: ProcessCameraProvider,
    cameraExecutor: ExecutorService,
    useFrontCamera: Boolean,
    onVideoCaptureReady: (VideoCapture<Recorder>, Camera) -> Unit
) {
    val preview = Preview.Builder().build()
    val cameraSelector = if (useFrontCamera) {
        CameraSelector.DEFAULT_FRONT_CAMERA
    } else {
        CameraSelector.DEFAULT_BACK_CAMERA
    }
    
    val recorder = Recorder.Builder()
        .setQualitySelector(QualitySelector.from(Quality.HD))
        .build()
    val videoCapture = VideoCapture.withOutput(recorder)
    
    try {
        cameraProvider.unbindAll()
        val camera = cameraProvider.bindToLifecycle(
            lifecycleOwner,
            cameraSelector,
            preview,
            videoCapture
        )
        
        preview.setSurfaceProvider(previewView.surfaceProvider)
        onVideoCaptureReady(videoCapture, camera)
        
    } catch (exc: Exception) {
        android.util.Log.e("CameraPreview", "Use case binding failed", exc)
    }
}

private fun startRecording(
    context: Context,
    videoCapture: VideoCapture<Recorder>?,
    onVideoRecorded: (String) -> Unit,
    onRecordingStarted: (Recording) -> Unit
) {
    if (videoCapture == null) return
    
    val outputFile = createVideoFile(context)
    val outputOptions = FileOutputOptions.Builder(outputFile).build()
    
    val recording = videoCapture.output
        .prepareRecording(context, outputOptions)
        .start(ContextCompat.getMainExecutor(context)) { recordEvent ->
            when (recordEvent) {
                is VideoRecordEvent.Start -> {
                    android.util.Log.d("CameraPreview", "Recording started")
                    // Note: recording is not yet available in this callback
                }
                is VideoRecordEvent.Finalize -> {
                    android.util.Log.d("CameraPreview", "Recording finalized: ${recordEvent.outputResults.outputUri}")
                    onVideoRecorded(outputFile.absolutePath)
                }
                is VideoRecordEvent.Status -> {
                    android.util.Log.d("CameraPreview", "Recording status: ${recordEvent.recordingStats.recordedDurationNanos}")
                }
            }
        }
    
    // Call onRecordingStarted after recording is created
    onRecordingStarted(recording)
}

private fun stopRecording(recording: Recording?, onStopped: () -> Unit) {
    recording?.stop()
    onStopped()
}


private fun createVideoFile(context: Context): File {
    val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
    val fileName = "pupil_video_$timeStamp.mp4"
    val storageDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_MOVIES)

    return File(storageDir, fileName).also {
        val filePath = it.absolutePath
        android.util.Log.d("CameraPreview", "Created video file: $filePath")

        File(filePath)
    }
}

// Flash control functions
fun enableFlash(camera: Camera?) {
    camera?.let {
        val cameraControl = it.cameraControl
        cameraControl.enableTorch(true)
        android.util.Log.d("CameraPreview", "Flash enabled")
    }
}

fun disableFlash(camera: Camera?) {
    camera?.let {
        val cameraControl = it.cameraControl
        cameraControl.enableTorch(false)
        android.util.Log.d("CameraPreview", "Flash disabled")
    }
}
