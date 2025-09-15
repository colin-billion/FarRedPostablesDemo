package co.billionlabs.farredpostablesdemo.ui.components

import android.content.Context
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
    
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    
    DisposableEffect(Unit) {
        onDispose {
            cameraExecutor.shutdown()
        }
    }
    
    // Handle recording countdown
    LaunchedEffect(shouldStartRecording) {
        if (shouldStartRecording) {
            isRecording = true
            recordingTime = 0
            onRecordingStateChanged(true)
            
            repeat(5) {
                delay(1000)
                recordingTime++
            }
            
            // Stop recording after 5 seconds
            stopRecording(recording) {
                isRecording = false
                recordingTime = 0
                onRecordingStateChanged(false)
            }
            shouldStartRecording = false
        }
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
                    startCamera(context, lifecycleOwner, previewView, cameraProvider!!, cameraExecutor) { videoCap ->
                        videoCapture = videoCap
                    }
                }
            }
        )
        
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
                        if (!isRecording) {
                            startRecording(
                                context,
                                videoCapture,
                                onVideoRecorded
                            ) { recordingRef ->
                                recording = recordingRef
                                shouldStartRecording = true
                            }
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
    onVideoCaptureReady: (VideoCapture<Recorder>) -> Unit
) {
    val preview = Preview.Builder().build()
    val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
    
    val recorder = Recorder.Builder()
        .setQualitySelector(QualitySelector.from(Quality.HD))
        .build()
    val videoCapture = VideoCapture.withOutput(recorder)
    
    try {
        cameraProvider.unbindAll()
        cameraProvider.bindToLifecycle(
            lifecycleOwner,
            cameraSelector,
            preview,
            videoCapture
        )
        
        preview.setSurfaceProvider(previewView.surfaceProvider)
        onVideoCaptureReady(videoCapture)
        
    } catch (exc: Exception) {
        Log.e("CameraPreview", "Use case binding failed", exc)
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
                    Log.d("CameraPreview", "Recording started")
                    // Note: recording is not yet available in this callback
                }
                is VideoRecordEvent.Finalize -> {
                    Log.d("CameraPreview", "Recording finalized: ${recordEvent.outputResults.outputUri}")
                    onVideoRecorded(outputFile.absolutePath)
                }
                is VideoRecordEvent.Status -> {
                    Log.d("CameraPreview", "Recording status: ${recordEvent.recordingStats.recordedDurationNanos}")
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
    val storageDir = File(context.getExternalFilesDir(null), "PupilVideos")
    if (!storageDir.exists()) {
        storageDir.mkdirs()
    }
    return File(storageDir, fileName)
}
