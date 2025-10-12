package co.billionlabs.farredpostablesdemo

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import co.billionlabs.farredpostablesdemo.ui.theme.FarRedPostablesDemoTheme
import co.billionlabs.farredpostablesdemo.ui.components.CameraPreview
import co.billionlabs.farredpostablesdemo.ui.components.PupilDataDialog
import co.billionlabs.farredpostablesdemo.ui.components.enableFlash
import co.billionlabs.farredpostablesdemo.ui.components.disableFlash
import co.billionlabs.farredpostablesdemo.utils.ScreenController
import co.billionlabs.farredpostablesdemo.utils.PythonHelper
import co.billionlabs.farredpostablesdemo.utils.PupilTrackingHelper
import kotlinx.coroutines.delay
import java.io.File

class MainActivity : ComponentActivity() {
    
    private var pupilHelper: PupilTrackingHelper? = null
    
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.values.all { it }
        if (!allGranted) {
            Log.e("MainActivity", "Some permissions denied")
        } else {
            Log.d("MainActivity", "All permissions granted")
        }
    }
    
    private val requestSettingsPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (Settings.System.canWrite(this)) {
            Log.d("MainActivity", "Settings permission granted")
        } else {
            Log.e("MainActivity", "Settings permission denied")
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        
        // Initialize Python
        try {
            Log.d("MainActivity", "Starting Python initialization...")
            Log.d("MainActivity", "Python path: C:\\Users\\colin\\AppData\\Local\\Programs\\Python\\Python311\\python.exe")
            Log.d("MainActivity", "Python version: 3.11")
            
            PythonHelper.initializePython(this)
            Log.d("MainActivity", "PythonHelper.initializePython() completed")
            
            pupilHelper = PupilTrackingHelper()
            Log.d("MainActivity", "PupilTrackingHelper created successfully")
            
            Log.d("MainActivity", "Python initialized successfully")
        } catch (e: Exception) {
            Log.e("MainActivity", "Failed to initialize Python: ${e.message}")
            Log.e("MainActivity", "Exception type: ${e.javaClass.simpleName}")
            Log.e("MainActivity", "Python initialization error", e)
            pupilHelper = null
        }
        
        // Request camera permissions
        requestCameraPermissions()
        
        setContent {
            FarRedPostablesDemoTheme {
                VideoRecordingScreen(pupilHelper)
            }
        }
    }
    
    private fun requestCameraPermissions() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                Log.d("MainActivity", "Camera permission already granted")
                requestSettingsPermission()
            }
            else -> {
                Log.d("MainActivity", "Requesting camera permissions")
                requestPermissionLauncher.launch(
                    arrayOf(
                        Manifest.permission.CAMERA,
                        Manifest.permission.RECORD_AUDIO,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE
                    )
                )
            }
        }
    }
    
    private fun requestSettingsPermission() {
        if (Settings.System.canWrite(this)) {
            Log.d("MainActivity", "Settings permission already granted")
        } else {
            Log.d("MainActivity", "Requesting settings permission")
            val intent = android.content.Intent(Settings.ACTION_MANAGE_WRITE_SETTINGS)
            requestSettingsPermissionLauncher.launch(intent)
        }
    }
}

@Composable
fun VideoRecordingScreen(pupilHelper: PupilTrackingHelper?) {
    val context = LocalContext.current
    val activity = context as ComponentActivity
    val screenController = remember { ScreenController(activity) }
    
    var videoPath by remember { mutableStateOf<String?>(null) }
    var isRecording by remember { mutableStateOf(false) }
    var isInSequence by remember { mutableStateOf(false) }
    var isCountdown by remember { mutableStateOf(false) }
    var countdownTime by remember { mutableStateOf(5) }
    var currentBackgroundColor by remember { mutableStateOf(Color.Red) }
    var sequencePhase by remember { mutableStateOf("") }
    var pythonTestResult by remember { mutableStateOf<String?>(null) }
    
    // Pupil tracking states
    var pupilData by remember { mutableStateOf<List<Map<String, Any>>>(emptyList()) }
    var pupilImagePath by remember { mutableStateOf<String?>(null) }
    var showPupilDialog by remember { mutableStateOf(false) }
    var isProcessingVideo by remember { mutableStateOf(false) }
    var processingMessage by remember { mutableStateOf("") }
    
    // Camera states
    var useFrontCamera by remember { mutableStateOf(true) }
    var camera by remember { mutableStateOf<androidx.camera.core.Camera?>(null) }
    var shouldStartActualRecording by remember { mutableStateOf(false) }
    
    // Handle camera mode changes
    LaunchedEffect(useFrontCamera) {
        if (useFrontCamera) {
            // Front camera mode: Start with maximum brightness for visibility
            screenController.setMaximumBrightness()
        } else {
            // Back camera mode: Use higher brightness for better visibility
            screenController.setMaximumBrightness()
        }
    }
    
    // Initialize screen settings
    LaunchedEffect(Unit) {
        screenController.saveCurrentBrightness()
        // Start with maximum brightness for front camera visibility
        screenController.setMaximumBrightness()
        screenController.setBackgroundColor(Color.Red)
        screenController.setFullscreen()
    }
    
    // Cleanup when component is disposed
    DisposableEffect(Unit) {
        onDispose {
            screenController.restoreOriginalBrightness()
            screenController.exitFullscreen()
        }
    }
    
    // Handle countdown for front camera
    LaunchedEffect(isCountdown) {
        if (isCountdown && useFrontCamera) {
            // Dim the screen and start countdown
            screenController.setMinimumBrightness()
            countdownTime = 5
            
            repeat(5) {
                delay(1000)
                countdownTime--
            }
            
            // Start the actual recording sequence
            isCountdown = false
            isInSequence = true
            shouldStartActualRecording = true
        }
    }
    
    // Handle the brightness/color sequence
    LaunchedEffect(isInSequence, useFrontCamera) {
        if (isInSequence) {
            if (useFrontCamera) {
                // Front camera mode: Use screen lighting
                // Phase 1: 1 second dim red (already set)
                sequencePhase = "Dim Red (1s) - Front Camera"
                currentBackgroundColor = Color.Red
                screenController.setMinimumBrightness() // Keep minimum for dim red
                delay(500)
                
                // Phase 2: 1 seconds bright white
                sequencePhase = "Bright White (1s) - Front Camera"
                currentBackgroundColor = Color.White
                screenController.setMaximumBrightness()
                screenController.setBackgroundColor(Color.White)
                delay(1500)
                
                // Phase 3: 3 seconds dim red
                sequencePhase = "Dim Red (3s) - Front Camera"
                currentBackgroundColor = Color.Red
                screenController.setMinimumBrightness()
                screenController.setBackgroundColor(Color.Red)
                delay(3000)
            } else {
                // Back camera mode: Use flash with high screen brightness
                // Phase 1: 1 second no flash (keep high brightness)
                sequencePhase = "No Flash (1s) - Back Camera"
                currentBackgroundColor = Color.Red
                screenController.setMaximumBrightness() // Keep high brightness for visibility
                disableFlash(camera)
                delay(500)
                
                // Phase 2: 1 seconds with flash
                sequencePhase = "Flash On (1s) - Back Camera"
                currentBackgroundColor = Color.White
                screenController.setMaximumBrightness() // Keep high brightness
                enableFlash(camera)
                delay(1500)
                
                // Phase 3: 3 seconds no flash
                sequencePhase = "No Flash (3s) - Back Camera"
                currentBackgroundColor = Color.Red
                screenController.setMaximumBrightness() // Keep high brightness
                disableFlash(camera)
                delay(3000)
            }
            
            // Reset
            sequencePhase = ""
            isInSequence = false
            shouldStartActualRecording = false
        }
    }
    
    Box(modifier = Modifier.fillMaxSize()) {
        // Top third - Dynamic background with status
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(200.dp)
                .background(currentBackgroundColor)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Text(
                    text = "Pupil Video Recording",
                    style = MaterialTheme.typography.headlineMedium,
                    color = if (currentBackgroundColor == Color.White) Color.Black else Color.White
                )
                
                Text(
                    text = if (useFrontCamera) "Front Camera (Screen Light)" else "Back Camera (Flash + Bright Screen)",
                    style = MaterialTheme.typography.bodyMedium,
                    color = if (currentBackgroundColor == Color.White) Color.Black else Color.White
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                Button(
                    onClick = {
                        pythonTestResult = pupilHelper?.testPythonIntegration() 
                            ?: "Python not initialized"
                    },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = if (currentBackgroundColor == Color.White) Color.Blue else Color.Green
                    )
                ) {
                    Text(
                        text = if (pupilHelper != null) "Test Python" else "Python Not Available",
                        color = Color.White
                    )
                }
                
                pythonTestResult?.let { result ->
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = result,
                        style = MaterialTheme.typography.bodySmall,
                        color = if (currentBackgroundColor == Color.White) Color.Black else Color.White,
                        modifier = Modifier.padding(horizontal = 16.dp)
                    )
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                when {
                    isCountdown -> {
                        Text(
                            text = "Get Ready!",
                            style = MaterialTheme.typography.bodyLarge,
                            color = Color.White
                        )
                        Text(
                            text = "Recording starts in: $countdownTime",
                            style = MaterialTheme.typography.titleLarge,
                            color = Color.White
                        )
                    }
                    isInSequence -> {
                        Text(
                            text = sequencePhase,
                            style = MaterialTheme.typography.bodyLarge,
                            color = if (currentBackgroundColor == Color.White) Color.Black else Color.White
                        )
                        Text(
                            text = "Recording in progress...",
                            style = MaterialTheme.typography.bodyMedium,
                            color = if (currentBackgroundColor == Color.White) Color.Black else Color.White
                        )
                    }
                    isRecording -> {
                        Text(
                            text = "Recording... (5 seconds)",
                            style = MaterialTheme.typography.bodyLarge,
                            color = if (currentBackgroundColor == Color.White) Color.Black else Color.White
                        )
                    }
                    isProcessingVideo -> {
                        Text(
                            text = processingMessage,
                            style = MaterialTheme.typography.bodyLarge,
                            color = Color.Yellow
                        )
                        Text(
                            text = "Please wait...",
                            style = MaterialTheme.typography.bodySmall,
                            color = Color.Gray
                        )
                    }
                    videoPath != null -> {
                        Text(
                            text = "Video saved successfully!",
                            style = MaterialTheme.typography.bodyLarge,
                            color = Color.Green
                        )
                        Text(
                            text = "Path: ${videoPath?.substringAfterLast("/")}",
                            style = MaterialTheme.typography.bodySmall,
                            color = Color.Gray
                        )
                        if (processingMessage.isNotEmpty()) {
                            Text(
                                text = processingMessage,
                                style = MaterialTheme.typography.bodySmall,
                                color = Color.Red
                            )
                        }
                    }
                    else -> {
                        Text(
                            text = "Tap the red button to start recording",
                            style = MaterialTheme.typography.bodyLarge,
                            color = Color.White
                        )
                    }
                }
            }
        }
        
        // Bottom two thirds - Camera preview
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .fillMaxHeight()
                .padding(top = 200.dp)
        ) {
            CameraPreview(
                onVideoRecorded = { path ->
                    videoPath = path
                    isRecording = false
                    Log.d("MainActivity", "Video recorded: $path")
                    
                    // Process the video with pupil tracking
                    if (pupilHelper != null) {
                        processVideoForPupilTracking(
                            path, 
                            pupilHelper!!,
                            onProcessingComplete = { data, imagePath ->
                                pupilData = data
                                pupilImagePath = imagePath
                                showPupilDialog = true
                                isProcessingVideo = false
                                processingMessage = ""
                            },
                            onProcessingError = { error ->
                                processingMessage = error
                                isProcessingVideo = false
                                Log.e("MainActivity", "Pupil processing error: $error")
                            }
                        )
                    }
                },
                onRecordingStateChanged = { recording ->
                    isRecording = recording
                },
                useFrontCamera = useFrontCamera,
                onCameraChanged = { newUseFrontCamera ->
                    useFrontCamera = newUseFrontCamera
                    Log.d("MainActivity", "Camera switched to: ${if (newUseFrontCamera) "Front" else "Back"}")
                },
                onCameraReady = { cam ->
                    camera = cam
                    Log.d("MainActivity", "Camera ready: ${if (useFrontCamera) "Front" else "Back"}")
                },
                shouldStartActualRecording = shouldStartActualRecording,
                onRecordButtonPressed = {
                    Log.d("MainActivity", "Record button pressed")
                    if (useFrontCamera) {
                        // Front camera: Start countdown first
                        isCountdown = true
                    } else {
                        // Back camera: Start sequence immediately
                        isInSequence = true
                        shouldStartActualRecording = true
                    }
                },
                onRecordingCompleted = {
                    Log.d("MainActivity", "Recording completed - restoring brightness")
                    // Restore maximum brightness after recording
                    screenController.setMaximumBrightness()
                    // Immediately show processing UI
                    isProcessingVideo = true
                    processingMessage = "Processing video for pupil tracking..."
                },
                modifier = Modifier.fillMaxSize()
            )
        }
        
        // Show pupil data dialog when processing is complete
        if (showPupilDialog) {
            PupilDataDialog(
                pupilData = pupilData,
                imagePath = pupilImagePath,
                onDismiss = {
                    showPupilDialog = false
                    pupilData = emptyList()
                    pupilImagePath = null
                }
            )
        }
    }
}

/**
 * Process video for pupil tracking in a background coroutine
 */
private fun processVideoForPupilTracking(
    videoPath: String,
    pupilHelper: PupilTrackingHelper,
    onProcessingComplete: (List<Map<String, Any>>, String?) -> Unit = { _, _ -> },
    onProcessingError: (String) -> Unit = {}
) {
    // This would be called from a coroutine scope in a real implementation
    // For now, we'll handle it synchronously in the UI thread
    try {
        Log.d("MainActivity", "Starting pupil tracking processing for: $videoPath")
        
        // Process the video
        val result = pupilHelper.processVideo(videoPath)
        
        if (result["success"] == true) {
            // Extract video name from path
            val videoFile = File(videoPath)
            val videoName = videoFile.nameWithoutExtension
            
            // Get pupil data from the output directory returned by the pipeline
            val outputDir = result["outputDir"]?.toString() ?: videoFile.parent ?: ""
            val pupilResult = pupilHelper.getPupilTimeSeries(outputDir, videoName)
            val pupilData = pupilResult["pupilData"] as? List<Map<String, Any>> ?: emptyList()
            val imagePath = pupilResult["imagePath"] as? String
            val imagePathOrNull = if (imagePath.isNullOrEmpty()) null else imagePath
            
            Log.d("MainActivity", "Pupil tracking completed, found ${pupilData.size} data points")
            onProcessingComplete(pupilData, imagePathOrNull)
        } else {
            val errorMsg = result["message"]?.toString() ?: "Unknown error"
            Log.e("MainActivity", "Pupil tracking failed: $errorMsg")
            onProcessingError(errorMsg)
        }
    } catch (e: Exception) {
        Log.e("MainActivity", "Error in pupil tracking: ${e.message}", e)
        onProcessingError("Processing error: ${e.message}")
    }
}