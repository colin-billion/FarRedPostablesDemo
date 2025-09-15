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
import co.billionlabs.farredpostablesdemo.utils.ScreenController
import kotlinx.coroutines.delay

class MainActivity : ComponentActivity() {
    
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
        
        // Request camera permissions
        requestCameraPermissions()
        
        setContent {
            FarRedPostablesDemoTheme {
                VideoRecordingScreen()
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
fun VideoRecordingScreen() {
    val context = LocalContext.current
    val activity = context as ComponentActivity
    val screenController = remember { ScreenController(activity) }
    
    var videoPath by remember { mutableStateOf<String?>(null) }
    var isRecording by remember { mutableStateOf(false) }
    var isInSequence by remember { mutableStateOf(false) }
    var currentBackgroundColor by remember { mutableStateOf(Color.Red) }
    var sequencePhase by remember { mutableStateOf("") }
    
    // Initialize screen settings
    LaunchedEffect(Unit) {
        screenController.saveCurrentBrightness()
        screenController.setMinimumBrightness()
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
    
    // Handle the brightness/color sequence
    LaunchedEffect(isInSequence) {
        if (isInSequence) {
            // Phase 1: 1 second dim red (already set)
            sequencePhase = "Dim Red (1s)"
            currentBackgroundColor = Color.Red
            delay(1000)
            
            // Phase 2: 2 seconds bright white
            sequencePhase = "Bright White (2s)"
            currentBackgroundColor = Color.White
            screenController.setMaximumBrightness()
            screenController.setBackgroundColor(Color.White)
            delay(2000)
            
            // Phase 3: 2 seconds dim red
            sequencePhase = "Dim Red (2s)"
            currentBackgroundColor = Color.Red
            screenController.setMinimumBrightness()
            screenController.setBackgroundColor(Color.Red)
            delay(2000)
            
            // Reset
            sequencePhase = ""
            isInSequence = false
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
                
                Spacer(modifier = Modifier.height(16.dp))
                
                when {
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
                },
                onRecordingStateChanged = { recording ->
                    isRecording = recording
                    if (recording) {
                        isInSequence = true
                    }
                },
                modifier = Modifier.fillMaxSize()
            )
        }
    }
}