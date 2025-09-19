package co.billionlabs.farredpostablesdemo.ui.components

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import android.graphics.BitmapFactory
import kotlin.math.pow

@Composable
fun PupilDataDialog(
    pupilData: List<Map<String, Any>>,
    imagePath: String? = null,
    onDismiss: () -> Unit
) {
    if (pupilData.isEmpty()) {
        AlertDialog(
            onDismissRequest = onDismiss,
            title = { Text("No Data Available") },
            text = { Text("No pupil data was found in the processed video.") },
            confirmButton = {
                TextButton(onClick = onDismiss) {
                    Text("OK")
                }
            }
        )
        return
    }

    Dialog(onDismissRequest = onDismiss) {
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .fillMaxHeight(0.8f)
                .padding(16.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp)
            ) {
                // Header
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Pupil Size Analysis",
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.Bold
                    )
                    IconButton(onClick = onDismiss) {
                        Text("✕", style = MaterialTheme.typography.headlineMedium)
                    }
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Pupil Size Chart Image
                if (imagePath != null) {
                    PupilSizeChart(imagePath = imagePath)
                } else {
                    // Fallback to statistics if no image
                    PupilStatistics(pupilData = pupilData)
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Close button
                Button(
                    onClick = onDismiss,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Close")
                }
            }
        }
    }
}

@Composable
private fun PupilSizeChart(imagePath: String) {
    val context = LocalContext.current
    var bitmap by remember { mutableStateOf<android.graphics.Bitmap?>(null) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    
    LaunchedEffect(imagePath) {
        try {
            val file = java.io.File(imagePath)
            if (file.exists()) {
                bitmap = BitmapFactory.decodeFile(imagePath)
                if (bitmap == null) {
                    errorMessage = "Failed to load image"
                }
            } else {
                errorMessage = "Image file not found: $imagePath"
            }
        } catch (e: Exception) {
            errorMessage = "Error loading image: ${e.message}"
        }
    }
    
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Pupil Size Chart",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            when {
                bitmap != null -> {
                    Image(
                        bitmap = bitmap!!.asImageBitmap(),
                        contentDescription = "Pupil Size Chart",
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(500.dp),
                        contentScale = ContentScale.Fit
                    )
                }
                errorMessage != null -> {
                    Text(
                        text = errorMessage!!,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.error
                    )
                }
                else -> {
                    Text(
                        text = "Loading chart...",
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }
        }
    }
}

@Composable
private fun PupilStatistics(pupilData: List<Map<String, Any>>) {
    if (pupilData.isEmpty()) return
    
    // Calculate statistics
    val pupilSizes = pupilData.mapNotNull { data ->
        when {
            data.containsKey("pupil_radius_smoothed") -> data["pupil_radius_smoothed"] as? Double
            data.containsKey("pupil_radius") -> data["pupil_radius"] as? Double
            data.containsKey("pupil_diameter") -> (data["pupil_diameter"] as? Double)?.div(2.0)
            else -> null
        }
    }.filter { it > 0 }
    
    if (pupilSizes.isEmpty()) return
    
    val meanSize = pupilSizes.average()
    val minSize = pupilSizes.minOrNull() ?: 0.0
    val maxSize = pupilSizes.maxOrNull() ?: 0.0
    val stdDev = kotlin.math.sqrt(pupilSizes.map { (it - meanSize).pow(2) }.average().toDouble())
    
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text(
                text = "Statistics",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                StatisticItem("Mean", String.format("%.2f", meanSize))
                StatisticItem("Min", String.format("%.2f", minSize))
                StatisticItem("Max", String.format("%.2f", maxSize))
                StatisticItem("Std Dev", String.format("%.2f", stdDev))
            }
            
            Spacer(modifier = Modifier.height(8.dp))
            
            Text(
                text = "Data Points: ${pupilData.size}",
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}

@Composable
private fun StatisticItem(label: String, value: String) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = value,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}


