package co.billionlabs.farredpostablesdemo.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import kotlin.math.pow

@Composable
fun PupilDataDialog(
    pupilData: List<Map<String, Any>>,
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
                
                // Statistics
                PupilStatistics(pupilData = pupilData)
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Data table
                Text(
                    text = "Pupil Size Over Time",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold
                )
                
                Spacer(modifier = Modifier.height(8.dp))
                
                PupilDataTable(
                    pupilData = pupilData,
                    modifier = Modifier.weight(1f)
                )
                
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

@Composable
private fun PupilDataTable(
    pupilData: List<Map<String, Any>>,
    modifier: Modifier = Modifier
) {
    val scrollState = rememberScrollState()
    
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(scrollState)
        ) {
            // Header
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(MaterialTheme.colorScheme.primaryContainer)
                    .padding(8.dp),
                horizontalArrangement = Arrangement.SpaceEvenly
            ) {
                Text("Frame", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                Text("Time (s)", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                Text("Pupil Radius", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
                Text("Diameter", style = MaterialTheme.typography.labelMedium, fontWeight = FontWeight.Bold)
            }
            
            // Data rows (show first 50 rows to avoid performance issues)
            pupilData.take(50).forEach { data ->
                val frameIdx = data["frame_idx"]?.toString() ?: "N/A"
                val timestamp = data["timestamp"]?.let { 
                    if (it is Double) String.format("%.2f", it) else it.toString() 
                } ?: "N/A"
                val pupilRadius = data["pupil_radius"]?.let {
                    if (it is Double) String.format("%.2f", it) else it.toString()
                } ?: "N/A"
                val pupilDiameter = data["pupil_diameter"]?.let {
                    if (it is Double) String.format("%.2f", it) else it.toString()
                } ?: "N/A"
                
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(
                            if (data == pupilData.first()) Color.Transparent 
                            else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f)
                        )
                        .padding(8.dp),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    Text(frameIdx, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                    Text(timestamp, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                    Text(pupilRadius, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                    Text(pupilDiameter, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                }
            }
            
            if (pupilData.size > 50) {
                Text(
                    text = "... and ${pupilData.size - 50} more rows",
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(8.dp),
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

