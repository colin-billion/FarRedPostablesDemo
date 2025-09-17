package co.billionlabs.farredpostablesdemo.utils

import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

/**
 * Helper class for running Python code from Android
 * This class provides easy access to Python modules and functions
 */
class PythonHelper {
    
    companion object {
        private var pythonInstance: Python? = null
        
        /**
         * Check if Python is initialized
         */
        fun isInitialized(): Boolean = pythonInstance != null
        
        /**
         * Get the Python instance (must be initialized first)
         */
        fun getInstance(): Python {
            if (pythonInstance == null) {
                throw IllegalStateException("Python not initialized. Call initializePython(context) first.")
            }
            return pythonInstance!!
        }
        
        /**
         * Initialize Python platform and get Python instance
         * Call this once when your app starts
         * @param context The application context (usually from Activity)
         */
        fun initializePython(context: android.content.Context): Python {
            android.util.Log.d("PythonHelper", "initializePython() called with context: ${context.javaClass.simpleName}")
            
            if (pythonInstance == null) {
                android.util.Log.d("PythonHelper", "pythonInstance is null, initializing...")
                
                if (!Python.isStarted()) {
                    android.util.Log.d("PythonHelper", "Python not started, starting Python...")
                    try {
                        Python.start(AndroidPlatform(context.applicationContext))
                        android.util.Log.d("PythonHelper", "Python.start() completed")
                    } catch (e: Exception) {
                        android.util.Log.e("PythonHelper", "Failed to start Python: ${e.message}", e)
                        throw e
                    }
                } else {
                    android.util.Log.d("PythonHelper", "Python already started")
                }
                
                try {
                    pythonInstance = Python.getInstance()
                    android.util.Log.d("PythonHelper", "Python.getInstance() completed")
                } catch (e: Exception) {
                    android.util.Log.e("PythonHelper", "Failed to get Python instance: ${e.message}", e)
                    throw e
                }
            } else {
                android.util.Log.d("PythonHelper", "pythonInstance already exists")
            }
            
            android.util.Log.d("PythonHelper", "initializePython() completed successfully")
            return pythonInstance!!
        }
        
        /**
         * Get a Python module (requires Python to be initialized first)
         * @param moduleName Name of the Python module to import
         * @return Python module object
         */
        fun getModule(moduleName: String): com.chaquo.python.PyObject {
            if (pythonInstance == null) {
                throw IllegalStateException("Python not initialized. Call initializePython(context) first.")
            }
            return pythonInstance!!.getModule(moduleName)
        }
        
        /**
         * Run a simple Python expression (requires Python to be initialized first)
         * @param expression Python expression to evaluate
         * @return Result of the expression, or null if not found
         */
        fun runExpression(expression: String): com.chaquo.python.PyObject? {
            if (pythonInstance == null) {
                throw IllegalStateException("Python not initialized. Call initializePython(context) first.")
            }
            return pythonInstance!!.getBuiltins().get(expression)
        }
    }
}

/**
 * Example usage of your pupil tracking Python modules
 */
class PupilTrackingHelper {
    
    private val python: Python
    
    init {
        python = PythonHelper.getInstance()
    }
    
    /**
     * Process a video file using your pupil tracking pipeline
     * @param videoPath Path to the video file
     * @return Processing result with output directory path
     */
    fun processVideo(videoPath: String): Map<String, Any> {
        try {
            android.util.Log.d("PupilTrackingHelper", "Starting video processing: $videoPath")
            
            // Import the simplified pupil tracking pipeline module
            val pupilModule = python.getModule("pupil_tracking_pipeline_simple")
            
            // Call the main processing function from your simplified pipeline
            // This should process the video and return the output directory
            val result = pupilModule.callAttr("main", videoPath)
            
            android.util.Log.d("PupilTrackingHelper", "Video processing completed")
            
            // Check if result is null (pipeline failed)
            if (result == null) {
                android.util.Log.e("PupilTrackingHelper", "Python script returned null - pipeline failed")
                return mapOf(
                    "success" to false,
                    "message" to "Pupil tracking pipeline failed - check video file and try again",
                    "videoPath" to videoPath
                )
            }
            
            // Return processing results
            return mapOf(
                "success" to true,
                "message" to "Processing completed successfully",
                "outputDir" to result.toString(),
                "videoPath" to videoPath
            )
        } catch (e: Exception) {
            android.util.Log.e("PupilTrackingHelper", "Error processing video: ${e.message}", e)
            return mapOf(
                "success" to false,
                "message" to "Error processing video: ${e.message}",
                "error" to e.toString()
            )
        }
    }
    
    /**
     * Get pupil data from the processed results
     * @param outputDir Directory where processing results were saved
     * @param videoName Name of the video file (without extension)
     * @return Pupil data as a list of maps
     */
    fun getPupilTimeSeries(outputDir: String, videoName: String): List<Map<String, Any>> {
        try {
            val pandas = python.getModule("pandas")
            val os = python.getModule("os")
            val path = python.getModule("pathlib").callAttr("Path")
            
            // Construct the CSV file path
            val csvPath = path.callAttr("join", outputDir, videoName, "pupil_data_filtered_${videoName}.csv")
            
            android.util.Log.d("PupilTrackingHelper", "Loading pupil data from: $csvPath")
            
            // Check if file exists
            if (!os.callAttr("path", "exists", csvPath).toBoolean()) {
                android.util.Log.w("PupilTrackingHelper", "Filtered data not found, trying clean data")
                // Try the clean data instead
                val cleanCsvPath = path.callAttr("join", outputDir, videoName, "pupil_data_clean_${videoName}.csv")
                if (os.callAttr("path", "exists", cleanCsvPath).toBoolean()) {
                    return loadPupilDataFromCsv(pandas, cleanCsvPath.toString())
                } else {
                    android.util.Log.e("PupilTrackingHelper", "No pupil data files found")
                    return emptyList()
                }
            }
            
            return loadPupilDataFromCsv(pandas, csvPath.toString())
        } catch (e: Exception) {
            android.util.Log.e("PupilTrackingHelper", "Error loading pupil data: ${e.message}", e)
            return emptyList()
        }
    }
    
    private fun loadPupilDataFromCsv(pandas: com.chaquo.python.PyObject, csvPath: String): List<Map<String, Any>> {
        try {
            // Read CSV file
            val df = pandas.callAttr("read_csv", csvPath)
            
            // Convert to list of dictionaries
            val result = mutableListOf<Map<String, Any>>()
            val pythonList = df.callAttr("to_dict", "records")
            val pythonListAsList = pythonList.asList()
            
            for (item in pythonListAsList) {
                val itemMap = mutableMapOf<String, Any>()
                val itemDict = item.asMap()
                
                for ((key, value) in itemDict) {
                    val keyStr = key.toString()
                    itemMap[keyStr] = when {
                        value != null && value.toString().matches(Regex("-?\\d+(\\.\\d+)?")) -> value.toDouble()
                        value != null && (value.toString().equals("true", ignoreCase = true) || value.toString().equals("false", ignoreCase = true)) -> value.toBoolean()
                        value != null -> value.toString()
                        else -> value?.toString() ?: ""
                    }
                }
                result.add(itemMap)
            }
            
            android.util.Log.d("PupilTrackingHelper", "Loaded ${result.size} pupil data points")
            return result
        } catch (e: Exception) {
            android.util.Log.e("PupilTrackingHelper", "Error parsing CSV: ${e.message}", e)
            return emptyList()
        }
    }
    
    /**
     * Get pupil data from a CSV file
     * @param csvPath Path to the CSV file
     * @return Pupil data as a list of maps
     */
    fun getPupilData(csvPath: String): List<Map<String, Any>> {
        try {
            val pandas = python.getModule("pandas")
            
            // Read CSV file
            val df = pandas.callAttr("read_csv", csvPath)
            
            // Convert to list of dictionaries
            val dataList = df.callAttr("to_dict", "records")
            
            // Convert Python list to Kotlin list of maps
            val result = mutableListOf<Map<String, Any>>()
            val pythonList = dataList.asList()
            
            for (item in pythonList) {
                val itemMap = mutableMapOf<String, Any>()
                
                // Convert Python dict to Kotlin map using asMap()
                val itemDict = item.asMap()
                
                for ((key, value) in itemDict) {
                    val keyStr = key.toString()
                    // Convert PyObject to appropriate Kotlin type
                    itemMap[keyStr] = try {
                        when {
                            value != null && value.toString().matches(Regex("-?\\d+\\.?\\d*")) -> value.toDouble()
                            value != null -> value.toString()
                            else -> ""
                        }
                    } catch (e: Exception) {
                        value?.toString() ?: ""
                    }
                }
                result.add(itemMap)
            }
            
            return result
        } catch (e: Exception) {
            return emptyList()
        }
    }
    
    /**
     * Test Python integration with a simple calculation
     * @return Test result
     */
    fun testPythonIntegration(): String {
        try {
            // Test basic Python functionality with Python list
            val builtins = python.getBuiltins()
            
            // Create a Python list instead of passing Java/Kotlin list
            val pythonList = python.getBuiltins().callAttr("list", arrayOf(1, 2, 3, 4, 5))
            val result = builtins.callAttr("sum", pythonList)
            
            // Test numpy if available
            try {
                val numpy = python.getModule("numpy")
                val array = numpy.callAttr("array", pythonList)
                val mean = array.callAttr("mean")
                return "Python test successful! Builtin sum = $result, NumPy mean = $mean"
            } catch (e: Exception) {
                return "Python test successful! Builtin sum = $result (NumPy not available: ${e.message})"
            }
        } catch (e: Exception) {
            return "Python test failed: ${e.message}"
        }
    }
}
