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
         * Initialize Python platform and get Python instance
         * Call this once when your app starts
         */
        fun initializePython(): Python {
            android.util.Log.d("PythonHelper", "initializePython() called")
            
            if (pythonInstance == null) {
                android.util.Log.d("PythonHelper", "pythonInstance is null, initializing...")
                
                if (!Python.isStarted()) {
                    android.util.Log.d("PythonHelper", "Python not started, starting Python...")
                    try {
                        Python.start(AndroidPlatform(android.app.Application()))
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
         * Get a Python module
         * @param moduleName Name of the Python module to import
         * @return Python module object
         */
        fun getModule(moduleName: String) = initializePython().getModule(moduleName)
        
        /**
         * Run a simple Python expression
         * @param expression Python expression to evaluate
         * @return Result of the expression
         */
        fun runExpression(expression: String) = initializePython().getBuiltins().get(expression)
    }
}

/**
 * Example usage of your pupil tracking Python modules
 */
class PupilTrackingHelper {
    
    private val python = PythonHelper.initializePython()
    
    /**
     * Process a video file using your pupil tracking pipeline
     * @param videoPath Path to the video file
     * @return Processing result
     */
    fun processVideo(videoPath: String): String {
        try {
            // Import your pupil tracking module
            val pupilModule = python.getModule("pupil_tracking_pipeline")
            
            // Call the main processing function
            // Note: You may need to modify your Python scripts to expose functions properly
            val result = pupilModule.callAttr("process_video", videoPath)
            
            return "Processing completed: $result"
        } catch (e: Exception) {
            return "Error processing video: ${e.message}"
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
            val numpy = python.getModule("numpy")
            val result = numpy.callAttr("array", listOf(1, 2, 3, 4, 5))
            val mean = result.callAttr("mean")
            return "Python test successful! Mean of [1,2,3,4,5] = $mean"
        } catch (e: Exception) {
            return "Python test failed: ${e.message}"
        }
    }
}
