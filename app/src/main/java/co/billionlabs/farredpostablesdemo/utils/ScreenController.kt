package co.billionlabs.farredpostablesdemo.utils

import android.app.Activity
import android.content.Context
import android.provider.Settings
import android.util.Log
import android.view.WindowManager
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb

class ScreenController(private val activity: Activity) {
    
    private var originalBrightness: Float = -1f
    private var originalSystemBrightness: Int = -1
    
    fun saveCurrentBrightness() {
        try {
            // Save current window brightness
            val layoutParams = activity.window.attributes
            originalBrightness = layoutParams.screenBrightness
            
            // Save current system brightness
            originalSystemBrightness = Settings.System.getInt(
                activity.contentResolver,
                Settings.System.SCREEN_BRIGHTNESS,
                125
            )
            
            Log.d("ScreenController", "Saved brightness - Window: $originalBrightness, System: $originalSystemBrightness")
        } catch (e: Exception) {
            Log.e("ScreenController", "Error saving brightness", e)
        }
    }
    
    fun setMinimumBrightness() {
        try {
            val layoutParams = activity.window.attributes
            layoutParams.screenBrightness = 0.01f // Minimum brightness (1%)
            activity.window.attributes = layoutParams
            
            Log.d("ScreenController", "Set minimum brightness")
        } catch (e: Exception) {
            Log.e("ScreenController", "Error setting minimum brightness", e)
        }
    }
    
    fun setMaximumBrightness() {
        try {
            val layoutParams = activity.window.attributes
            layoutParams.screenBrightness = 1.0f // Maximum brightness (100%)
            activity.window.attributes = layoutParams
            
            Log.d("ScreenController", "Set maximum brightness")
        } catch (e: Exception) {
            Log.e("ScreenController", "Error setting maximum brightness", e)
        }
    }
    
    fun restoreOriginalBrightness() {
        try {
            val layoutParams = activity.window.attributes
            if (originalBrightness >= 0) {
                layoutParams.screenBrightness = originalBrightness
            } else {
                layoutParams.screenBrightness = -1f // Use system brightness
            }
            activity.window.attributes = layoutParams
            
            Log.d("ScreenController", "Restored original brightness: $originalBrightness")
        } catch (e: Exception) {
            Log.e("ScreenController", "Error restoring brightness", e)
        }
    }
    
    fun setBackgroundColor(color: Color) {
        try {
            activity.window.statusBarColor = color.toArgb()
            activity.window.navigationBarColor = color.toArgb()
            
            // Set the window background
            activity.window.decorView.setBackgroundColor(color.toArgb())
            
            Log.d("ScreenController", "Set background color: ${color}")
        } catch (e: Exception) {
            Log.e("ScreenController", "Error setting background color", e)
        }
    }
    
    fun setFullscreen() {
        try {
            activity.window.setFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN
            )
            Log.d("ScreenController", "Set fullscreen mode")
        } catch (e: Exception) {
            Log.e("ScreenController", "Error setting fullscreen", e)
        }
    }
    
    fun exitFullscreen() {
        try {
            activity.window.clearFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN)
            Log.d("ScreenController", "Exited fullscreen mode")
        } catch (e: Exception) {
            Log.e("ScreenController", "Error exiting fullscreen", e)
        }
    }
}
