"""
Display system for Opi Voice Assistant.
Handles HDMI passthrough, overlay graphics, and mode switching.
Adapted from the HDMI switcher codebase.
"""

import asyncio
import logging
import threading
import time
from typing import Optional, Dict, Any
from enum import Enum

# For display/graphics (these might need to be installed)
try:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib
    import cairo
    GST_AVAILABLE = True
except ImportError:
    GST_AVAILABLE = False
    logging.warning("GStreamer not available - display features will be limited")

from core.config import DisplayConfig


class DisplayMode(Enum):
    """Available display modes."""
    PASSTHROUGH = "passthrough"
    OVERLAY = "overlay"
    DESKTOP = "desktop"


class DisplayManager:
    """Manages display pipeline and overlay system."""
    
    def __init__(self, config: DisplayConfig):
        self.config = config
        self.logger = logging.getLogger("DisplayManager")
        
        # GStreamer components
        self.pipeline = None
        self.overlay_element = None
        self.loop = None
        
        # State
        self.initialized = False
        self.current_mode = DisplayMode.PASSTHROUGH
        self.overlay_text = ""
        self.overlay_visible = False
        self.overlay_timeout_task = None
        
        # Threading
        self.gst_thread = None
        self.gst_loop = None
        
        if GST_AVAILABLE:
            Gst.init(None)
    
    async def initialize(self):
        """Initialize display system."""
        if self.initialized:
            return
        
        if not GST_AVAILABLE:
            self.logger.warning("GStreamer not available - using fallback display system")
            self.initialized = True
            return
        
        self.logger.info("Initializing display system...")
        
        try:
            # Start GStreamer in separate thread
            self.gst_thread = threading.Thread(target=self._run_gst_loop, daemon=True)
            self.gst_thread.start()
            
            # Wait a bit for GStreamer to initialize
            await asyncio.sleep(1)
            
            # Create initial pipeline
            await self._switch_to_mode(DisplayMode(self.config.default_mode))
            
            self.initialized = True
            self.logger.info("Display system initialized")
            
        except Exception as e:
            self.logger.error(f"Display initialization failed: {e}")
            # Continue without display features
            self.initialized = True
    
    def _run_gst_loop(self):
        """Run GStreamer main loop in separate thread."""
        if not GST_AVAILABLE:
            return
        
        self.gst_loop = GLib.MainLoop()
        try:
            self.gst_loop.run()
        except Exception as e:
            self.logger.error(f"GStreamer loop error: {e}")
    
    async def switch_mode(self, mode: str):
        """Switch display mode."""
        try:
            display_mode = DisplayMode(mode)
            await self._switch_to_mode(display_mode)
            self.logger.info(f"Switched to display mode: {mode}")
        except ValueError:
            self.logger.error(f"Invalid display mode: {mode}")
            raise
    
    async def _switch_to_mode(self, mode: DisplayMode):
        """Internal method to switch display mode."""
        if not GST_AVAILABLE:
            self.logger.info(f"Display mode would be: {mode.value}")
            self.current_mode = mode
            return
        
        # Stop current pipeline
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        # Create new pipeline based on mode
        if mode == DisplayMode.PASSTHROUGH:
            pipeline_str = self._get_passthrough_pipeline()
        elif mode == DisplayMode.OVERLAY:
            pipeline_str = self._get_overlay_pipeline()
        elif mode == DisplayMode.DESKTOP:
            pipeline_str = self._get_desktop_pipeline()
        else:
            raise ValueError(f"nknown display mode: {mode}")
        
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
            
            # Connect overlay callback if needed
            if mode == DisplayMode.OVERLAY:
                overlay = self.pipeline.get_by_name("text_overlay")
                if overlay:
                    overlay.connect("draw", self._on_overlay_draw)
                    self.overlay_element = overlay
            
            # Start pipeline
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise Exception("Failed to start pipeline")
            
            self.current_mode = mode
            
        except Exception as e:
            self.logger.error(f"Pipeline creation failed: {e}")
            # Fallback to simple mode
            self.current_mode = mode
    
    def _get_passthrough_pipeline(self) -> str:
        """Get GStreamer pipeline for passthrough mode."""
        return (
            f"v4l2src device={self.config.video_device} ! "
            f"video/x-raw,width={self.config.width},height={self.config.height} ! "
            f"kmssink connector-id={self.config.connector_id} "
            f"plane-id={self.config.plane_id} sync=false"
        )
    
    def _get_overlay_pipeline(self) -> str:
        """Get GStreamer pipeline for overlay mode."""
        return (
            f"v4l2src device={self.config.video_device} ! "
            f"video/x-raw,width={self.config.width},height={self.config.height} ! "
            f"videoconvert ! "
            f"cairooverlay name=text_overlay ! "
            f"videoconvert ! "
            f"kmssink connector-id={self.config.connector_id} "
            f"plane-id={self.config.plane_id} sync=false"
        )
    
    def _get_desktop_pipeline(self) -> str:
        """Get GStreamer pipeline for desktop mode."""
        # This would show the Orange Pi desktop instead of passthrough
        return (
            f"ximagesrc ! "
            f"video/x-raw,width={self.config.width},height={self.config.height} ! "
            f"videoconvert ! "
            f"kmssink connector-id={self.config.connector_id} "
            f"plane-id={self.config.plane_id} sync=false"
        )
    
    def _on_overlay_draw(self, overlay, context, timestamp, duration):
        """Callback for drawing overlay graphics."""
        if not self.overlay_visible or not self.overlay_text:
            return
        
        try:
            # Set up text rendering
            context.select_font_face(
                self.config.font_family,
                cairo.FONT_SLANT_NORMAL,
                cairo.FONT_WEIGHT_NORMAL
            )
            context.set_font_size(self.config.font_size)
            
            # Set text color (white with semi-transparent background)
            context.set_source_rgba(1, 1, 1, 0.9)  # White text
            
            # Calculate text position (bottom of screen)
            text_extents = context.text_extents(self.overlay_text)
            x = 50  # Left margin
            y = self.config.height - 100  # Bottom margin
            
            # Draw background rectangle
            context.set_source_rgba(0, 0, 0, 0.7)  # Semi-transparent black
            context.rectangle(
                x - 20,
                y - text_extents.height - 10,
                text_extents.width + 40,
                text_extents.height + 20
            )
            context.fill()
            
            # Draw text
            context.set_source_rgba(1, 1, 1, 0.9)  # White text
            context.move_to(x, y)
            context.show_text(self.overlay_text)
            
        except Exception as e:
            self.logger.error(f"Overlay drawing error: {e}")
    
    async def show_overlay(self, text: str, timeout: Optional[int] = None):
        """Show text overlay."""
        self.overlay_text = text
        self.overlay_visible = True
        
        self.logger.debug(f"Showing overlay: {text}")
        
        # Cancel existing timeout
        if self.overlay_timeout_task:
            self.overlay_timeout_task.cancel()
        
        # Set timeout for auto-hide
        if timeout is None:
            timeout = self.config.overlay_timeout
        
        if timeout > 0:
            self.overlay_timeout_task = asyncio.create_task(
                self._auto_hide_overlay(timeout)
            )
        
        # If not in overlay mode, temporarily switch
        if self.current_mode != DisplayMode.OVERLAY:
            await self._switch_to_mode(DisplayMode.OVERLAY)
    
    async def _auto_hide_overlay(self, timeout: int):
        """Auto-hide overlay after timeout."""
        try:
            await asyncio.sleep(timeout)
            await self.clear_overlay()
        except asyncio.CancelledError:
            pass
    
    async def clear_overlay(self):
        """Clear text overlay."""
        self.overlay_text = ""
        self.overlay_visible = False
        
        self.logger.debug("Clearing overlay")
        
        # Switch back to passthrough mode
        if self.current_mode == DisplayMode.OVERLAY:
            await self._switch_to_mode(DisplayMode.PASSTHROUGH)
    
    async def stream_text(self, text: str, delay: float = 0.05):
        """Stream text to overlay character by character."""
        if not text:
            return
        
        self.logger.debug(f"Streaming text: {text}")
        
        # Switch to overlay mode if needed
        if self.current_mode != DisplayMode.OVERLAY:
            await self._switch_to_mode(DisplayMode.OVERLAY)
        
        # Stream text character by character
        displayed_text = ""
        for char in text:
            displayed_text += char
            self.overlay_text = displayed_text
            self.overlay_visible = True
            await asyncio.sleep(delay)
        
        # Keep final text visible for a while
        await asyncio.sleep(3)
    
    async def take_screenshot(self) -> Optional[str]:
        """Take a screenshot and return base64 encoded image."""
        # This is a placeholder - would need actual screenshot implementation
        self.logger.info("Screenshot requested")
        return None
    
    async def get_display_info(self) -> Dict[str, Any]:
        """Get current display information."""
        return {
            "mode": self.current_mode.value,
            "resolution": f"{self.config.width}x{self.config.height}",
            "overlay_visible": self.overlay_visible,
            "overlay_text": self.overlay_text,
            "connector_id": self.config.connector_id,
            "plane_id": self.config.plane_id
        }
    
    async def shutdown(self):
        """Shutdown display system."""
        self.logger.info("Shutting down display system...")
        
        # Cancel overlay timeout
        if self.overlay_timeout_task:
            self.overlay_timeout_task.cancel()
        
        # Stop pipeline
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        # Stop GStreamer loop
        if self.gst_loop:
            self.gst_loop.quit()
        
        # Wait for thread to finish
        if self.gst_thread and self.gst_thread.is_alive():
            self.gst_thread.join(timeout=2)
        
        self.initialized = False


class FallbackDisplayManager:
    """Fallback display manager when GStreamer is not available."""
    
    def __init__(self, config: DisplayConfig):
        self.config = config
        self.logger = logging.getLogger("FallbackDisplay")
        self.current_mode = DisplayMode.PASSTHROUGH
        self.overlay_text = ""
    
    async def initialize(self):
        self.logger.info("Using fallback display manager")
    
    async def switch_mode(self, mode: str):
        self.current_mode = DisplayMode(mode)
        self.logger.info(f"Display mode: {mode}")
    
    async def show_overlay(self, text: str, timeout: Optional[int] = None):
        self.overlay_text = text
        self.logger.info(f"Overlay: {text}")
    
    async def clear_overlay(self):
        self.overlay_text = ""
        self.logger.info("Overlay cleared")
    
    async def stream_text(self, text: str, delay: float = 0.05):
        self.logger.info(f"Streaming: {text}")
        # Just show the final text
        await self.show_overlay(text)
    
    async def take_screenshot(self) -> Optional[str]:
        self.logger.info("Screenshot requested (not available)")
        return None
    
    async def get_display_info(self) -> Dict[str, Any]:
        return {
            "mode": self.current_mode.value,
            "overlay_text": self.overlay_text,
            "fallback": True
        }
    
    async def shutdown(self):
        self.logger.info("Fallback display shutdown")U
