"""
Webcam Capture Module
=====================

This module handles real-time video capture from a webcam or video file.
It provides a clean interface for frame acquisition with configurable
resolution, FPS, and preprocessing options.

Key Concepts:
-------------
1. **Double Buffering**: We use a separate thread for frame capture to ensure
   the main processing loop isn't blocked by I/O operations. This is crucial
   for maintaining real-time performance.

2. **Frame Buffer**: A rolling buffer stores recent frames for temporal models
   that need sequence data (like LSTMs).

3. **Graceful Degradation**: If the camera can't maintain target FPS, we
   handle frame drops gracefully rather than crashing.

Why OpenCV?
-----------
OpenCV (cv2) is the industry standard for computer vision. It provides:
- Cross-platform webcam access
- Efficient frame handling
- Built-in image processing
- Hardware acceleration support
"""

import cv2
import numpy as np
import threading
import time
from collections import deque
from typing import Optional, Tuple, Generator
import logging

logger = logging.getLogger(__name__)


class FrameBuffer:
    """
    A thread-safe circular buffer for storing video frames.

    This is essential for temporal models (CNN+LSTM) that need to analyze
    sequences of frames to recognize dynamic gestures.

    Design Rationale:
    -----------------
    - Uses collections.deque for O(1) append/pop operations
    - Thread-safe with a lock to prevent race conditions
    - Fixed size to prevent memory issues during long sessions

    Example:
    --------
    >>> buffer = FrameBuffer(maxlen=30)  # 1 second at 30 FPS
    >>> buffer.add(frame)
    >>> sequence = buffer.get_sequence()  # Get all 30 frames
    """

    def __init__(self, maxlen: int = 30):
        """
        Initialize the frame buffer.

        Args:
            maxlen: Maximum number of frames to store. For a model expecting
                   1 second of video at 30 FPS, use maxlen=30.
        """
        self.buffer = deque(maxlen=maxlen)
        self.maxlen = maxlen
        self._lock = threading.Lock()

    def add(self, frame: np.ndarray) -> None:
        """
        Add a frame to the buffer (thread-safe).

        The deque automatically removes the oldest frame when full,
        maintaining a sliding window of recent frames.
        """
        with self._lock:
            self.buffer.append(frame.copy())  # Copy to prevent reference issues

    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Get the current frame sequence as a numpy array.

        Returns:
            numpy array of shape (num_frames, height, width, channels)
            or None if buffer isn't full yet.
        """
        with self._lock:
            if len(self.buffer) < self.maxlen:
                return None
            return np.array(list(self.buffer))

    def get_latest(self) -> Optional[np.ndarray]:
        """Get the most recent frame."""
        with self._lock:
            if len(self.buffer) == 0:
                return None
            return self.buffer[-1].copy()

    def is_ready(self) -> bool:
        """Check if buffer has enough frames for processing."""
        with self._lock:
            return len(self.buffer) >= self.maxlen

    def clear(self) -> None:
        """Clear all frames from the buffer."""
        with self._lock:
            self.buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self.buffer)


class WebcamCapture:
    """
    Real-time webcam capture with threading support.

    This class provides a high-level interface for webcam access,
    handling all the low-level details of video capture.

    Architecture:
    -------------
    ┌─────────────────────────────────────────────────────────┐
    │                    WebcamCapture                         │
    │  ┌─────────────┐    ┌──────────────┐    ┌────────────┐  │
    │  │   Capture   │───▶│  Processing  │───▶│   Output   │  │
    │  │   Thread    │    │   (resize,   │    │   Frame    │  │
    │  │             │    │    flip)     │    │            │  │
    │  └─────────────┘    └──────────────┘    └────────────┘  │
    └─────────────────────────────────────────────────────────┘

    Usage:
    ------
    >>> capture = WebcamCapture(device_id=0, width=640, height=480)
    >>> capture.start()
    >>> for frame in capture.frames():
    ...     # Process frame
    ...     cv2.imshow('Frame', frame)
    >>> capture.stop()

    Or using context manager:
    >>> with WebcamCapture() as capture:
    ...     for frame in capture.frames():
    ...         process(frame)
    """

    def __init__(
        self,
        device_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        flip_horizontal: bool = True,
        buffer_size: int = 30
    ):
        """
        Initialize webcam capture.

        Args:
            device_id: Camera device index (0 for default webcam)
            width: Desired frame width in pixels
            height: Desired frame height in pixels
            fps: Target frames per second
            flip_horizontal: Mirror the image (True for natural interaction)
            buffer_size: Number of frames to keep in buffer

        Technical Notes:
        ----------------
        - The actual FPS may be lower than requested if the camera or
          processing can't keep up.
        - flip_horizontal=True creates a mirror effect, which is more
          intuitive for users as their movements match the display.
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_horizontal = flip_horizontal

        # Video capture object (initialized in start())
        self._cap: Optional[cv2.VideoCapture] = None

        # Threading components
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        # Current frame storage
        self._current_frame: Optional[np.ndarray] = None

        # Frame buffer for sequence models
        self.frame_buffer = FrameBuffer(maxlen=buffer_size)

        # Performance metrics
        self._frame_count = 0
        self._start_time = 0
        self._actual_fps = 0

    def start(self) -> bool:
        """
        Start the webcam capture thread.

        Returns:
            True if started successfully, False otherwise.

        Implementation Notes:
        ---------------------
        We use a separate thread for capture because cv2.read() is a
        blocking operation. By running it in a thread, the main loop
        can continue processing while the next frame is being captured.
        """
        if self._running:
            logger.warning("Capture already running")
            return True

        # Initialize video capture
        self._cap = cv2.VideoCapture(self.device_id)

        if not self._cap.isOpened():
            logger.error(f"Failed to open camera {self.device_id}")
            return False

        # Configure camera properties
        # Note: Not all cameras support all properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Some cameras have a buffer that causes latency - minimize it
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Log actual camera settings (may differ from requested)
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")

        # Start capture thread
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        logger.info("Webcam capture started")
        return True

    def _capture_loop(self) -> None:
        """
        Internal capture loop running in a separate thread.

        This continuously reads frames from the camera and stores them.
        The main thread can then access the latest frame without blocking.

        Why daemon=True?
        ----------------
        A daemon thread automatically terminates when the main program exits,
        preventing the application from hanging if stop() isn't called.
        """
        while self._running:
            ret, frame = self._cap.read()

            if not ret:
                logger.warning("Failed to read frame")
                time.sleep(0.01)  # Brief pause before retry
                continue

            # Apply preprocessing
            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)  # 1 = horizontal flip

            # Thread-safe frame update
            with self._lock:
                self._current_frame = frame
                self._frame_count += 1

            # Add to buffer for sequence models
            self.frame_buffer.add(frame)

            # Calculate actual FPS
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                self._actual_fps = self._frame_count / elapsed

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the latest frame (thread-safe).

        Returns:
            Tuple of (success, frame) similar to cv2.VideoCapture.read()

        This mimics the OpenCV interface for easy integration with
        existing code that uses cv2.VideoCapture directly.
        """
        with self._lock:
            if self._current_frame is None:
                return False, None
            return True, self._current_frame.copy()

    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields frames continuously.

        This provides a clean iteration interface:
        >>> for frame in capture.frames():
        ...     process(frame)

        The generator handles frame timing internally to maintain
        consistent frame rate without busy-waiting.
        """
        frame_interval = 1.0 / self.fps
        last_frame_time = time.time()

        while self._running:
            # Rate limiting to prevent CPU overuse
            current_time = time.time()
            elapsed = current_time - last_frame_time

            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

            ret, frame = self.read()
            if ret:
                last_frame_time = time.time()
                yield frame

    def get_fps(self) -> float:
        """Get the actual frames per second being achieved."""
        return round(self._actual_fps, 1)

    def get_frame_count(self) -> int:
        """Get total number of frames captured."""
        return self._frame_count

    def stop(self) -> None:
        """
        Stop the capture thread and release resources.

        Always call this when done to properly release the camera.
        Failure to do so may cause issues with other applications
        trying to access the camera.
        """
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        logger.info("Webcam capture stopped")

    def __enter__(self) -> 'WebcamCapture':
        """Context manager entry - start capture."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop capture."""
        self.stop()

    def __del__(self) -> None:
        """Destructor - ensure resources are released."""
        self.stop()


# =============================================================================
# Testing and Demo
# =============================================================================

def test_webcam_capture():
    """
    Test function to verify webcam capture is working.

    Run this directly to test your webcam:
    $ python -m src.webcam.capture
    """
    print("Testing webcam capture...")
    print("Press 'q' to quit, 's' to save a frame")

    with WebcamCapture(width=640, height=480, fps=30) as capture:
        # Wait a moment for camera to initialize
        time.sleep(0.5)

        frame_count = 0
        for frame in capture.frames():
            # Display FPS on frame
            fps_text = f"FPS: {capture.get_fps()}"
            cv2.putText(
                frame, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # Display buffer status
            buffer_text = f"Buffer: {len(capture.frame_buffer)}/{capture.frame_buffer.maxlen}"
            cv2.putText(
                frame, buffer_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

            cv2.imshow('Webcam Test', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")

            frame_count += 1

        cv2.destroyAllWindows()
        print(f"Captured {frame_count} frames")
        print(f"Average FPS: {capture.get_fps()}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    test_webcam_capture()
