"""
Landmark Detection Module

Uses MediaPipe for hand and pose landmark detection.
"""

from .landmark_detector import LandmarkDetector, HandLandmarks, PoseLandmarks

__all__ = ['LandmarkDetector', 'HandLandmarks', 'PoseLandmarks']
