"""
Landmark Detection Module
=========================

This module uses Google's MediaPipe library to detect hand and body landmarks
in video frames. These landmarks are the foundation of our sign language
recognition system.

Why MediaPipe?
--------------
MediaPipe is chosen over alternatives (OpenPose, custom CNN) for several reasons:

1. **Real-time Performance**: Optimized for mobile/edge devices, runs at 30+ FPS
   on CPU without GPU acceleration.

2. **Accuracy**: State-of-the-art hand tracking with 21 landmarks per hand,
   including fingertip positions crucial for sign language.

3. **Robustness**: Works well with varying lighting, skin tones, and partial
   occlusions - common in real-world signing.

4. **Ease of Use**: Simple Python API with pre-trained models included.

Hand Landmark Model:
--------------------
MediaPipe provides 21 3D landmarks per hand:

         8   12  16  20
         |   |   |   |
     7   11  15  19  |
     |   |   |   |   |
     6   10  14  18  |
     |   |   |   |   |
     5---9---13--17--+
          \         /
           4       /
            \     /
             3   /
              \ /
               2
               |
               1
               |
               0 (Wrist)

Landmarks 0-4: Thumb (CMC, MCP, IP, TIP)
Landmarks 5-8: Index finger
Landmarks 9-12: Middle finger
Landmarks 13-16: Ring finger
Landmarks 17-20: Pinky finger

Each landmark has (x, y, z) coordinates:
- x, y: Normalized to [0, 1] relative to image dimensions
- z: Depth relative to wrist, smaller = closer to camera

Pose Landmark Model:
--------------------
MediaPipe Pose provides 33 body landmarks, useful for signs that involve
arm/body movements (e.g., "big", "small", "me", "you").
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class HandLandmarks:
    """
    Container for hand landmark data from MediaPipe.

    This dataclass provides a clean interface to hand detection results,
    with utility methods for common operations.

    Attributes:
        landmarks: Raw landmark coordinates as numpy array (21, 3)
        handedness: 'Left' or 'Right' hand classification
        confidence: Detection confidence score (0-1)
        world_landmarks: 3D coordinates in real-world metric space

    Why a Dataclass?
    ----------------
    Using a dataclass instead of raw MediaPipe results provides:
    1. Type hints for better IDE support
    2. Easy serialization for logging/debugging
    3. Custom methods for common transformations
    4. Decoupling from MediaPipe's internal structures
    """
    landmarks: np.ndarray  # Shape: (21, 3) - normalized x, y, z
    handedness: str  # 'Left' or 'Right'
    confidence: float
    world_landmarks: Optional[np.ndarray] = None  # Real-world 3D coordinates

    def get_fingertips(self) -> np.ndarray:
        """
        Get fingertip positions (indices 4, 8, 12, 16, 20).

        Returns:
            Array of shape (5, 3) with [thumb, index, middle, ring, pinky] tips

        This is useful for gesture recognition as fingertip positions
        are often the most discriminative features for ASL letters.
        """
        fingertip_indices = [4, 8, 12, 16, 20]
        return self.landmarks[fingertip_indices]

    def get_finger_angles(self) -> Dict[str, float]:
        """
        Calculate finger bend angles (0° = straight, 90° = bent).

        Returns:
            Dictionary with angles for each finger

        These angles help distinguish between similar hand shapes
        (e.g., 'A' vs 'S' in ASL - both are fists but differ in thumb position).
        """
        angles = {}
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

        # MCP (knuckle) indices
        mcp_indices = [2, 5, 9, 13, 17]
        # PIP (middle joint) indices
        pip_indices = [3, 6, 10, 14, 18]
        # Tip indices
        tip_indices = [4, 8, 12, 16, 20]

        for i, name in enumerate(finger_names):
            # Vectors from MCP to PIP and PIP to TIP
            v1 = self.landmarks[pip_indices[i]] - self.landmarks[mcp_indices[i]]
            v2 = self.landmarks[tip_indices[i]] - self.landmarks[pip_indices[i]]

            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles[name] = np.degrees(angle)

        return angles

    def normalize(self) -> np.ndarray:
        """
        Normalize landmarks for model input.

        Normalization steps:
        1. Center on wrist (landmark 0)
        2. Scale by palm size (wrist to middle finger MCP)
        3. This makes the model invariant to hand position and size

        Returns:
            Normalized landmarks array of shape (21, 3)
        """
        # Center on wrist
        centered = self.landmarks - self.landmarks[0]

        # Scale by palm size (distance from wrist to middle finger MCP)
        palm_size = np.linalg.norm(self.landmarks[9] - self.landmarks[0])
        if palm_size > 0:
            normalized = centered / palm_size
        else:
            normalized = centered

        return normalized

    def flatten(self) -> np.ndarray:
        """Flatten to 1D array for simple models."""
        return self.landmarks.flatten()  # Shape: (63,)


@dataclass
class PoseLandmarks:
    """
    Container for pose (body) landmark data.

    The 33 pose landmarks include face, arms, and legs.
    For sign language, we primarily use the upper body landmarks (0-24).

    Key landmarks for signing:
    - 11, 12: Shoulders (left, right)
    - 13, 14: Elbows (left, right)
    - 15, 16: Wrists (left, right)
    - 17-22: Hands (pinky, index) - less accurate than hand model
    """
    landmarks: np.ndarray  # Shape: (33, 3)
    confidence: float
    visibility: np.ndarray  # Per-landmark visibility scores

    def get_upper_body(self) -> np.ndarray:
        """Get only upper body landmarks (shoulders to wrists)."""
        upper_body_indices = [11, 12, 13, 14, 15, 16]
        return self.landmarks[upper_body_indices]


class LandmarkDetector:
    """
    Unified landmark detection using MediaPipe.

    This class wraps MediaPipe's hand and pose solutions into a single
    interface optimized for sign language recognition.

    Architecture:
    -------------
    ┌─────────────────────────────────────────────────────────────┐
    │                    LandmarkDetector                         │
    │  ┌─────────────┐                      ┌──────────────────┐ │
    │  │   Frame     │──┬─────────────────▶│  Hand Detection  │ │
    │  │   Input     │  │                   │   (21 landmarks) │ │
    │  │             │  │                   └──────────────────┘ │
    │  └─────────────┘  │                                        │
    │                   │                   ┌──────────────────┐ │
    │                   └─────────────────▶│  Pose Detection  │ │
    │                                       │   (33 landmarks) │ │
    │                                       └──────────────────┘ │
    └─────────────────────────────────────────────────────────────┘

    Usage:
    ------
    >>> detector = LandmarkDetector()
    >>> hands, pose = detector.detect(frame)
    >>> for hand in hands:
    ...     print(f"{hand.handedness} hand: confidence={hand.confidence:.2f}")
    ...     normalized = hand.normalize()  # Ready for model input

    Performance Considerations:
    ---------------------------
    - Hand detection: ~10-15ms per frame on modern CPU
    - Pose detection: ~15-20ms per frame on modern CPU
    - Running both: ~20-30ms (some parallelization internally)

    For maximum performance, you can disable pose detection if not needed:
    >>> detector = LandmarkDetector(detect_pose=False)
    """

    # MediaPipe hand landmark connections for drawing
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),    # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        (5, 9), (9, 13), (13, 17),  # Palm
    ]

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        detect_pose: bool = True,
        model_complexity: int = 1
    ):
        """
        Initialize the landmark detector.

        Args:
            max_hands: Maximum number of hands to detect (1 or 2)
            min_detection_confidence: Minimum confidence for hand detection.
                Higher = fewer false positives but might miss some hands.
            min_tracking_confidence: Minimum confidence for tracking.
                Lower allows tracking through brief occlusions.
            detect_pose: Whether to also detect body pose landmarks.
            model_complexity: MediaPipe model complexity (0=lite, 1=full, 2=heavy)
                Higher = more accurate but slower.

        Model Complexity Trade-offs:
        ----------------------------
        - 0 (Lite): Fastest, good for simple gestures, ~10ms
        - 1 (Full): Balanced accuracy/speed, ~15ms (recommended)
        - 2 (Heavy): Most accurate, ~25ms, use if accuracy is critical
        """
        self.max_hands = max_hands
        self.detect_pose = detect_pose

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # False = video mode (uses tracking)
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Initialize MediaPipe Pose (optional)
        self.pose = None
        if detect_pose:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )

        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        logger.info(f"LandmarkDetector initialized: max_hands={max_hands}, "
                   f"detect_pose={detect_pose}, complexity={model_complexity}")

    def detect(
        self,
        frame: np.ndarray
    ) -> Tuple[List[HandLandmarks], Optional[PoseLandmarks]]:
        """
        Detect landmarks in a frame.

        Args:
            frame: BGR image from OpenCV (shape: H, W, 3)

        Returns:
            Tuple of (list of HandLandmarks, PoseLandmarks or None)

        Implementation Notes:
        ---------------------
        1. MediaPipe expects RGB images, so we convert from BGR (OpenCV default)
        2. We set writeable=False for performance (tells MediaPipe not to copy)
        3. Results are converted to our dataclass format for consistency
        """
        # Convert BGR to RGB (MediaPipe requirement)
        # Setting writeable=False improves performance
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        # Detect hands
        hand_results = self.hands.process(rgb_frame)
        hands = self._process_hand_results(hand_results)

        # Detect pose (if enabled)
        pose_landmarks = None
        if self.detect_pose and self.pose is not None:
            pose_results = self.pose.process(rgb_frame)
            pose_landmarks = self._process_pose_results(pose_results)

        return hands, pose_landmarks

    def _process_hand_results(
        self,
        results
    ) -> List[HandLandmarks]:
        """
        Convert MediaPipe hand results to our HandLandmarks format.

        MediaPipe returns landmarks as a protobuf message. We convert this
        to numpy arrays for easier manipulation and model input.
        """
        hands = []

        if results.multi_hand_landmarks is None:
            return hands

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Extract landmarks as numpy array
            landmarks = np.array([
                [lm.x, lm.y, lm.z]
                for lm in hand_landmarks.landmark
            ])

            # Get handedness (Left/Right)
            handedness = "Unknown"
            confidence = 0.0
            if results.multi_handedness:
                hand_info = results.multi_handedness[idx]
                handedness = hand_info.classification[0].label
                confidence = hand_info.classification[0].score

            # Extract world landmarks if available
            world_landmarks = None
            if results.multi_hand_world_landmarks:
                world_lm = results.multi_hand_world_landmarks[idx]
                world_landmarks = np.array([
                    [lm.x, lm.y, lm.z]
                    for lm in world_lm.landmark
                ])

            hands.append(HandLandmarks(
                landmarks=landmarks,
                handedness=handedness,
                confidence=confidence,
                world_landmarks=world_landmarks
            ))

        return hands

    def _process_pose_results(
        self,
        results
    ) -> Optional[PoseLandmarks]:
        """Convert MediaPipe pose results to our PoseLandmarks format."""
        if results.pose_landmarks is None:
            return None

        landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in results.pose_landmarks.landmark
        ])

        visibility = np.array([
            lm.visibility
            for lm in results.pose_landmarks.landmark
        ])

        return PoseLandmarks(
            landmarks=landmarks,
            confidence=np.mean(visibility),  # Average visibility as confidence
            visibility=visibility
        )

    def draw_landmarks(
        self,
        frame: np.ndarray,
        hands: List[HandLandmarks],
        pose: Optional[PoseLandmarks] = None,
        draw_connections: bool = True
    ) -> np.ndarray:
        """
        Draw detected landmarks on the frame.

        Args:
            frame: Input BGR frame
            hands: List of detected hands
            pose: Optional pose landmarks
            draw_connections: Whether to draw skeleton connections

        Returns:
            Frame with landmarks drawn

        Visualization helps with:
        - Debugging detection issues
        - User feedback during signing
        - Demonstration and presentations
        """
        output = frame.copy()
        h, w = output.shape[:2]

        # Draw hand landmarks
        for hand in hands:
            # Convert normalized coordinates to pixel coordinates
            landmarks_px = hand.landmarks.copy()
            landmarks_px[:, 0] *= w
            landmarks_px[:, 1] *= h

            # Draw landmark points
            for i, (x, y, z) in enumerate(landmarks_px):
                # Color based on finger
                if i in [4, 8, 12, 16, 20]:  # Fingertips
                    color = (0, 255, 0)  # Green
                    radius = 8
                else:
                    color = (255, 255, 255)  # White
                    radius = 5

                cv2.circle(output, (int(x), int(y)), radius, color, -1)

            # Draw connections (skeleton)
            if draw_connections:
                for start, end in self.HAND_CONNECTIONS:
                    pt1 = (int(landmarks_px[start, 0]), int(landmarks_px[start, 1]))
                    pt2 = (int(landmarks_px[end, 0]), int(landmarks_px[end, 1]))
                    cv2.line(output, pt1, pt2, (0, 255, 255), 2)

            # Draw handedness label
            wrist = landmarks_px[0]
            label = f"{hand.handedness} ({hand.confidence:.2f})"
            cv2.putText(
                output, label,
                (int(wrist[0]) - 40, int(wrist[1]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )

        # Draw pose landmarks (upper body only)
        if pose is not None:
            pose_px = pose.landmarks.copy()
            pose_px[:, 0] *= w
            pose_px[:, 1] *= h

            # Draw shoulders and arms
            upper_body = [11, 12, 13, 14, 15, 16]  # shoulders, elbows, wrists
            for idx in upper_body:
                if pose.visibility[idx] > 0.5:
                    x, y = int(pose_px[idx, 0]), int(pose_px[idx, 1])
                    cv2.circle(output, (x, y), 6, (255, 0, 255), -1)

            # Draw arm connections
            arm_connections = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12)]
            for start, end in arm_connections:
                if pose.visibility[start] > 0.5 and pose.visibility[end] > 0.5:
                    pt1 = (int(pose_px[start, 0]), int(pose_px[start, 1]))
                    pt2 = (int(pose_px[end, 0]), int(pose_px[end, 1]))
                    cv2.line(output, pt1, pt2, (255, 0, 255), 2)

        return output

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()
        if self.pose is not None:
            self.pose.close()
        logger.info("LandmarkDetector closed")


# =============================================================================
# Testing and Demo
# =============================================================================

def test_landmark_detector():
    """
    Test landmark detection with webcam.

    Run: python -m src.detection.landmark_detector
    """
    from ..webcam.capture import WebcamCapture

    print("Testing landmark detector...")
    print("Press 'q' to quit, 'p' to toggle pose detection")

    detector = LandmarkDetector(max_hands=2, detect_pose=True)
    show_pose = True

    with WebcamCapture(width=640, height=480) as capture:
        import time
        time.sleep(0.5)

        for frame in capture.frames():
            # Detect landmarks
            start_time = time.time()
            hands, pose = detector.detect(frame)
            detection_time = (time.time() - start_time) * 1000  # ms

            # Draw landmarks
            output = detector.draw_landmarks(
                frame, hands,
                pose if show_pose else None
            )

            # Display info
            info_text = f"Hands: {len(hands)} | Detection: {detection_time:.1f}ms"
            cv2.putText(
                output, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            # Display finger angles for first hand
            if hands:
                angles = hands[0].get_finger_angles()
                y_pos = 60
                for finger, angle in angles.items():
                    angle_text = f"{finger}: {angle:.1f}°"
                    cv2.putText(
                        output, angle_text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
                    )
                    y_pos += 20

            cv2.imshow('Landmark Detection Test', output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                show_pose = not show_pose
                print(f"Pose detection: {'ON' if show_pose else 'OFF'}")

    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_landmark_detector()
