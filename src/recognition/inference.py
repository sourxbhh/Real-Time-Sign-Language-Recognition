"""
Real-Time Gesture Recognition Inference
========================================

This module handles real-time inference for sign language recognition,
including frame buffering, prediction smoothing, and gesture segmentation.

Key Challenges in Real-Time Inference:
--------------------------------------

1. **Latency vs Accuracy Trade-off**
   - More frames = better accuracy but higher latency
   - We use 30 frames (1 second) as a good balance

2. **Noisy Predictions**
   - Frame-by-frame predictions can be jittery
   - We apply temporal smoothing (majority vote) for stability

3. **Gesture Segmentation**
   - Continuous signing needs to be split into discrete signs
   - We detect "idle" periods (no clear gesture) as boundaries

4. **Confidence Handling**
   - Low confidence predictions should be filtered
   - We use adaptive thresholding based on recent predictions

Pipeline:
---------
Frame → Landmarks → Normalize → Buffer → Model → Smooth → Output
                                  ↓
                          Gesture Segmentation
                                  ↓
                          Word/Sentence Building
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import time
import logging

from .model import SignLanguageModel, SignLanguageModelStatic

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """
    Container for a single prediction result.

    Attributes:
        label: Predicted sign label (e.g., 'A', 'B', 'hello')
        confidence: Confidence score (0-1)
        class_idx: Raw class index
        probabilities: Full probability distribution
        timestamp: When prediction was made
        is_stable: Whether prediction has been stable for smoothing window
    """
    label: str
    confidence: float
    class_idx: int
    probabilities: np.ndarray
    timestamp: float
    is_stable: bool = False


class PredictionSmoother:
    """
    Temporal smoothing for prediction stability.

    Why Smooth Predictions?
    -----------------------
    Raw frame-by-frame predictions can be noisy:
    - Slight hand movement → different prediction
    - Model uncertainty → alternating predictions
    - Transition between signs → ambiguous frames

    Smoothing Methods:
    ------------------
    1. **Majority Vote**: Most common prediction in window
       - Simple, interpretable
       - Good for distinct signs

    2. **Weighted Average**: Recent predictions weighted higher
       - Smoother transitions
       - Better for continuous signing

    3. **Confidence Weighted**: High-confidence predictions count more
       - Filters out uncertain frames
       - Most robust method

    We implement all three and default to confidence-weighted majority vote.
    """

    def __init__(
        self,
        window_size: int = 5,
        method: str = 'confidence_vote',
        min_confidence: float = 0.5
    ):
        """
        Initialize the smoother.

        Args:
            window_size: Number of predictions to consider
            method: 'majority_vote', 'weighted_average', or 'confidence_vote'
            min_confidence: Minimum confidence to include in smoothing
        """
        self.window_size = window_size
        self.method = method
        self.min_confidence = min_confidence

        # Recent predictions buffer
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)

    def add(self, class_idx: int, confidence: float) -> None:
        """Add a new prediction to the buffer."""
        self.predictions.append(class_idx)
        self.confidences.append(confidence)

    def get_smoothed(self) -> Tuple[int, float, bool]:
        """
        Get smoothed prediction.

        Returns:
            Tuple of (class_idx, confidence, is_stable)
            is_stable is True if all recent predictions agree
        """
        if len(self.predictions) == 0:
            return -1, 0.0, False

        predictions = list(self.predictions)
        confidences = list(self.confidences)

        if self.method == 'majority_vote':
            return self._majority_vote(predictions, confidences)
        elif self.method == 'weighted_average':
            return self._weighted_average(predictions, confidences)
        else:  # confidence_vote
            return self._confidence_vote(predictions, confidences)

    def _majority_vote(
        self,
        predictions: List[int],
        confidences: List[float]
    ) -> Tuple[int, float, bool]:
        """Simple majority voting."""
        from collections import Counter

        counts = Counter(predictions)
        winner, count = counts.most_common(1)[0]

        # Stability: winner was predicted in majority of window
        is_stable = count >= len(predictions) // 2 + 1

        # Confidence: average confidence of winning class
        winner_confs = [c for p, c in zip(predictions, confidences) if p == winner]
        avg_conf = np.mean(winner_confs)

        return winner, avg_conf, is_stable

    def _weighted_average(
        self,
        predictions: List[int],
        confidences: List[float]
    ) -> Tuple[int, float, bool]:
        """Recency-weighted voting."""
        # More recent predictions get higher weight
        weights = np.linspace(0.5, 1.0, len(predictions))

        # Weighted vote count per class
        class_scores = {}
        for pred, conf, weight in zip(predictions, confidences, weights):
            if pred not in class_scores:
                class_scores[pred] = 0
            class_scores[pred] += weight * conf

        winner = max(class_scores, key=class_scores.get)
        winner_score = class_scores[winner] / np.sum(weights)

        # Stability check
        is_stable = all(p == winner for p in predictions[-3:])

        return winner, winner_score, is_stable

    def _confidence_vote(
        self,
        predictions: List[int],
        confidences: List[float]
    ) -> Tuple[int, float, bool]:
        """Confidence-weighted majority voting."""
        # Filter low-confidence predictions
        filtered = [
            (p, c) for p, c in zip(predictions, confidences)
            if c >= self.min_confidence
        ]

        if not filtered:
            # All low confidence - return most recent
            return predictions[-1], confidences[-1], False

        # Weighted vote
        class_scores = {}
        for pred, conf in filtered:
            if pred not in class_scores:
                class_scores[pred] = 0
            class_scores[pred] += conf

        winner = max(class_scores, key=class_scores.get)
        total_conf = sum(c for p, c in filtered)
        winner_conf = class_scores[winner] / total_conf if total_conf > 0 else 0

        # Stability: high-confidence predictions agree
        high_conf_preds = [p for p, c in filtered if c >= 0.7]
        is_stable = len(high_conf_preds) >= 3 and len(set(high_conf_preds)) == 1

        return winner, winner_conf, is_stable

    def clear(self) -> None:
        """Clear the prediction buffer."""
        self.predictions.clear()
        self.confidences.clear()


class GestureSegmenter:
    """
    Segments continuous signing into discrete gestures.

    The Problem:
    -----------
    In real signing, there are no clear "start" and "end" markers.
    We need to detect:
    1. When a sign starts (hand becomes active)
    2. When a sign ends (transition to next sign or idle)
    3. When to commit a sign to the output

    Our Approach:
    -------------
    We use a state machine with three states:
    - IDLE: No clear gesture detected
    - ACTIVE: Currently performing a sign
    - TRANSITION: Moving between signs

    State Transitions:
    -----------------
    IDLE → ACTIVE: High confidence prediction for N frames
    ACTIVE → TRANSITION: Confidence drops or prediction changes
    TRANSITION → ACTIVE: New stable prediction emerges
    TRANSITION → IDLE: Extended low confidence
    """

    class State:
        IDLE = 'idle'
        ACTIVE = 'active'
        TRANSITION = 'transition'

    def __init__(
        self,
        idle_threshold: float = 0.3,
        min_gesture_frames: int = 10,
        max_transition_frames: int = 15
    ):
        """
        Initialize the segmenter.

        Args:
            idle_threshold: Confidence below this = idle
            min_gesture_frames: Minimum frames for valid gesture
            max_transition_frames: Max frames in transition before going idle
        """
        self.idle_threshold = idle_threshold
        self.min_gesture_frames = min_gesture_frames
        self.max_transition_frames = max_transition_frames

        # State tracking
        self.state = self.State.IDLE
        self.current_sign = None
        self.frame_count = 0
        self.transition_count = 0

        # Committed signs
        self.committed_signs: List[str] = []

    def update(
        self,
        label: str,
        confidence: float,
        is_stable: bool
    ) -> Optional[str]:
        """
        Update segmenter with new prediction.

        Args:
            label: Predicted sign label
            confidence: Prediction confidence
            is_stable: Whether prediction is stable (from smoother)

        Returns:
            Committed sign label if a sign was completed, else None
        """
        committed = None

        if self.state == self.State.IDLE:
            # Check for gesture start
            if confidence > self.idle_threshold and label != 'nothing':
                self.state = self.State.ACTIVE
                self.current_sign = label
                self.frame_count = 1

        elif self.state == self.State.ACTIVE:
            if label == self.current_sign and confidence > self.idle_threshold:
                # Same sign continues
                self.frame_count += 1
            elif confidence < self.idle_threshold or label == 'nothing':
                # Gesture ending - commit if long enough
                if self.frame_count >= self.min_gesture_frames:
                    committed = self.current_sign
                    self.committed_signs.append(committed)
                self.state = self.State.IDLE
                self.current_sign = None
                self.frame_count = 0
            else:
                # Different sign - enter transition
                self.state = self.State.TRANSITION
                self.transition_count = 1
                # Commit current sign if valid
                if self.frame_count >= self.min_gesture_frames:
                    committed = self.current_sign
                    self.committed_signs.append(committed)
                self.current_sign = label
                self.frame_count = 1

        elif self.state == self.State.TRANSITION:
            self.transition_count += 1

            if label == self.current_sign and is_stable:
                # New sign stabilized
                self.state = self.State.ACTIVE
                self.transition_count = 0
            elif self.transition_count > self.max_transition_frames:
                # Transition timeout - go idle
                self.state = self.State.IDLE
                self.current_sign = None
                self.frame_count = 0
                self.transition_count = 0

        return committed

    def get_current_sign(self) -> Optional[str]:
        """Get the currently active sign (if any)."""
        if self.state == self.State.ACTIVE:
            return self.current_sign
        return None

    def get_committed_signs(self) -> List[str]:
        """Get all committed signs."""
        return self.committed_signs.copy()

    def clear(self) -> None:
        """Reset the segmenter."""
        self.state = self.State.IDLE
        self.current_sign = None
        self.frame_count = 0
        self.transition_count = 0
        self.committed_signs.clear()


class GestureRecognizer:
    """
    Main class for real-time gesture recognition.

    This is the primary interface for using the sign language model
    in real-time applications.

    Usage:
    ------
    >>> recognizer = GestureRecognizer(model_path='model.pt')
    >>> # In your frame processing loop:
    >>> result = recognizer.process_landmarks(landmarks)
    >>> if result:
    ...     print(f"Detected: {result.label} ({result.confidence:.2f})")

    For static (single-frame) recognition:
    >>> recognizer = GestureRecognizer(model_path='model.pt', mode='static')

    For temporal (sequence) recognition:
    >>> recognizer = GestureRecognizer(model_path='model.pt', mode='temporal')
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        mode: str = 'static',  # 'static' or 'temporal'
        device: str = 'auto',
        sequence_length: int = 30,
        smoothing_window: int = 5,
        min_confidence: float = 0.5
    ):
        """
        Initialize the gesture recognizer.

        Args:
            model_path: Path to saved model weights (None for untrained)
            mode: 'static' for single-frame, 'temporal' for sequence
            device: 'cpu', 'cuda', or 'auto'
            sequence_length: Number of frames for temporal model
            smoothing_window: Size of prediction smoothing window
            min_confidence: Minimum confidence to report prediction
        """
        self.mode = mode
        self.sequence_length = sequence_length
        self.min_confidence = min_confidence

        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"GestureRecognizer using device: {self.device}")

        # Initialize model
        if mode == 'static':
            self.model = SignLanguageModelStatic(num_classes=29)
        else:
            self.model = SignLanguageModel(num_classes=29)

        # Load weights if provided
        if model_path:
            self._load_model(model_path)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Frame buffer for temporal model
        self.frame_buffer = deque(maxlen=sequence_length)

        # Prediction smoothing
        self.smoother = PredictionSmoother(
            window_size=smoothing_window,
            method='confidence_vote',
            min_confidence=min_confidence * 0.5  # Lower threshold for smoothing
        )

        # Gesture segmentation
        self.segmenter = GestureSegmenter(
            idle_threshold=min_confidence * 0.3,
            min_gesture_frames=10
        )

        # Performance tracking
        self._inference_times = deque(maxlen=100)

    def _load_model(self, model_path: str) -> None:
        """Load model weights from file."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def process_landmarks(
        self,
        landmarks: np.ndarray,
        normalize: bool = True
    ) -> Optional[PredictionResult]:
        """
        Process hand landmarks and return prediction.

        Args:
            landmarks: Hand landmarks array of shape (21, 3)
            normalize: Whether to normalize landmarks (recommended)

        Returns:
            PredictionResult if confident prediction, else None
        """
        if landmarks is None:
            return None

        start_time = time.time()

        # Normalize landmarks
        if normalize:
            landmarks = self._normalize_landmarks(landmarks)

        # Convert to tensor
        landmarks_tensor = torch.FloatTensor(landmarks)

        if self.mode == 'static':
            result = self._process_static(landmarks_tensor)
        else:
            result = self._process_temporal(landmarks_tensor)

        # Track inference time
        inference_time = time.time() - start_time
        self._inference_times.append(inference_time)

        return result

    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks for model input.

        Normalization steps:
        1. Center on wrist (landmark 0)
        2. Scale by palm size
        3. This makes the model position and scale invariant
        """
        # Center on wrist
        centered = landmarks - landmarks[0]

        # Scale by palm size (wrist to middle finger MCP)
        palm_size = np.linalg.norm(landmarks[9] - landmarks[0])
        if palm_size > 0:
            normalized = centered / palm_size
        else:
            normalized = centered

        return normalized

    def _process_static(self, landmarks: torch.Tensor) -> Optional[PredictionResult]:
        """Process single frame with static model."""
        # Add batch dimension
        x = landmarks.unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)

        probs_np = probs.cpu().numpy()[0]
        class_idx = np.argmax(probs_np)
        confidence = probs_np[class_idx]

        # Apply smoothing
        self.smoother.add(class_idx, confidence)
        smoothed_idx, smoothed_conf, is_stable = self.smoother.get_smoothed()

        # Get label
        label = self.model.get_label(smoothed_idx)

        # Filter low confidence
        if smoothed_conf < self.min_confidence:
            return None

        return PredictionResult(
            label=label,
            confidence=smoothed_conf,
            class_idx=smoothed_idx,
            probabilities=probs_np,
            timestamp=time.time(),
            is_stable=is_stable
        )

    def _process_temporal(self, landmarks: torch.Tensor) -> Optional[PredictionResult]:
        """Process frame sequence with temporal model."""
        # Add to buffer
        self.frame_buffer.append(landmarks)

        # Need full sequence
        if len(self.frame_buffer) < self.sequence_length:
            return None

        # Stack sequence
        sequence = torch.stack(list(self.frame_buffer))
        x = sequence.unsqueeze(0).to(self.device)  # (1, seq_len, 21, 3)

        # Inference
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)

        probs_np = probs.cpu().numpy()[0]
        class_idx = np.argmax(probs_np)
        confidence = probs_np[class_idx]

        # Apply smoothing
        self.smoother.add(class_idx, confidence)
        smoothed_idx, smoothed_conf, is_stable = self.smoother.get_smoothed()

        # Get label
        label = self.model.get_label(smoothed_idx)

        # Filter low confidence
        if smoothed_conf < self.min_confidence:
            return None

        return PredictionResult(
            label=label,
            confidence=smoothed_conf,
            class_idx=smoothed_idx,
            probabilities=probs_np,
            timestamp=time.time(),
            is_stable=is_stable
        )

    def get_average_inference_time(self) -> float:
        """Get average inference time in milliseconds."""
        if not self._inference_times:
            return 0.0
        return np.mean(self._inference_times) * 1000

    def reset(self) -> None:
        """Reset all buffers and state."""
        self.frame_buffer.clear()
        self.smoother.clear()
        self.segmenter.clear()

    def get_committed_signs(self) -> List[str]:
        """Get list of committed (completed) signs."""
        return self.segmenter.get_committed_signs()


# =============================================================================
# Testing
# =============================================================================

def test_inference():
    """Test inference with random data."""
    print("Testing GestureRecognizer...")

    # Test static mode
    print("\n--- Static Mode ---")
    recognizer = GestureRecognizer(mode='static')

    for i in range(10):
        # Random landmarks
        landmarks = np.random.randn(21, 3).astype(np.float32)
        result = recognizer.process_landmarks(landmarks)

        if result:
            print(f"Frame {i}: {result.label} ({result.confidence:.2f})")
        else:
            print(f"Frame {i}: Low confidence")

    print(f"Avg inference time: {recognizer.get_average_inference_time():.2f}ms")

    # Test temporal mode
    print("\n--- Temporal Mode ---")
    recognizer = GestureRecognizer(mode='temporal', sequence_length=10)

    for i in range(20):
        landmarks = np.random.randn(21, 3).astype(np.float32)
        result = recognizer.process_landmarks(landmarks)

        if result:
            print(f"Frame {i}: {result.label} ({result.confidence:.2f}) "
                  f"stable={result.is_stable}")
        elif i < 9:
            print(f"Frame {i}: Buffering...")
        else:
            print(f"Frame {i}: Low confidence")

    print(f"Avg inference time: {recognizer.get_average_inference_time():.2f}ms")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_inference()
