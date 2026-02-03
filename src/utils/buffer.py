"""
Buffer Utilities
================

Thread-safe buffers for real-time data processing.
"""

import threading
import numpy as np
from collections import deque
from typing import Optional, List, Any, Generic, TypeVar


T = TypeVar('T')


class CircularBuffer(Generic[T]):
    """
    Thread-safe circular buffer for storing recent data.

    This is useful for:
    - Frame buffering for temporal models
    - Prediction history for smoothing
    - Sliding window analysis

    Usage:
    ------
    >>> buffer = CircularBuffer(maxlen=30)
    >>> buffer.add(frame1)
    >>> buffer.add(frame2)
    >>> recent = buffer.get_all()  # Get all frames
    >>> latest = buffer.get_latest()  # Get most recent
    """

    def __init__(self, maxlen: int):
        """
        Initialize circular buffer.

        Args:
            maxlen: Maximum number of items to store
        """
        self.maxlen = maxlen
        self._buffer = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, item: T) -> None:
        """Add an item to the buffer (thread-safe)."""
        with self._lock:
            self._buffer.append(item)

    def get_all(self) -> List[T]:
        """Get all items in the buffer."""
        with self._lock:
            return list(self._buffer)

    def get_latest(self, n: int = 1) -> List[T]:
        """Get the n most recent items."""
        with self._lock:
            if n >= len(self._buffer):
                return list(self._buffer)
            return list(self._buffer)[-n:]

    def get_oldest(self, n: int = 1) -> List[T]:
        """Get the n oldest items."""
        with self._lock:
            if n >= len(self._buffer):
                return list(self._buffer)
            return list(self._buffer)[:n]

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        with self._lock:
            return len(self._buffer) >= self.maxlen

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self._buffer) == 0

    def clear(self) -> None:
        """Clear all items from buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def __iter__(self):
        with self._lock:
            return iter(list(self._buffer))


class PredictionBuffer:
    """
    Specialized buffer for prediction smoothing.

    Stores recent predictions and provides smoothed output using
    various methods (majority vote, weighted average, etc.).

    Usage:
    ------
    >>> buffer = PredictionBuffer(window_size=5)
    >>> buffer.add(prediction=1, confidence=0.9)
    >>> buffer.add(prediction=1, confidence=0.85)
    >>> buffer.add(prediction=2, confidence=0.3)
    >>> smoothed = buffer.get_smoothed()  # Returns 1 (majority)
    """

    def __init__(
        self,
        window_size: int = 5,
        min_confidence: float = 0.3
    ):
        """
        Initialize prediction buffer.

        Args:
            window_size: Number of predictions to consider
            min_confidence: Minimum confidence to include in smoothing
        """
        self.window_size = window_size
        self.min_confidence = min_confidence
        self._predictions = deque(maxlen=window_size)
        self._confidences = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def add(self, prediction: Any, confidence: float = 1.0) -> None:
        """Add a prediction with its confidence."""
        with self._lock:
            self._predictions.append(prediction)
            self._confidences.append(confidence)

    def get_majority_vote(self) -> Optional[Any]:
        """
        Get the most common prediction.

        Returns:
            Most common prediction, or None if buffer is empty
        """
        with self._lock:
            if not self._predictions:
                return None

            from collections import Counter
            counts = Counter(self._predictions)
            return counts.most_common(1)[0][0]

    def get_weighted_vote(self) -> Optional[Any]:
        """
        Get prediction weighted by confidence.

        Higher confidence predictions have more influence.
        """
        with self._lock:
            if not self._predictions:
                return None

            # Filter by minimum confidence
            valid = [
                (p, c) for p, c in zip(self._predictions, self._confidences)
                if c >= self.min_confidence
            ]

            if not valid:
                return list(self._predictions)[-1]  # Return most recent

            # Weighted vote
            scores = {}
            for pred, conf in valid:
                if pred not in scores:
                    scores[pred] = 0
                scores[pred] += conf

            return max(scores, key=scores.get)

    def get_smoothed(self, method: str = 'weighted') -> Optional[Any]:
        """
        Get smoothed prediction.

        Args:
            method: 'majority' or 'weighted'

        Returns:
            Smoothed prediction
        """
        if method == 'majority':
            return self.get_majority_vote()
        else:
            return self.get_weighted_vote()

    def get_stability(self) -> float:
        """
        Get prediction stability score (0-1).

        Higher score means predictions have been consistent.
        """
        with self._lock:
            if len(self._predictions) < 2:
                return 0.0

            from collections import Counter
            counts = Counter(self._predictions)
            most_common_count = counts.most_common(1)[0][1]

            return most_common_count / len(self._predictions)

    def get_average_confidence(self) -> float:
        """Get average confidence of predictions."""
        with self._lock:
            if not self._confidences:
                return 0.0
            return sum(self._confidences) / len(self._confidences)

    def clear(self) -> None:
        """Clear all predictions."""
        with self._lock:
            self._predictions.clear()
            self._confidences.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._predictions)


class FrameSequenceBuffer:
    """
    Buffer for storing frame sequences for temporal models.

    Handles numpy arrays efficiently and provides the sequence
    in the format expected by the model.
    """

    def __init__(self, sequence_length: int = 30, feature_shape: tuple = (21, 3)):
        """
        Initialize frame sequence buffer.

        Args:
            sequence_length: Number of frames in sequence
            feature_shape: Shape of features per frame (e.g., (21, 3) for landmarks)
        """
        self.sequence_length = sequence_length
        self.feature_shape = feature_shape
        self._buffer = deque(maxlen=sequence_length)
        self._lock = threading.Lock()

    def add(self, frame_features: np.ndarray) -> None:
        """Add frame features to buffer."""
        with self._lock:
            self._buffer.append(frame_features.copy())

    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Get the current sequence as numpy array.

        Returns:
            Array of shape (sequence_length, *feature_shape)
            or None if buffer isn't full
        """
        with self._lock:
            if len(self._buffer) < self.sequence_length:
                return None

            return np.stack(list(self._buffer), axis=0)

    def get_partial_sequence(self) -> np.ndarray:
        """
        Get current frames, padding if needed.

        Returns:
            Array of shape (sequence_length, *feature_shape)
            with zero padding for missing frames
        """
        with self._lock:
            current = list(self._buffer)
            n_current = len(current)

            if n_current == 0:
                return np.zeros((self.sequence_length,) + self.feature_shape)

            if n_current >= self.sequence_length:
                return np.stack(current[-self.sequence_length:], axis=0)

            # Pad with zeros
            padding_shape = (self.sequence_length - n_current,) + self.feature_shape
            padding = np.zeros(padding_shape)
            frames = np.stack(current, axis=0)

            return np.concatenate([padding, frames], axis=0)

    def is_ready(self) -> bool:
        """Check if buffer has full sequence."""
        with self._lock:
            return len(self._buffer) >= self.sequence_length

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)
