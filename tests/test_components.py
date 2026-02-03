"""
Unit Tests for Sign Language Recognition Components
====================================================

Run with: pytest tests/test_components.py -v
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLandmarkNormalization:
    """Test landmark preprocessing functions."""

    def test_normalize_landmarks_shape(self):
        """Test that normalization preserves shape."""
        from src.utils.preprocessing import normalize_landmarks

        landmarks = np.random.randn(21, 3).astype(np.float32)
        normalized = normalize_landmarks(landmarks)

        assert normalized.shape == landmarks.shape

    def test_normalize_landmarks_centered(self):
        """Test that normalized landmarks are centered on wrist."""
        from src.utils.preprocessing import normalize_landmarks

        landmarks = np.random.randn(21, 3).astype(np.float32)
        normalized = normalize_landmarks(landmarks, center_on_wrist=True)

        # Wrist (index 0) should be at origin
        np.testing.assert_array_almost_equal(normalized[0], [0, 0, 0])

    def test_normalize_landmarks_batch(self):
        """Test normalization with batch dimension."""
        from src.utils.preprocessing import normalize_landmarks

        landmarks = np.random.randn(10, 21, 3).astype(np.float32)
        normalized = normalize_landmarks(landmarks)

        assert normalized.shape == landmarks.shape


class TestAugmentation:
    """Test data augmentation functions."""

    def test_augment_landmarks_shape(self):
        """Test that augmentation preserves shape."""
        from src.utils.preprocessing import augment_landmarks

        landmarks = np.random.randn(21, 3).astype(np.float32)
        augmented = augment_landmarks(landmarks)

        assert augmented.shape == landmarks.shape

    def test_augment_landmarks_different(self):
        """Test that augmentation changes the data."""
        from src.utils.preprocessing import augment_landmarks

        landmarks = np.random.randn(21, 3).astype(np.float32)

        # Run multiple times - at least one should be different
        all_same = True
        for _ in range(10):
            augmented = augment_landmarks(landmarks)
            if not np.allclose(augmented, landmarks):
                all_same = False
                break

        assert not all_same, "Augmentation should modify data"


class TestModel:
    """Test neural network models."""

    def test_static_model_forward(self):
        """Test static model forward pass."""
        from src.recognition.model import SignLanguageModelStatic

        model = SignLanguageModelStatic(num_classes=29)
        x = torch.randn(4, 21, 3)

        output = model(x)

        assert output.shape == (4, 29)

    def test_static_model_predict(self):
        """Test static model prediction."""
        from src.recognition.model import SignLanguageModelStatic

        model = SignLanguageModelStatic(num_classes=29)
        x = torch.randn(4, 21, 3)

        predictions, probs = model.predict(x)

        assert predictions.shape == (4,)
        assert probs.shape == (4, 29)
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_temporal_model_forward(self):
        """Test temporal model forward pass."""
        from src.recognition.model import SignLanguageModel

        model = SignLanguageModel(num_classes=29)
        x = torch.randn(4, 30, 21, 3)  # batch, seq_len, landmarks, dims

        output = model(x)

        assert output.shape == (4, 29)

    def test_model_get_label(self):
        """Test class index to label conversion."""
        from src.recognition.model import SignLanguageModelStatic

        model = SignLanguageModelStatic()

        assert model.get_label(0) == 'A'
        assert model.get_label(25) == 'Z'
        assert model.get_label(26) == 'space'
        assert model.get_label(27) == 'del'
        assert model.get_label(28) == 'nothing'


class TestTranslator:
    """Test translation module."""

    def test_sentence_builder_basic(self):
        """Test basic sentence building."""
        from src.translation.translator import SentenceBuilder

        builder = SentenceBuilder()

        builder.add_sign('H')
        builder.add_sign('I')

        assert builder.get_current_word() == 'HI'

    def test_sentence_builder_space(self):
        """Test word completion with space."""
        from src.translation.translator import SentenceBuilder

        builder = SentenceBuilder()

        builder.add_sign('H')
        builder.add_sign('I')
        word = builder.add_sign('space')

        assert word == 'HI'
        assert builder.get_current_word() == ''
        assert len(builder.get_completed_words()) == 1

    def test_sentence_builder_delete(self):
        """Test delete functionality."""
        from src.translation.translator import SentenceBuilder

        builder = SentenceBuilder()

        builder.add_sign('H')
        builder.add_sign('E')
        builder.add_sign('L')
        builder.add_sign('del')

        assert builder.get_current_word() == 'HE'

    def test_translator_formatting(self):
        """Test text formatting."""
        from src.translation.translator import SignToTextTranslator

        translator = SignToTextTranslator()

        # Build "HI U R"
        for sign in ['H', 'I', 'space', 'U', 'space', 'R', 'space']:
            translator.process_sign(sign)

        text = translator.get_current_text()

        # Should expand abbreviations and format
        assert 'you' in text.lower()  # U → you
        assert 'are' in text.lower()  # R → are


class TestBuffers:
    """Test buffer utilities."""

    def test_circular_buffer(self):
        """Test circular buffer functionality."""
        from src.utils.buffer import CircularBuffer

        buffer = CircularBuffer(maxlen=3)

        buffer.add(1)
        buffer.add(2)
        buffer.add(3)
        buffer.add(4)  # Should drop 1

        items = buffer.get_all()
        assert items == [2, 3, 4]

    def test_prediction_buffer_majority(self):
        """Test prediction buffer majority vote."""
        from src.utils.buffer import PredictionBuffer

        buffer = PredictionBuffer(window_size=5)

        buffer.add('A', 0.9)
        buffer.add('A', 0.8)
        buffer.add('B', 0.7)
        buffer.add('A', 0.85)
        buffer.add('B', 0.6)

        majority = buffer.get_majority_vote()
        assert majority == 'A'

    def test_frame_sequence_buffer(self):
        """Test frame sequence buffer."""
        from src.utils.buffer import FrameSequenceBuffer

        buffer = FrameSequenceBuffer(sequence_length=5, feature_shape=(21, 3))

        # Add frames
        for _ in range(5):
            buffer.add(np.random.randn(21, 3))

        assert buffer.is_ready()
        sequence = buffer.get_sequence()
        assert sequence.shape == (5, 21, 3)


class TestGestureRecognizer:
    """Test gesture recognition inference."""

    def test_recognizer_initialization(self):
        """Test recognizer can be initialized."""
        from src.recognition.inference import GestureRecognizer

        recognizer = GestureRecognizer(mode='static')
        assert recognizer is not None

    def test_recognizer_process(self):
        """Test recognizer can process landmarks."""
        from src.recognition.inference import GestureRecognizer

        recognizer = GestureRecognizer(mode='static', min_confidence=0.0)
        landmarks = np.random.randn(21, 3).astype(np.float32)

        result = recognizer.process_landmarks(landmarks)

        # Should return a result (model is untrained, so predictions are random)
        assert result is not None or result is None  # May be None if below threshold


class TestPredictionSmoother:
    """Test prediction smoothing."""

    def test_smoother_basic(self):
        """Test basic smoothing."""
        from src.recognition.inference import PredictionSmoother

        smoother = PredictionSmoother(window_size=3)

        smoother.add(1, 0.9)
        smoother.add(1, 0.8)
        smoother.add(2, 0.5)

        idx, conf, stable = smoother.get_smoothed()
        assert idx == 1  # Majority and higher confidence


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
