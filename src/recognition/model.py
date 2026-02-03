"""
Sign Language Recognition Model
================================

This module contains the neural network architecture for recognizing sign language
gestures from hand landmarks.

Model Architecture Decision: Landmark-Based vs Image-Based
==========================================================

We use a LANDMARK-BASED approach rather than processing raw images. Here's why:

Landmark-Based (Our Choice):
---------------------------
Pros:
✓ Much smaller input (63 values vs 224×224×3 = 150,528 pixels)
✓ Faster inference (crucial for real-time)
✓ More robust to lighting, background, skin tone variations
✓ Easier to augment (geometric transformations)
✓ Model can be smaller (less data to process)
✓ Interpretable features (we know what each input means)

Cons:
✗ Loses texture information (relevant for some signs)
✗ Depends on MediaPipe accuracy
✗ Hand must be clearly visible for MediaPipe to work

Image-Based Alternative:
-----------------------
Pros:
✓ Captures texture and appearance
✓ Can work even with partial hand visibility
✓ End-to-end learning (no separate detection step)

Cons:
✗ Requires much larger model (CNN backbone)
✗ Slower inference
✗ Needs more training data
✗ Sensitive to lighting, background, skin tone

For a real-time system on consumer hardware, landmark-based is clearly superior.

Model Architecture
==================

We use a CNN+LSTM architecture for temporal gesture recognition:

┌────────────────────────────────────────────────────────────────────────────┐
│                        SIGN LANGUAGE MODEL                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Input: Sequence of landmarks (batch, seq_len, 21, 3)                      │
│                           │                                                │
│                           ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                    LANDMARK CNN (per frame)                          │ │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────────────┐   │ │
│  │  │ Conv1D  │───▶│ Conv1D  │───▶│ Conv1D  │───▶│ Global Avg Pool │   │ │
│  │  │  64ch   │    │  128ch  │    │  256ch  │    │                 │   │ │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────────────┘   │ │
│  │       │              │              │                │              │ │
│  │   BatchNorm      BatchNorm      BatchNorm       Flatten             │ │
│  │   ReLU           ReLU           ReLU                │              │ │
│  │   Dropout        Dropout        Dropout             │              │ │
│  │                                                     │              │ │
│  │  Output: Frame features (batch, seq_len, 256)       │              │ │
│  └─────────────────────────────────────────────────────┴──────────────┘ │
│                           │                                                │
│                           ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                    TEMPORAL LSTM                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐│ │
│  │  │ Bidirectional LSTM (256 hidden, 2 layers)                       ││ │
│  │  │                                                                  ││ │
│  │  │  ←──────────────────────────────────────────────────────────→  ││ │
│  │  │  [h₀] [h₁] [h₂] ... [h_{T-1}]                                  ││ │
│  │  └─────────────────────────────────────────────────────────────────┘│ │
│  │                           │                                          │ │
│  │  Output: Last hidden state (batch, 512)  [256 × 2 for bidirectional]│ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                           │                                                │
│                           ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                    CLASSIFICATION HEAD                               │ │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐                          │ │
│  │  │ Linear  │───▶│ Linear  │───▶│ Linear  │  → Softmax (29 classes) │ │
│  │  │   256   │    │   128   │    │num_class│                          │ │
│  │  └─────────┘    └─────────┘    └─────────┘                          │ │
│  │       │              │                                               │ │
│  │   BatchNorm      BatchNorm                                           │ │
│  │   ReLU           ReLU                                                │ │
│  │   Dropout(0.4)   Dropout(0.4)                                        │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  Output: Class probabilities (batch, num_classes)                          │
└────────────────────────────────────────────────────────────────────────────┘

Why CNN + LSTM?
===============

1. CNN extracts spatial features from hand landmarks:
   - Relative finger positions
   - Hand shape patterns
   - Finger spread/curl

2. LSTM captures temporal patterns:
   - Motion trajectory
   - Gesture dynamics
   - Sign boundaries

This combination is ideal for sign language because:
- Static signs (ASL alphabet) use CNN features
- Dynamic signs (words like "hello") need LSTM for motion

Alternative: Transformer
========================
Transformers could replace LSTM for potentially better long-range dependencies,
but they:
- Require more data to train effectively
- Are more computationally expensive
- Don't significantly outperform LSTM for short sequences (1-2 seconds)

For a student project, CNN+LSTM is more practical and interpretable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LandmarkCNN(nn.Module):
    """
    CNN for extracting features from hand landmarks.

    This processes each frame's landmarks independently, extracting
    spatial patterns that represent hand shape.

    Architecture Details:
    ---------------------
    We use 1D convolutions treating landmarks as a "sequence" of 21 points.
    This allows the network to learn patterns like:
    - "If landmark 8 (index tip) is above landmark 5 (index MCP), finger is extended"
    - "If all tips are close to wrist, hand is in fist shape"

    Why 1D Conv instead of treating landmarks as an image?
    - Landmarks have natural ordering (finger by finger)
    - 1D conv is more efficient
    - Works well empirically for landmark data
    """

    def __init__(
        self,
        input_dim: int = 3,  # x, y, z per landmark
        num_landmarks: int = 21,
        channels: Tuple[int, ...] = (64, 128, 256),
        dropout: float = 0.3
    ):
        """
        Initialize the CNN feature extractor.

        Args:
            input_dim: Dimensions per landmark (typically 3 for x, y, z)
            num_landmarks: Number of landmarks (21 for MediaPipe hand)
            channels: Number of channels in each conv layer
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_landmarks = num_landmarks

        # Build convolutional layers
        layers = []
        in_channels = input_dim

        for out_channels in channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Global average pooling to get fixed-size output
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Output feature dimension
        self.output_dim = channels[-1]

        logger.debug(f"LandmarkCNN initialized: output_dim={self.output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, num_landmarks, input_dim)
               or (batch, seq_len, num_landmarks, input_dim) for sequences

        Returns:
            Features of shape (batch, output_dim)
            or (batch, seq_len, output_dim) for sequences
        """
        # Handle sequence input
        has_sequence = x.dim() == 4
        if has_sequence:
            batch, seq_len, num_landmarks, input_dim = x.shape
            # Reshape to process all frames at once
            x = x.view(batch * seq_len, num_landmarks, input_dim)

        # Permute for Conv1d: (batch, input_dim, num_landmarks)
        x = x.permute(0, 2, 1)

        # Apply convolutions
        x = self.conv_layers(x)

        # Global average pooling
        x = self.global_pool(x)  # (batch, channels, 1)
        x = x.squeeze(-1)  # (batch, channels)

        # Reshape back if sequence input
        if has_sequence:
            x = x.view(batch, seq_len, -1)

        return x


class TemporalLSTM(nn.Module):
    """
    LSTM for capturing temporal patterns in gesture sequences.

    Why Bidirectional?
    ------------------
    Sign language gestures often have meaning that depends on both:
    - What comes before (context)
    - What comes after (completion)

    A bidirectional LSTM processes the sequence in both directions,
    capturing this full context. This is especially important for
    word-level signs where the ending matters as much as the beginning.

    Why Multiple Layers?
    --------------------
    Stacking LSTM layers creates a hierarchy:
    - Layer 1: Low-level motion patterns (velocity, acceleration)
    - Layer 2: Higher-level gesture components (stroke, hold)

    For sign language, 2 layers usually suffice. More layers would:
    - Increase risk of overfitting
    - Slow down inference
    - Not significantly improve accuracy for short sequences
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize the temporal LSTM.

        Args:
            input_dim: Size of input features (from CNN)
            hidden_size: LSTM hidden state size
            num_layers: Number of stacked LSTM layers
            bidirectional: Use bidirectional LSTM
            dropout: Dropout between LSTM layers
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output dimension (doubled if bidirectional)
        self.output_dim = hidden_size * (2 if bidirectional else 1)

        logger.debug(f"TemporalLSTM initialized: output_dim={self.output_dim}")

    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch, seq_len, input_dim)
            return_all: If True, return all timesteps; else just last

        Returns:
            If return_all: (batch, seq_len, output_dim)
            Else: (batch, output_dim)
        """
        # LSTM forward
        output, (h_n, c_n) = self.lstm(x)

        if return_all:
            return output

        # For classification, we typically want just the last state
        # For bidirectional: concatenate last forward and first backward
        if self.bidirectional:
            # h_n shape: (num_layers * 2, batch, hidden_size)
            # Get last layer's forward and backward states
            forward_last = h_n[-2]  # Last forward state
            backward_last = h_n[-1]  # Last backward state
            return torch.cat([forward_last, backward_last], dim=1)
        else:
            return h_n[-1]


class ClassificationHead(nn.Module):
    """
    Multi-layer classification head with dropout.

    Why Multiple Layers?
    --------------------
    A single linear layer might be too simple to capture the mapping
    from temporal features to sign classes. Adding hidden layers allows
    the network to learn more complex decision boundaries.

    The architecture is:
    input → Linear → BN → ReLU → Dropout → Linear → BN → ReLU → Dropout → Linear → output

    This is sometimes called an "MLP head" and is standard practice
    in modern deep learning.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_sizes: Tuple[int, ...] = (256, 128),
        num_classes: int = 29,
        dropout: float = 0.4
    ):
        """
        Initialize classification head.

        Args:
            input_dim: Input feature dimension (from LSTM)
            hidden_sizes: Sizes of hidden layers
            num_classes: Number of output classes
            dropout: Dropout rate (higher for classification to prevent overfitting)
        """
        super().__init__()

        layers = []
        in_features = input_dim

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_features = hidden_size

        # Final classification layer
        layers.append(nn.Linear(in_features, num_classes))

        self.classifier = nn.Sequential(*layers)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch, input_dim)

        Returns:
            Logits of shape (batch, num_classes)
            Note: We return logits (pre-softmax) for use with CrossEntropyLoss
        """
        return self.classifier(x)


class SignLanguageModel(nn.Module):
    """
    Complete sign language recognition model.

    This combines all components into a single end-to-end trainable model.

    Usage:
    ------
    >>> model = SignLanguageModel(num_classes=29)
    >>> x = torch.randn(8, 30, 21, 3)  # batch=8, seq=30 frames, 21 landmarks, xyz
    >>> logits = model(x)  # (8, 29)
    >>> probs = F.softmax(logits, dim=1)

    For training:
    >>> criterion = nn.CrossEntropyLoss()
    >>> loss = criterion(logits, labels)
    """

    # Class labels for ASL alphabet + special tokens
    CLASSES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['space', 'del', 'nothing']

    def __init__(
        self,
        num_landmarks: int = 21,
        input_dim: int = 3,
        cnn_channels: Tuple[int, ...] = (64, 128, 256),
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        classifier_hidden: Tuple[int, ...] = (256, 128),
        num_classes: int = 29,
        dropout: float = 0.3
    ):
        """
        Initialize the complete model.

        Args:
            num_landmarks: Number of hand landmarks (21 for MediaPipe)
            input_dim: Dimensions per landmark (3 for x, y, z)
            cnn_channels: Channel sizes for CNN layers
            lstm_hidden: LSTM hidden state size
            lstm_layers: Number of LSTM layers
            bidirectional: Use bidirectional LSTM
            classifier_hidden: Hidden layer sizes for classifier
            num_classes: Number of sign classes (26 letters + 3 special)
            dropout: Base dropout rate
        """
        super().__init__()

        self.num_landmarks = num_landmarks
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Spatial feature extractor
        self.cnn = LandmarkCNN(
            input_dim=input_dim,
            num_landmarks=num_landmarks,
            channels=cnn_channels,
            dropout=dropout
        )

        # Temporal sequence processor
        self.lstm = TemporalLSTM(
            input_dim=self.cnn.output_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )

        # Classification head
        self.classifier = ClassificationHead(
            input_dim=self.lstm.output_dim,
            hidden_sizes=classifier_hidden,
            num_classes=num_classes,
            dropout=dropout + 0.1  # Slightly higher dropout for classifier
        )

        # Initialize weights
        self._init_weights()

        # Log model summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"SignLanguageModel initialized: {total_params:,} params "
                   f"({trainable_params:,} trainable)")

    def _init_weights(self):
        """
        Initialize model weights.

        Good initialization is crucial for training stability:
        - Xavier (Glorot) initialization for linear layers
        - Orthogonal initialization for LSTM (helps with gradient flow)
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
                        # Set forget gate bias to 1 (helps LSTM remember)
                        n = param.size(0)
                        param.data[n // 4:n // 2].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input landmarks of shape (batch, seq_len, num_landmarks, input_dim)
            return_features: If True, also return intermediate features

        Returns:
            Logits of shape (batch, num_classes)
            If return_features: tuple of (logits, cnn_features, lstm_features)
        """
        # Extract spatial features per frame
        cnn_features = self.cnn(x)  # (batch, seq_len, cnn_dim)

        # Process temporal sequence
        lstm_features = self.lstm(cnn_features)  # (batch, lstm_dim)

        # Classify
        logits = self.classifier(lstm_features)  # (batch, num_classes)

        if return_features:
            return logits, cnn_features, lstm_features
        return logits

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with probabilities.

        Args:
            x: Input landmarks

        Returns:
            Tuple of (predicted_classes, probabilities)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            predicted = torch.argmax(probs, dim=1)
        return predicted, probs

    def get_label(self, class_idx: int) -> str:
        """Convert class index to label string."""
        if 0 <= class_idx < len(self.CLASSES):
            return self.CLASSES[class_idx]
        return "unknown"


class SignLanguageModelStatic(nn.Module):
    """
    Simplified model for static (single-frame) sign recognition.

    Use this for ASL alphabet recognition where each sign is a static
    hand pose, not requiring temporal information.

    This is faster and simpler, good for:
    - Quick prototyping
    - Alphabet-only recognition
    - Resource-constrained environments
    """

    CLASSES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['space', 'del', 'nothing']

    def __init__(
        self,
        num_landmarks: int = 21,
        input_dim: int = 3,
        hidden_sizes: Tuple[int, ...] = (256, 128, 64),
        num_classes: int = 29,
        dropout: float = 0.3
    ):
        """
        Initialize static model.

        Args:
            num_landmarks: Number of hand landmarks
            input_dim: Dimensions per landmark
            hidden_sizes: MLP hidden layer sizes
            num_classes: Number of sign classes
            dropout: Dropout rate
        """
        super().__init__()

        self.num_classes = num_classes
        self.input_size = num_landmarks * input_dim  # 21 * 3 = 63

        # Simple MLP
        layers = []
        in_features = self.input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_features = hidden_size

        layers.append(nn.Linear(in_features, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input landmarks of shape (batch, num_landmarks, input_dim)

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Flatten landmarks
        x = x.view(x.size(0), -1)  # (batch, 63)
        return self.model(x)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictions with probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            predicted = torch.argmax(probs, dim=1)
        return predicted, probs

    def get_label(self, class_idx: int) -> str:
        """Convert class index to label string."""
        if 0 <= class_idx < len(self.CLASSES):
            return self.CLASSES[class_idx]
        return "unknown"


# =============================================================================
# Model Utilities
# =============================================================================

def count_parameters(model: nn.Module) -> dict:
    """
    Count model parameters.

    Returns:
        Dictionary with total, trainable, and non-trainable counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> str:
    """
    Generate a summary of the model architecture.

    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (without batch dimension)

    Returns:
        String summary of the model
    """
    summary_lines = ["=" * 70]
    summary_lines.append(f"Model: {model.__class__.__name__}")
    summary_lines.append("=" * 70)

    params = count_parameters(model)
    summary_lines.append(f"Total parameters: {params['total']:,}")
    summary_lines.append(f"Trainable parameters: {params['trainable']:,}")
    summary_lines.append(f"Non-trainable parameters: {params['non_trainable']:,}")

    # Memory estimate (rough, assumes float32)
    memory_mb = params['total'] * 4 / (1024 * 1024)
    summary_lines.append(f"Estimated model size: {memory_mb:.2f} MB")

    summary_lines.append("=" * 70)

    return "\n".join(summary_lines)


# =============================================================================
# Testing
# =============================================================================

def test_model():
    """Test the model with random input."""
    print("Testing SignLanguageModel...")

    # Create model
    model = SignLanguageModel(num_classes=29)
    print(model_summary(model, (30, 21, 3)))

    # Test forward pass with sequence input
    batch_size = 8
    seq_len = 30
    num_landmarks = 21
    input_dim = 3

    x = torch.randn(batch_size, seq_len, num_landmarks, input_dim)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    logits = model(x)
    print(f"Output shape: {logits.shape}")

    # Test prediction
    predicted, probs = model.predict(x)
    print(f"Predictions: {predicted}")
    print(f"Top class labels: {[model.get_label(p.item()) for p in predicted]}")

    # Test static model
    print("\n" + "=" * 70)
    print("Testing SignLanguageModelStatic...")

    static_model = SignLanguageModelStatic(num_classes=29)
    x_static = torch.randn(batch_size, num_landmarks, input_dim)
    print(f"Input shape: {x_static.shape}")

    logits_static = static_model(x_static)
    print(f"Output shape: {logits_static.shape}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_model()
