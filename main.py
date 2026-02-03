#!/usr/bin/env python3
"""
Real-Time Sign Language Recognition System
===========================================

Main entry point for the sign language recognition application.

This script provides multiple ways to run the system:
1. Interactive demo with webcam
2. Streamlit web UI
3. Training mode
4. Evaluation mode

Usage:
------
# Run interactive demo
python main.py

# Run with specific options
python main.py --mode demo --camera 0

# Run Streamlit UI
python main.py --mode ui

# Train a model
python main.py --mode train --data-dir data/processed --epochs 50

# Evaluate a model
python main.py --mode eval --model models/best_model.pt
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from src.webcam.capture import WebcamCapture
from src.detection.landmark_detector import LandmarkDetector
from src.recognition.inference import GestureRecognizer
from src.translation.translator import SignToTextTranslator
from src.speech.tts_engine import TTSEngine, TTSConfig


def setup_logging(level: str = 'INFO') -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def run_demo(
    camera_id: int = 0,
    model_path: str = None,
    enable_speech: bool = True,
    min_confidence: float = 0.5
) -> None:
    """
    Run interactive demonstration with webcam.

    Controls:
    - q: Quit
    - c: Clear text
    - s: Toggle speech
    - Space: Complete sentence
    - Backspace: Delete last character

    Args:
        camera_id: Webcam device ID
        model_path: Path to trained model (optional)
        enable_speech: Enable text-to-speech output
        min_confidence: Minimum recognition confidence
    """
    print("=" * 60)
    print("   Real-Time Sign Language Recognition Demo")
    print("=" * 60)
    print("\nControls:")
    print("  q - Quit")
    print("  c - Clear text")
    print("  s - Toggle speech")
    print("  SPACE - Complete sentence and speak")
    print("  BACKSPACE - Delete last character")
    print("\nInitializing components...")

    # Initialize components
    capture = WebcamCapture(
        device_id=camera_id,
        width=640,
        height=480,
        flip_horizontal=True
    )

    detector = LandmarkDetector(
        max_hands=1,
        min_detection_confidence=0.7,
        detect_pose=False  # Disable pose for performance
    )

    recognizer = GestureRecognizer(
        model_path=model_path,
        mode='static',
        min_confidence=min_confidence
    )

    translator = SignToTextTranslator()

    # Initialize TTS
    tts = None
    if enable_speech:
        tts_config = TTSConfig(
            speak_letters=False,
            speak_words=True,
            speak_sentences=True,
            rate=150
        )
        tts = TTSEngine(tts_config)

    # State
    speech_enabled = enable_speech
    last_sign = None
    sign_start_time = 0
    stable_duration = 0.5  # Seconds to hold sign before confirming

    print("Starting camera...")
    if not capture.start():
        print("ERROR: Failed to start camera!")
        print("Please check that your webcam is connected and not in use.")
        return

    print("Ready! Show your hand to the camera.")
    print("-" * 60)

    try:
        for frame in capture.frames():
            current_time = time.time()

            # Detect hand landmarks
            hands, _ = detector.detect(frame)

            # Process recognition
            result = None
            if hands:
                # Get normalized landmarks
                landmarks = hands[0].normalize()

                # Recognize gesture
                result = recognizer.process_landmarks(landmarks)

                # Draw landmarks on frame
                frame = detector.draw_landmarks(frame, hands, None, draw_connections=True)

            # Handle stable sign detection
            if result and result.is_stable:
                if result.label != last_sign:
                    # New sign detected
                    last_sign = result.label
                    sign_start_time = current_time
                else:
                    # Same sign - check if held long enough
                    hold_time = current_time - sign_start_time
                    if hold_time >= stable_duration:
                        # Confirm sign
                        trans_result = translator.process_sign(
                            result.label,
                            timestamp=current_time,
                            confidence=result.confidence
                        )

                        # Speak completed word
                        if trans_result['new_word'] and tts and speech_enabled:
                            tts.speak_word(trans_result['new_word'])
                            print(f"Word: {trans_result['new_word']}")

                        # Reset timer
                        sign_start_time = current_time
            else:
                last_sign = None

            # Create info panel
            panel = create_info_panel(
                width=frame.shape[1],
                current_sign=result.label if result else None,
                confidence=result.confidence if result else 0,
                current_word=translator.builder.get_current_word(),
                current_sentence=translator.get_current_text(),
                fps=capture.get_fps(),
                speech_enabled=speech_enabled
            )

            # Combine frame and panel
            display = np.vstack([frame, panel])

            # Show
            cv2.imshow('Sign Language Recognition', display)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break

            elif key == ord('c'):
                translator.clear()
                print("Text cleared.")

            elif key == ord('s'):
                speech_enabled = not speech_enabled
                print(f"Speech: {'ON' if speech_enabled else 'OFF'}")

            elif key == ord(' '):  # Space
                sentence = translator.complete_sentence()
                if sentence and tts and speech_enabled:
                    tts.speak_sentence(sentence)
                print(f"\nCompleted: {sentence}")

            elif key == 8:  # Backspace
                translator.process_sign('del')

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # Cleanup
        print("\nCleaning up...")
        capture.stop()
        detector.close()
        if tts:
            tts.shutdown()
        cv2.destroyAllWindows()

    print("Demo finished.")


def create_info_panel(
    width: int,
    current_sign: str,
    confidence: float,
    current_word: str,
    current_sentence: str,
    fps: float,
    speech_enabled: bool
) -> np.ndarray:
    """Create information panel for display."""
    height = 120
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)  # Dark gray background

    # Draw separator line
    cv2.line(panel, (0, 0), (width, 0), (100, 100, 100), 2)

    # Current sign (large)
    sign_text = current_sign if current_sign else "—"
    sign_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
    cv2.putText(
        panel, sign_text, (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.5, sign_color, 3
    )

    # Confidence bar
    bar_x, bar_y = 120, 20
    bar_width, bar_height = 100, 20
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                  (100, 100, 100), 2)
    fill_width = int(bar_width * confidence)
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                  sign_color, -1)

    # Current word
    word_text = f"Word: {current_word if current_word else '—'}"
    cv2.putText(
        panel, word_text, (250, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )

    # Current sentence
    sentence_display = current_sentence[:50] + "..." if len(current_sentence) > 50 else current_sentence
    sentence_text = f"Text: {sentence_display if sentence_display else '—'}"
    cv2.putText(
        panel, sentence_text, (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
    )

    # FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        panel, fps_text, (width - 100, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2
    )

    # Speech status
    speech_text = "🔊" if speech_enabled else "🔇"
    cv2.putText(
        panel, "Speech: " + ("ON" if speech_enabled else "OFF"),
        (width - 120, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
    )

    return panel


def run_training(
    data_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_type: str
) -> None:
    """Run model training."""
    from training.train import Trainer, TrainingConfig

    config = TrainingConfig(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        model_type=model_type
    )

    trainer = Trainer(config)
    history = trainer.train()

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")


def run_evaluation(
    model_path: str,
    data_dir: str
) -> None:
    """Evaluate a trained model."""
    from training.train import Trainer, TrainingConfig
    from training.dataset import create_data_loaders

    config = TrainingConfig(data_dir=data_dir)
    trainer = Trainer(config)
    trainer.load_checkpoint(model_path)

    _, _, test_loader = create_data_loaders(data_dir)
    results = trainer.evaluate(test_loader)

    print("\nEvaluation Results:")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print("\nPer-class Accuracy:")
    for cls, acc in results['per_class_accuracy'].items():
        print(f"  {cls}: {acc:.4f}")


def run_ui() -> None:
    """Launch Streamlit UI."""
    import subprocess
    import os

    ui_path = Path(__file__).parent / 'ui' / 'app.py'
    subprocess.run(['streamlit', 'run', str(ui_path)])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Real-Time Sign Language Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run interactive demo
  python main.py --mode ui                # Run Streamlit web UI
  python main.py --mode train --epochs 50 # Train a model
  python main.py --mode eval --model best_model.pt  # Evaluate model
        """
    )

    parser.add_argument(
        '--mode', type=str, default='demo',
        choices=['demo', 'ui', 'train', 'eval'],
        help='Operation mode (default: demo)'
    )

    parser.add_argument(
        '--camera', type=int, default=0,
        help='Camera device ID (default: 0)'
    )

    parser.add_argument(
        '--model', type=str, default=None,
        help='Path to trained model weights'
    )

    parser.add_argument(
        '--data-dir', type=str, default='data/processed',
        help='Path to data directory for training/evaluation'
    )

    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Training batch size'
    )

    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate'
    )

    parser.add_argument(
        '--model-type', type=str, default='static',
        choices=['static', 'temporal'],
        help='Model architecture type'
    )

    parser.add_argument(
        '--no-speech', action='store_true',
        help='Disable text-to-speech'
    )

    parser.add_argument(
        '--confidence', type=float, default=0.5,
        help='Minimum recognition confidence (0-1)'
    )

    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Run selected mode
    if args.mode == 'demo':
        run_demo(
            camera_id=args.camera,
            model_path=args.model,
            enable_speech=not args.no_speech,
            min_confidence=args.confidence
        )

    elif args.mode == 'ui':
        run_ui()

    elif args.mode == 'train':
        run_training(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            model_type=args.model_type
        )

    elif args.mode == 'eval':
        if not args.model:
            print("ERROR: --model path required for evaluation")
            sys.exit(1)
        run_evaluation(args.model, args.data_dir)


if __name__ == '__main__':
    main()
