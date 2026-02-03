"""
Streamlit User Interface for Sign Language Recognition
======================================================

This module provides a clean, interactive web interface for the sign
language recognition system using Streamlit.

Why Streamlit?
--------------
1. **Rapid Development**: Build UIs with pure Python
2. **Real-time Updates**: Built-in support for live data
3. **No Frontend Experience Needed**: No HTML/CSS/JS required
4. **Easy Deployment**: Can be deployed to Streamlit Cloud

Alternative UI Options:
-----------------------
1. **Gradio**: Similar to Streamlit, more ML-focused
2. **Flask + HTML/JS**: More control, more work
3. **PyQt/Tkinter**: Desktop app, more complex
4. **React + FastAPI**: Professional web app, most work

For a student project, Streamlit provides the best balance of
features and development speed.

Running the App:
----------------
$ streamlit run ui/app.py

Or from Python:
>>> from ui.app import main
>>> main()
"""

import streamlit as st
import cv2
import numpy as np
import time
from pathlib import Path
import sys
import threading
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.webcam.capture import WebcamCapture
from src.detection.landmark_detector import LandmarkDetector
from src.recognition.inference import GestureRecognizer
from src.translation.translator import SignToTextTranslator
from src.speech.tts_engine import TTSEngine, TTSConfig


# =============================================================================
# Session State Management
# =============================================================================

def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.running = False
        st.session_state.current_sign = ""
        st.session_state.current_word = ""
        st.session_state.current_sentence = ""
        st.session_state.history = []
        st.session_state.fps = 0
        st.session_state.confidence = 0


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Sign Language Recognition",
        page_icon="👋",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    init_session_state()

    # Custom CSS
    st.markdown("""
    <style>
    .big-font {
        font-size: 48px !important;
        font-weight: bold;
        text-align: center;
    }
    .medium-font {
        font-size: 24px !important;
        text-align: center;
    }
    .stVideo {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🤟 Real-Time Sign Language Recognition")
    st.markdown("---")

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")

        # Camera settings
        st.subheader("📷 Camera")
        camera_id = st.selectbox("Camera Device", [0, 1, 2], index=0)
        resolution = st.selectbox(
            "Resolution",
            ["640x480", "1280x720", "1920x1080"],
            index=0
        )
        width, height = map(int, resolution.split('x'))

        # Detection settings
        st.subheader("🖐️ Detection")
        min_confidence = st.slider(
            "Min Detection Confidence",
            0.1, 1.0, 0.7, 0.05
        )
        show_landmarks = st.checkbox("Show Landmarks", value=True)
        show_skeleton = st.checkbox("Show Skeleton", value=True)

        # Recognition settings
        st.subheader("🧠 Recognition")
        model_type = st.selectbox(
            "Model Type",
            ["Static (Letters)", "Temporal (Words)"],
            index=0
        )
        recognition_threshold = st.slider(
            "Recognition Threshold",
            0.1, 1.0, 0.5, 0.05
        )

        # TTS settings
        st.subheader("🔊 Speech")
        enable_tts = st.checkbox("Enable Text-to-Speech", value=True)
        speak_letters = st.checkbox("Speak Letters", value=False)
        speak_words = st.checkbox("Speak Words", value=True)
        speech_rate = st.slider("Speech Rate", 100, 300, 150, 10)

        # About
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("""
        This system recognizes American Sign Language (ASL)
        gestures in real-time using computer vision and
        deep learning.

        **Supported Signs:**
        - ASL Alphabet (A-Z)
        - Space, Delete, Nothing

        **How to use:**
        1. Position your hand in the camera view
        2. Make ASL letter signs
        3. Use 'space' gesture to complete words
        4. The system will translate and speak
        """)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Live Camera Feed")

        # Video placeholder
        video_placeholder = st.empty()

        # Control buttons
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            start_btn = st.button("▶️ Start", type="primary", use_container_width=True)
        with btn_col2:
            stop_btn = st.button("⏹️ Stop", use_container_width=True)
        with btn_col3:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)

    with col2:
        st.subheader("📝 Recognition Output")

        # Current sign display
        st.markdown("**Current Sign:**")
        sign_placeholder = st.empty()
        sign_placeholder.markdown(
            f"<p class='big-font'>{st.session_state.current_sign or '—'}</p>",
            unsafe_allow_html=True
        )

        # Confidence display
        confidence_placeholder = st.empty()
        confidence_placeholder.progress(0)

        # Current word
        st.markdown("**Current Word:**")
        word_placeholder = st.empty()
        word_placeholder.markdown(
            f"<p class='medium-font'>{st.session_state.current_word or '—'}</p>",
            unsafe_allow_html=True
        )

        # Current sentence
        st.markdown("**Translated Text:**")
        sentence_placeholder = st.empty()
        sentence_placeholder.text_area(
            label="Sentence",
            value=st.session_state.current_sentence,
            height=100,
            disabled=True,
            label_visibility="collapsed"
        )

        # Stats
        st.markdown("---")
        stats_placeholder = st.empty()
        stats_placeholder.markdown(f"**FPS:** {st.session_state.fps}")

    # History section
    st.markdown("---")
    st.subheader("📜 Translation History")
    history_placeholder = st.empty()

    # Handle button clicks
    if clear_btn:
        st.session_state.current_sign = ""
        st.session_state.current_word = ""
        st.session_state.current_sentence = ""
        st.session_state.history = []
        st.rerun()

    # Main recognition loop
    if start_btn or st.session_state.running:
        st.session_state.running = True

        run_recognition_loop(
            camera_id=camera_id,
            width=width,
            height=height,
            min_confidence=min_confidence,
            show_landmarks=show_landmarks,
            show_skeleton=show_skeleton,
            model_type=model_type,
            recognition_threshold=recognition_threshold,
            enable_tts=enable_tts,
            speak_letters=speak_letters,
            speak_words=speak_words,
            speech_rate=speech_rate,
            video_placeholder=video_placeholder,
            sign_placeholder=sign_placeholder,
            confidence_placeholder=confidence_placeholder,
            word_placeholder=word_placeholder,
            sentence_placeholder=sentence_placeholder,
            stats_placeholder=stats_placeholder,
            history_placeholder=history_placeholder,
            stop_btn=stop_btn
        )

    if stop_btn:
        st.session_state.running = False
        st.rerun()


def run_recognition_loop(
    camera_id: int,
    width: int,
    height: int,
    min_confidence: float,
    show_landmarks: bool,
    show_skeleton: bool,
    model_type: str,
    recognition_threshold: float,
    enable_tts: bool,
    speak_letters: bool,
    speak_words: bool,
    speech_rate: int,
    video_placeholder,
    sign_placeholder,
    confidence_placeholder,
    word_placeholder,
    sentence_placeholder,
    stats_placeholder,
    history_placeholder,
    stop_btn
):
    """Run the main recognition loop."""

    # Initialize components
    capture = WebcamCapture(
        device_id=camera_id,
        width=width,
        height=height
    )

    detector = LandmarkDetector(
        max_hands=1,
        min_detection_confidence=min_confidence
    )

    recognizer = GestureRecognizer(
        mode='static' if 'Static' in model_type else 'temporal',
        min_confidence=recognition_threshold
    )

    translator = SignToTextTranslator()

    # Initialize TTS if enabled
    tts = None
    if enable_tts:
        tts_config = TTSConfig(
            speak_letters=speak_letters,
            speak_words=speak_words,
            rate=speech_rate
        )
        tts = TTSEngine(tts_config)

    # Start capture
    if not capture.start():
        st.error("Failed to start camera. Please check your webcam connection.")
        return

    try:
        frame_count = 0
        start_time = time.time()

        while st.session_state.running:
            # Read frame
            ret, frame = capture.read()
            if not ret:
                continue

            # Detect landmarks
            hands, pose = detector.detect(frame)

            # Process recognition
            result = None
            if hands:
                landmarks = hands[0].normalize()
                result = recognizer.process_landmarks(landmarks)

                # Draw landmarks
                if show_landmarks or show_skeleton:
                    frame = detector.draw_landmarks(
                        frame, hands, None,
                        draw_connections=show_skeleton
                    )

            # Update display
            if result:
                st.session_state.current_sign = result.label
                st.session_state.confidence = result.confidence

                # Process translation
                trans_result = translator.process_sign(
                    result.label,
                    timestamp=time.time(),
                    confidence=result.confidence
                )

                st.session_state.current_word = trans_result['current_word']
                st.session_state.current_sentence = trans_result['formatted_sentence']

                # Speak completed word
                if trans_result['new_word'] and tts:
                    tts.speak_word(trans_result['new_word'])

                # Update UI
                sign_placeholder.markdown(
                    f"<p class='big-font'>{result.label}</p>",
                    unsafe_allow_html=True
                )
                confidence_placeholder.progress(result.confidence)
            else:
                sign_placeholder.markdown(
                    "<p class='big-font'>—</p>",
                    unsafe_allow_html=True
                )
                confidence_placeholder.progress(0)

            # Update word and sentence
            word_placeholder.markdown(
                f"<p class='medium-font'>{st.session_state.current_word or '—'}</p>",
                unsafe_allow_html=True
            )
            sentence_placeholder.text_area(
                label="Sentence",
                value=st.session_state.current_sentence,
                height=100,
                disabled=True,
                label_visibility="collapsed"
            )

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                stats_placeholder.markdown(f"**FPS:** {fps:.1f}")

            # Add FPS text to frame
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # Add recognition info to frame
            if result:
                text = f"Sign: {result.label} ({result.confidence:.2f})"
                cv2.putText(
                    frame, text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
                )

            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display frame
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Small delay to prevent UI freezing
            time.sleep(0.01)

            # Check for stop
            if not st.session_state.running:
                break

    finally:
        # Cleanup
        capture.stop()
        detector.close()
        if tts:
            tts.shutdown()


# =============================================================================
# Alternative: Simple OpenCV-based UI
# =============================================================================

def run_opencv_ui():
    """
    Simple OpenCV-based UI as an alternative to Streamlit.

    Run this directly for a simpler, faster interface:
    $ python ui/app.py --opencv
    """
    print("Starting OpenCV-based Sign Language Recognition...")
    print("Press 'q' to quit, 'c' to clear text, 's' to toggle speech")

    # Initialize components
    capture = WebcamCapture(device_id=0, width=640, height=480)
    detector = LandmarkDetector(max_hands=1)
    recognizer = GestureRecognizer(mode='static', min_confidence=0.5)
    translator = SignToTextTranslator()
    tts = TTSEngine()

    # State
    speech_enabled = True
    current_sentence = ""

    if not capture.start():
        print("Failed to start camera!")
        return

    try:
        for frame in capture.frames():
            # Detect and recognize
            hands, _ = detector.detect(frame)

            result = None
            if hands:
                landmarks = hands[0].normalize()
                result = recognizer.process_landmarks(landmarks)
                frame = detector.draw_landmarks(frame, hands, None)

            # Translation
            if result:
                trans = translator.process_sign(result.label, time.time())
                current_sentence = trans['formatted_sentence']

                if trans['new_word'] and speech_enabled:
                    tts.speak_word(trans['new_word'])

            # Draw info panel
            panel_height = 100
            panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
            panel[:] = (40, 40, 40)

            # Draw text
            if result:
                sign_text = f"Sign: {result.label} ({result.confidence:.2f})"
                cv2.putText(panel, sign_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(panel, f"Text: {current_sentence}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Combine frame and panel
            combined = np.vstack([frame, panel])

            # Add FPS
            cv2.putText(combined, f"FPS: {capture.get_fps()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show
            cv2.imshow('Sign Language Recognition', combined)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                translator.clear()
                current_sentence = ""
            elif key == ord('s'):
                speech_enabled = not speech_enabled
                print(f"Speech: {'ON' if speech_enabled else 'OFF'}")

    finally:
        capture.stop()
        detector.close()
        tts.shutdown()
        cv2.destroyAllWindows()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Sign Language Recognition UI')
    parser.add_argument('--opencv', action='store_true',
                       help='Use simple OpenCV UI instead of Streamlit')

    args = parser.parse_args()

    if args.opencv:
        run_opencv_ui()
    else:
        main()
