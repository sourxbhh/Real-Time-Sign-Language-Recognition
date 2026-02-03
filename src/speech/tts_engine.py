"""
Text-to-Speech Engine
=====================

This module handles text-to-speech synthesis, converting translated
sign language text into spoken audio.

TTS Options Analysis:
---------------------

1. **pyttsx3** (Chosen for offline mode)
   Pros:
   ✓ Works completely offline
   ✓ Uses system voices (SAPI5 on Windows, NSSpeechSynthesizer on Mac)
   ✓ Fast, low latency
   ✓ No API costs
   Cons:
   ✗ Voice quality varies by system
   ✗ Limited voice customization

2. **gTTS (Google Text-to-Speech)**
   Pros:
   ✓ High quality voices
   ✓ Many languages
   Cons:
   ✗ Requires internet
   ✗ Higher latency (network round trip)
   ✗ Potential rate limits

3. **Coqui TTS / Piper TTS**
   Pros:
   ✓ Neural TTS quality offline
   ✓ Many voice models available
   Cons:
   ✗ Larger model files (~50-100MB)
   ✗ More complex setup
   ✗ Higher CPU/memory usage

4. **Azure/AWS/Google Cloud TTS**
   Pros:
   ✓ Best quality
   ✓ Many voices and languages
   Cons:
   ✗ Requires API key
   ✗ Costs money
   ✗ Internet required

For this project, we default to pyttsx3 for simplicity and offline
capability, with gTTS as a fallback for better quality when online.
"""

import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Callable, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Try importing TTS libraries
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 not installed - offline TTS unavailable")

try:
    from gtts import gTTS
    import io
    import pygame
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("gTTS or pygame not installed - online TTS unavailable")


class TTSBackend(Enum):
    """Available TTS backends."""
    PYTTSX3 = "pyttsx3"
    GTTS = "gtts"
    NONE = "none"


@dataclass
class TTSConfig:
    """
    TTS configuration settings.

    Attributes:
        backend: Which TTS backend to use
        rate: Speech rate (words per minute)
        volume: Volume level (0.0 to 1.0)
        voice_id: Specific voice identifier (backend-dependent)
        language: Language code (for gTTS)
        speak_letters: Whether to speak individual letters
        speak_words: Whether to speak completed words
        speak_sentences: Whether to speak completed sentences
        min_word_length: Minimum word length to speak
    """
    backend: TTSBackend = TTSBackend.PYTTSX3
    rate: int = 150
    volume: float = 1.0
    voice_id: Optional[str] = None
    language: str = "en"
    speak_letters: bool = False
    speak_words: bool = True
    speak_sentences: bool = True
    min_word_length: int = 2


class TTSEngine:
    """
    Text-to-Speech engine with asynchronous playback.

    This class provides a unified interface for TTS, handling:
    - Multiple backend support (pyttsx3, gTTS)
    - Asynchronous speech (non-blocking)
    - Speech queue management
    - Interruption handling

    Architecture:
    -------------
    ┌────────────────────────────────────────────────────────────────┐
    │                        TTSEngine                               │
    │  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐   │
    │  │   Text      │───▶│   Speech     │───▶│  Backend        │   │
    │  │   Input     │    │   Queue      │    │  (pyttsx3/gTTS) │   │
    │  └─────────────┘    └──────────────┘    └─────────────────┘   │
    │                           │                      │             │
    │                           ▼                      ▼             │
    │                    ┌──────────────┐       ┌───────────┐        │
    │                    │   Worker     │       │   Audio   │        │
    │                    │   Thread     │──────▶│   Output  │        │
    │                    └──────────────┘       └───────────┘        │
    └────────────────────────────────────────────────────────────────┘

    Usage:
    ------
    >>> engine = TTSEngine()
    >>> engine.speak("Hello world")  # Async, returns immediately
    >>> engine.speak_sync("This blocks until done")
    >>> engine.stop()  # Stop current speech

    For sign language:
    >>> engine.speak_word("HELLO")  # Only if word meets criteria
    >>> engine.speak_sentence("Hello world")  # Speaks full sentence
    """

    def __init__(self, config: Optional[TTSConfig] = None):
        """
        Initialize the TTS engine.

        Args:
            config: TTS configuration (uses defaults if None)
        """
        self.config = config or TTSConfig()

        # Speech queue for async operation
        self._queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

        # State tracking
        self._is_speaking = False
        self._current_text = ""

        # Callbacks
        self._on_start: Optional[Callable[[str], None]] = None
        self._on_end: Optional[Callable[[str], None]] = None

        # Initialize backend
        self._backend = None
        self._init_backend()

        # Start worker thread
        self._start_worker()

        logger.info(f"TTSEngine initialized with backend: {self.config.backend.value}")

    def _init_backend(self) -> None:
        """Initialize the selected TTS backend."""
        if self.config.backend == TTSBackend.PYTTSX3:
            if PYTTSX3_AVAILABLE:
                try:
                    self._backend = pyttsx3.init()
                    self._configure_pyttsx3()
                except Exception as e:
                    logger.error(f"Failed to initialize pyttsx3: {e}")
                    self.config.backend = TTSBackend.NONE
            else:
                logger.warning("pyttsx3 not available, falling back to none")
                self.config.backend = TTSBackend.NONE

        elif self.config.backend == TTSBackend.GTTS:
            if GTTS_AVAILABLE:
                try:
                    pygame.mixer.init()
                except Exception as e:
                    logger.error(f"Failed to initialize pygame mixer: {e}")
                    self.config.backend = TTSBackend.NONE
            else:
                logger.warning("gTTS not available, falling back to none")
                self.config.backend = TTSBackend.NONE

    def _configure_pyttsx3(self) -> None:
        """Configure pyttsx3 settings."""
        if self._backend is None:
            return

        try:
            # Set rate
            self._backend.setProperty('rate', self.config.rate)

            # Set volume
            self._backend.setProperty('volume', self.config.volume)

            # Set voice if specified
            if self.config.voice_id:
                self._backend.setProperty('voice', self.config.voice_id)
            else:
                # Try to get a good default voice
                voices = self._backend.getProperty('voices')
                if voices:
                    # Prefer English voice
                    for voice in voices:
                        if 'english' in voice.name.lower():
                            self._backend.setProperty('voice', voice.id)
                            break

        except Exception as e:
            logger.error(f"Error configuring pyttsx3: {e}")

    def _start_worker(self) -> None:
        """Start the background worker thread."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """
        Background worker that processes speech queue.

        This runs in a separate thread to prevent blocking the main
        application while speech is being generated and played.
        """
        while self._running:
            try:
                # Get text from queue (with timeout for clean shutdown)
                text = self._queue.get(timeout=0.1)

                if text is None:  # Shutdown signal
                    break

                self._is_speaking = True
                self._current_text = text

                # Notify start
                if self._on_start:
                    self._on_start(text)

                # Speak based on backend
                if self.config.backend == TTSBackend.PYTTSX3:
                    self._speak_pyttsx3(text)
                elif self.config.backend == TTSBackend.GTTS:
                    self._speak_gtts(text)
                else:
                    logger.warning(f"No TTS backend available, skipping: {text}")

                # Notify end
                if self._on_end:
                    self._on_end(text)

                self._is_speaking = False
                self._current_text = ""
                self._queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS worker error: {e}")
                self._is_speaking = False

    def _speak_pyttsx3(self, text: str) -> None:
        """Speak using pyttsx3 backend."""
        if self._backend is None:
            return

        try:
            self._backend.say(text)
            self._backend.runAndWait()
        except Exception as e:
            logger.error(f"pyttsx3 speak error: {e}")

    def _speak_gtts(self, text: str) -> None:
        """Speak using gTTS backend."""
        if not GTTS_AVAILABLE:
            return

        try:
            # Generate speech
            tts = gTTS(text=text, lang=self.config.language)

            # Save to memory buffer
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)

            # Play with pygame
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.play()

            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"gTTS speak error: {e}")

    def speak(self, text: str) -> None:
        """
        Speak text asynchronously (non-blocking).

        Args:
            text: Text to speak

        The text is added to a queue and spoken in the background.
        This method returns immediately.
        """
        if not text or self.config.backend == TTSBackend.NONE:
            return

        self._queue.put(text)
        logger.debug(f"Queued for speech: {text}")

    def speak_sync(self, text: str, timeout: float = 10.0) -> bool:
        """
        Speak text synchronously (blocking).

        Args:
            text: Text to speak
            timeout: Maximum time to wait for speech to complete

        Returns:
            True if speech completed, False if timed out
        """
        if not text or self.config.backend == TTSBackend.NONE:
            return False

        self._queue.put(text)

        # Wait for queue to be empty
        start_time = time.time()
        while not self._queue.empty() or self._is_speaking:
            if time.time() - start_time > timeout:
                return False
            time.sleep(0.05)

        return True

    def speak_letter(self, letter: str) -> None:
        """
        Speak a single letter (if enabled in config).

        Args:
            letter: Single letter to speak
        """
        if not self.config.speak_letters:
            return

        # Don't speak empty or single-space
        if letter and letter.strip():
            self.speak(letter)

    def speak_word(self, word: str) -> None:
        """
        Speak a completed word (if enabled and meets criteria).

        Args:
            word: Word to speak
        """
        if not self.config.speak_words:
            return

        # Check minimum length
        if len(word) < self.config.min_word_length:
            return

        self.speak(word)

    def speak_sentence(self, sentence: str) -> None:
        """
        Speak a complete sentence (if enabled).

        Args:
            sentence: Sentence to speak
        """
        if not self.config.speak_sentences:
            return

        if sentence.strip():
            self.speak(sentence)

    def stop(self) -> None:
        """Stop current speech and clear queue."""
        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break

        # Stop current speech
        if self.config.backend == TTSBackend.PYTTSX3 and self._backend:
            try:
                self._backend.stop()
            except Exception:
                pass
        elif self.config.backend == TTSBackend.GTTS and GTTS_AVAILABLE:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

        self._is_speaking = False
        logger.debug("Speech stopped")

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._is_speaking

    def get_queue_size(self) -> int:
        """Get number of items in speech queue."""
        return self._queue.qsize()

    def set_callbacks(
        self,
        on_start: Optional[Callable[[str], None]] = None,
        on_end: Optional[Callable[[str], None]] = None
    ) -> None:
        """
        Set callbacks for speech events.

        Args:
            on_start: Called when speech starts, with text as argument
            on_end: Called when speech ends, with text as argument
        """
        self._on_start = on_start
        self._on_end = on_end

    def get_available_voices(self) -> List[dict]:
        """
        Get list of available voices.

        Returns:
            List of voice info dictionaries
        """
        voices = []

        if self.config.backend == TTSBackend.PYTTSX3 and self._backend:
            try:
                for voice in self._backend.getProperty('voices'):
                    voices.append({
                        'id': voice.id,
                        'name': voice.name,
                        'languages': voice.languages,
                        'gender': getattr(voice, 'gender', 'unknown')
                    })
            except Exception as e:
                logger.error(f"Error getting voices: {e}")

        return voices

    def set_voice(self, voice_id: str) -> bool:
        """
        Set the voice by ID.

        Args:
            voice_id: Voice identifier from get_available_voices()

        Returns:
            True if successful
        """
        if self.config.backend == TTSBackend.PYTTSX3 and self._backend:
            try:
                self._backend.setProperty('voice', voice_id)
                self.config.voice_id = voice_id
                return True
            except Exception as e:
                logger.error(f"Error setting voice: {e}")

        return False

    def set_rate(self, rate: int) -> None:
        """Set speech rate (words per minute)."""
        self.config.rate = rate
        if self.config.backend == TTSBackend.PYTTSX3 and self._backend:
            self._backend.setProperty('rate', rate)

    def set_volume(self, volume: float) -> None:
        """Set volume (0.0 to 1.0)."""
        self.config.volume = max(0.0, min(1.0, volume))
        if self.config.backend == TTSBackend.PYTTSX3 and self._backend:
            self._backend.setProperty('volume', self.config.volume)

    def shutdown(self) -> None:
        """Shutdown the TTS engine and release resources."""
        self._running = False
        self._queue.put(None)  # Signal worker to stop

        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)

        if self.config.backend == TTSBackend.PYTTSX3 and self._backend:
            try:
                self._backend.stop()
            except Exception:
                pass

        logger.info("TTSEngine shutdown complete")

    def __enter__(self) -> 'TTSEngine':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()


# =============================================================================
# Testing
# =============================================================================

def test_tts():
    """Test TTS engine."""
    print("Testing TTS Engine...")

    config = TTSConfig(
        speak_words=True,
        speak_sentences=True,
        rate=150
    )

    with TTSEngine(config) as engine:
        print(f"Backend: {engine.config.backend.value}")

        # List available voices
        voices = engine.get_available_voices()
        print(f"\nAvailable voices: {len(voices)}")
        for voice in voices[:3]:  # Show first 3
            print(f"  - {voice['name']}")

        # Test speaking
        print("\nTesting speech...")

        # Word
        print("Speaking word 'Hello'...")
        engine.speak_word("Hello")
        time.sleep(1)

        # Sentence
        print("Speaking sentence...")
        engine.speak_sentence("Hello, I am a sign language recognition system.")
        time.sleep(4)

        # Multiple queued
        print("Testing queue...")
        engine.speak("First")
        engine.speak("Second")
        engine.speak("Third")
        print(f"Queue size: {engine.get_queue_size()}")
        time.sleep(3)

        print("\nTest complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_tts()
