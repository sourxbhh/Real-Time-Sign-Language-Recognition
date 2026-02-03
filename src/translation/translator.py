"""
Sign to Text Translation Module
================================

This module converts recognized sign sequences into coherent English text.
It handles sentence construction, word building (from spelled letters),
and basic grammar cleanup.

Translation Pipeline:
---------------------
Signs → Word Builder → Sentence Buffer → Grammar Cleanup → English Text

Challenges in Sign Language Translation:
----------------------------------------

1. **Word Building**
   - ASL fingerspelling spells words letter by letter
   - Need to detect word boundaries (space sign or pause)
   - Handle corrections (del sign)

2. **Grammar Differences**
   - ASL has different grammar than English
   - Example: "STORE GO I" = "I am going to the store"
   - For student project: we handle basic word-level translation

3. **Context Sensitivity**
   - Same sign may mean different things in context
   - We use simple rule-based mapping for now

4. **Real-time Constraints**
   - Translation must happen as signs are recognized
   - Balance between responsiveness and accuracy
"""

import re
from collections import deque
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TranslationState:
    """
    Tracks the current state of translation.

    Attributes:
        current_word: Word being built from letters
        words: Completed words in current sentence
        sentences: Completed sentences
        raw_signs: All recognized signs (for debugging)
    """
    current_word: str = ""
    words: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    raw_signs: List[str] = field(default_factory=list)


class SentenceBuilder:
    """
    Builds sentences from recognized signs.

    This handles the low-level construction of text from individual
    sign predictions, including:
    - Letter concatenation for fingerspelling
    - Word boundary detection
    - Backspace/delete handling
    - Sentence completion

    Design Decisions:
    -----------------
    1. **Double-letter handling**: Consecutive same letters are treated
       as one letter (common in fingerspelling). A brief pause or
       explicit re-signing is needed for double letters.

    2. **Word boundary**: The 'space' sign or a significant pause
       indicates word boundary.

    3. **Sentence boundary**: Extended pause or specific punctuation
       signs (not implemented in basic ASL alphabet).

    Usage:
    ------
    >>> builder = SentenceBuilder()
    >>> builder.add_sign('H')
    >>> builder.add_sign('I')
    >>> builder.add_sign('space')
    >>> print(builder.get_current_sentence())  # "Hi "
    """

    # Signs that trigger special actions
    SPACE_SIGN = 'space'
    DELETE_SIGN = 'del'
    NOTHING_SIGN = 'nothing'

    def __init__(
        self,
        deduplicate_letters: bool = True,
        min_dedup_interval: float = 0.3,
        max_word_length: int = 20
    ):
        """
        Initialize the sentence builder.

        Args:
            deduplicate_letters: Ignore consecutive same letters
            min_dedup_interval: Minimum time between same letters (seconds)
            max_word_length: Maximum word length (for safety)
        """
        self.deduplicate_letters = deduplicate_letters
        self.min_dedup_interval = min_dedup_interval
        self.max_word_length = max_word_length

        # State
        self.state = TranslationState()
        self._last_sign = None
        self._last_sign_time = 0

        logger.debug("SentenceBuilder initialized")

    def add_sign(
        self,
        sign: str,
        timestamp: float = 0,
        confidence: float = 1.0
    ) -> Optional[str]:
        """
        Add a recognized sign.

        Args:
            sign: The sign label (e.g., 'A', 'space', 'del')
            timestamp: When the sign was recognized
            confidence: Recognition confidence (for future use)

        Returns:
            Completed word if a word boundary was reached, else None
        """
        if sign is None or sign == self.NOTHING_SIGN:
            return None

        # Track raw signs
        self.state.raw_signs.append(sign)

        # Handle deduplication
        if self.deduplicate_letters and sign == self._last_sign:
            time_diff = timestamp - self._last_sign_time
            if time_diff < self.min_dedup_interval:
                return None

        self._last_sign = sign
        self._last_sign_time = timestamp

        # Handle special signs
        if sign == self.SPACE_SIGN:
            return self._complete_word()
        elif sign == self.DELETE_SIGN:
            self._handle_delete()
            return None

        # Add letter to current word
        if len(sign) == 1 and sign.isalpha():
            if len(self.state.current_word) < self.max_word_length:
                self.state.current_word += sign.upper()

        return None

    def _complete_word(self) -> Optional[str]:
        """Complete the current word and return it."""
        if self.state.current_word:
            word = self.state.current_word
            self.state.words.append(word)
            self.state.current_word = ""
            logger.debug(f"Word completed: {word}")
            return word
        return None

    def _handle_delete(self) -> None:
        """Handle delete sign - remove last character or word."""
        if self.state.current_word:
            # Delete last character
            self.state.current_word = self.state.current_word[:-1]
            logger.debug("Deleted last character")
        elif self.state.words:
            # Delete last word and continue editing it
            last_word = self.state.words.pop()
            self.state.current_word = last_word[:-1] if last_word else ""
            logger.debug(f"Deleted from last word: {last_word}")

    def get_current_word(self) -> str:
        """Get the word currently being built."""
        return self.state.current_word

    def get_current_sentence(self) -> str:
        """Get the current sentence (completed words + current word)."""
        words = self.state.words.copy()
        if self.state.current_word:
            words.append(self.state.current_word)
        return ' '.join(words)

    def get_completed_words(self) -> List[str]:
        """Get list of completed words."""
        return self.state.words.copy()

    def complete_sentence(self) -> str:
        """
        Force-complete current sentence.

        Returns:
            The completed sentence
        """
        # Complete any pending word
        if self.state.current_word:
            self._complete_word()

        sentence = ' '.join(self.state.words)
        self.state.sentences.append(sentence)

        # Clear for next sentence
        self.state.words = []

        return sentence

    def clear(self) -> None:
        """Clear all state."""
        self.state = TranslationState()
        self._last_sign = None
        self._last_sign_time = 0


class SignToTextTranslator:
    """
    Translates sign language to English text with grammar cleanup.

    This is the high-level translation interface that combines:
    - Sentence building
    - Basic grammar rules
    - Common word expansions
    - Text cleanup

    Future Enhancements:
    --------------------
    For a more sophisticated system, consider:
    1. Neural machine translation (ASL → English)
    2. Language model integration (GPT-style)
    3. Context-aware word prediction
    4. Idiom detection and translation

    For a student project, rule-based translation is appropriate.
    """

    # Common ASL abbreviations and their expansions
    ABBREVIATIONS = {
        'U': 'you',
        'R': 'are',
        'UR': 'your',
        'B4': 'before',
        'BC': 'because',
        'Y': 'why',
        'THX': 'thanks',
        'PLZ': 'please',
        'HI': 'hello',
        'BYE': 'goodbye',
        'OK': 'okay',
    }

    # Common words that should be lowercase
    LOWERCASE_WORDS = {'a', 'an', 'the', 'is', 'are', 'am', 'was', 'were',
                       'be', 'been', 'to', 'for', 'of', 'in', 'on', 'at'}

    def __init__(
        self,
        auto_capitalize: bool = True,
        auto_punctuate: bool = True,
        expand_abbreviations: bool = True
    ):
        """
        Initialize the translator.

        Args:
            auto_capitalize: Automatically capitalize sentences
            auto_punctuate: Add basic punctuation
            expand_abbreviations: Expand common abbreviations
        """
        self.auto_capitalize = auto_capitalize
        self.auto_punctuate = auto_punctuate
        self.expand_abbreviations = expand_abbreviations

        # Sentence builder
        self.builder = SentenceBuilder()

        # Translation history
        self._history: List[str] = []

    def process_sign(
        self,
        sign: str,
        timestamp: float = 0,
        confidence: float = 1.0
    ) -> Dict[str, str]:
        """
        Process a recognized sign and return translation state.

        Args:
            sign: Recognized sign label
            timestamp: Recognition timestamp
            confidence: Recognition confidence

        Returns:
            Dictionary with current state:
            {
                'current_word': 'HEL',
                'current_sentence': 'HELLO WORLD HEL',
                'formatted_sentence': 'Hello world hel',
                'new_word': 'WORLD' or None
            }
        """
        new_word = self.builder.add_sign(sign, timestamp, confidence)

        current_sentence = self.builder.get_current_sentence()
        formatted = self._format_text(current_sentence)

        return {
            'current_word': self.builder.get_current_word(),
            'current_sentence': current_sentence,
            'formatted_sentence': formatted,
            'new_word': new_word
        }

    def _format_text(self, text: str) -> str:
        """
        Apply formatting rules to text.

        Steps:
        1. Expand abbreviations
        2. Apply proper capitalization
        3. Add punctuation (basic)
        """
        if not text:
            return text

        words = text.split()
        formatted_words = []

        for i, word in enumerate(words):
            # Expand abbreviations
            if self.expand_abbreviations:
                word = self.ABBREVIATIONS.get(word.upper(), word)

            # Capitalize first word, lowercase common words
            if self.auto_capitalize:
                if i == 0:
                    word = word.capitalize()
                elif word.lower() in self.LOWERCASE_WORDS:
                    word = word.lower()
                else:
                    word = word.lower()

            formatted_words.append(word)

        result = ' '.join(formatted_words)

        # Add period if this looks like a complete sentence
        if self.auto_punctuate and len(formatted_words) >= 3:
            if not result.endswith(('.', '!', '?')):
                # Don't add period if sentence is incomplete
                pass  # Could add logic to detect sentence completion

        return result

    def complete_sentence(self) -> str:
        """
        Force-complete and return the current sentence.

        Returns:
            Formatted sentence
        """
        raw_sentence = self.builder.complete_sentence()
        formatted = self._format_text(raw_sentence)

        if self.auto_punctuate and formatted:
            if not formatted.endswith(('.', '!', '?')):
                formatted += '.'

        self._history.append(formatted)
        return formatted

    def get_current_text(self) -> str:
        """Get the current formatted text."""
        return self._format_text(self.builder.get_current_sentence())

    def get_history(self) -> List[str]:
        """Get translation history."""
        return self._history.copy()

    def clear(self) -> None:
        """Clear all state."""
        self.builder.clear()
        self._history.clear()


class GlossToEnglish:
    """
    Converts ASL gloss notation to English sentences.

    ASL Gloss:
    ----------
    Gloss is a writing system for sign language that represents
    signs with English words in ASL grammar order.

    Example:
    - ASL gloss: "STORE GO-TO I"
    - English: "I am going to the store"

    This is a simplified rule-based translator. For production use,
    you would want a neural sequence-to-sequence model.
    """

    # Simple reordering patterns (very basic)
    # In practice, ASL-to-English requires much more sophisticated processing
    PATTERNS = [
        # (pattern, replacement)
        (r'(.+) I$', r'I \1'),  # Move "I" to start
        (r'(.+) YOU$', r'You \1'),  # Move "YOU" to start
    ]

    def __init__(self):
        """Initialize the gloss translator."""
        pass

    def translate(self, gloss: str) -> str:
        """
        Translate ASL gloss to English.

        Args:
            gloss: Space-separated ASL gloss words

        Returns:
            English translation

        Note: This is very basic. Real ASL translation requires:
        - Understanding ASL grammar rules
        - Handling non-manual markers (facial expressions)
        - Context-aware translation
        - Proper verb conjugation
        """
        text = gloss.upper()

        # Apply patterns
        for pattern, replacement in self.PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Basic cleanup
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:].lower()

        return text


# =============================================================================
# Testing
# =============================================================================

def test_translator():
    """Test the translation pipeline."""
    print("Testing SignToTextTranslator...")

    translator = SignToTextTranslator()

    # Simulate spelling "HELLO WORLD"
    test_signs = ['H', 'E', 'L', 'L', 'O', 'space',
                  'W', 'O', 'R', 'L', 'D']

    print("\nSimulating fingerspelling 'HELLO WORLD':")
    for i, sign in enumerate(test_signs):
        result = translator.process_sign(sign, timestamp=i * 0.5)
        print(f"  Sign '{sign}': current='{result['current_word']}', "
              f"sentence='{result['formatted_sentence']}'")
        if result['new_word']:
            print(f"    → Word completed: {result['new_word']}")

    # Complete sentence
    final = translator.complete_sentence()
    print(f"\nFinal sentence: '{final}'")

    # Test delete
    print("\n--- Testing Delete ---")
    translator.clear()
    for sign in ['H', 'E', 'L', 'del', 'L', 'P', 'space']:
        result = translator.process_sign(sign)
        print(f"  Sign '{sign}': word='{result['current_word']}'")

    print(f"  Result: '{translator.get_current_text()}'")

    # Test abbreviations
    print("\n--- Testing Abbreviations ---")
    translator.clear()
    for sign in ['H', 'I', 'space', 'U', 'space', 'R', 'space']:
        translator.process_sign(sign)
    print(f"  'HI U R' → '{translator.get_current_text()}'")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_translator()
