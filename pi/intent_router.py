"""
Project Qwen - voice command intent matching.

Checks a transcript against known command phrases before falling through to
the general-purpose LLM chat, so "Hey Qwen, start recording" triggers the
actual camera action instead of getting answered conversationally.

Uses the same fuzzy sliding-window matching approach as the wake-word
detection in assistant.py, generalized to phrases of any length.
"""

import difflib
import re

INTENT_PHRASES = {
    "start_recording": ["start recording", "record", "start filming"],
    "stop_recording": ["stop recording", "stop filming"],
    "take_photo": ["take a picture", "take a photo", "capture this"],
    "security_mode": ["security mode", "arm security mode", "activate security mode",
                       "enable security mode", "turn on security mode"],
    # On vs. off is decided by on_off_command() from the exact words, not by fuzzy
    # matching - "lights on" and "lights off" are ~95% similar, so a fuzzy match
    # could easily pick the wrong one. "flash" in the words means the head
    # flashlight; any other "light(s)" means the room lights (smart-home hub,
    # both light-switch servos at once).
    "flashlight": ["flashlight", "flash light", "flashlight on", "flashlight off",
                   "turn on the flashlight", "turn off the flashlight"],
    "room_lights": ["lights on", "lights off", "light on", "light off", "turn on the light",
                    "turn off the light", "turn on the lights", "turn off the lights",
                    "room lights on", "room lights off"],
    "look_up": ["look up", "raise your head", "head up"],
    "look_down": ["look down", "lower your head", "head down"],
    "turn_left": ["turn left", "rotate left"],
    "turn_right": ["turn right", "rotate right"],
    "move_forward": ["come forward", "come forward a bit", "move forward", "go forward",
                     "come here"],
    "move_backward": ["back up", "back up a little", "move back", "move backward"],
    "what_do_you_see": ["what do you see", "what are you looking at", "what can you see",
                        "what is in front of you", "describe what you see"],
    "anyone_here": ["is anyone here", "is anybody here", "is there anyone here",
                    "is someone here", "do you see anyone", "do you see anybody"],
    # The siren - see _ALARM_EXCLUDE: "set an alarm for 7" must never sound it.
    "alarm": ["alarm", "sound the alarm", "trigger the alarm", "start the alarm",
              "turn on the alarm"],
}

# Words that mean the user wants an alarm *clock* / is asking a question,
# not the security siren.
_ALARM_EXCLUDE = {"set", "clock", "timer", "wake", "remind", "reminder", "morning", "am", "pm",
                  "oclock", "tomorrow"}
_QUESTION_STARTS = {"is", "what", "when", "why", "how", "did", "does", "do", "was", "where"}

MOVEMENT_INTENTS = ("look_up", "look_down", "turn_left", "turn_right",
                    "move_forward", "move_backward")

INTENT_MATCH_THRESHOLD = 0.8
# Stricter for anything that moves the robot - a false match on a question
# ("how do I get back home") must never make it drive, least of all in reverse.
MOVEMENT_MATCH_THRESHOLD = 0.9


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _best_fuzzy_match(words: list[str], phrase: str) -> float:
    """Slides a window the length of `phrase` over `words` and returns the
    best match ratio found - lets a short phrase match even when it's
    embedded in a longer sentence ("please start recording now")."""
    phrase_words = phrase.split()
    n = len(phrase_words)
    if len(words) < n:
        return difflib.SequenceMatcher(None, " ".join(words), phrase).ratio()

    best = 0.0
    for i in range(len(words) - n + 1):
        window = " ".join(words[i:i + n])
        ratio = difflib.SequenceMatcher(None, window, phrase).ratio()
        best = max(best, ratio)
    return best


def match_intent(transcript: str) -> str | None:
    """Returns the matched intent name, or None if nothing matches well
    enough (caller should fall through to the LLM in that case)."""
    words = _normalize(transcript).split()
    if not words:
        return None

    best_intent = None
    best_ratio = 0.0
    for intent, phrases in INTENT_PHRASES.items():
        for phrase in phrases:
            ratio = _best_fuzzy_match(words, phrase)
            if ratio > best_ratio:
                best_ratio = ratio
                best_intent = intent

    threshold = MOVEMENT_MATCH_THRESHOLD if best_intent in MOVEMENT_INTENTS else INTENT_MATCH_THRESHOLD
    if best_ratio < threshold:
        return None
    if best_intent == "alarm" and (_ALARM_EXCLUDE & set(words) or words[0] in _QUESTION_STARTS
                                   or any(w.isdigit() for w in words)):
        return None
    if best_intent in ("flashlight", "room_lights"):
        best_intent = "flashlight" if any(w.startswith("flash") for w in words) else "room_lights"
    return best_intent


def on_off_command(transcript: str) -> str:
    """For a transcript already matched to 'flashlight' or 'room_lights', returns
    'on', 'off', or 'toggle' (neither word heard, e.g. just "flashlight")."""
    words = _normalize(transcript).split()
    if "off" in words:
        return "off"
    if "on" in words:
        return "on"
    return "toggle"


flashlight_command = on_off_command   # older name
