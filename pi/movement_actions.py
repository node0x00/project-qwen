"""
Project Qwen - movement voice commands ("follow simple verbal instructions").

"look up" / "look down"           -> scissor lift up/down by LOOK_STEP_DEG
"turn left" / "turn right"        -> in-place spin, ~90 deg by timing (TURN_90_S)
"come forward a bit"              -> short, slow forward move
"back up a little"                -> shorter, slower reverse (no rear sensors!)

All motion goes through ESP32Link; the ESP32 still enforces its own safety
rules (obstacle reflex, reverse speed/time caps, watchdog). This side only
asks, and speaks a refusal whenever the ESP32 reports BLOCKED.

Successful moves are silent - the motion itself is the feedback. Only
refusals and failures are spoken.

Standalone test (robot on blocks!):
    python movement_actions.py
"""

LOOK_STEP_DEG = 40           # how far one "look up/down" moves the lift
LIFT_MIN_DEG, LIFT_MAX_DEG = 0, 130
LIFT_MOVE_TIMEOUT_S = 10.0

# Turns are open-loop (no wheel encoders yet), so ~90 deg is a timing guess -
# calibrate TURN_90_S on the real floor.
TURN_SPEED = 80
TURN_90_S = 1.0

FORWARD_SPEED = 60
FORWARD_S = 1.0

# No rear sensors: slow and short. The ESP32 also caps reverse to 70 / 1.5 s.
REVERSE_SPEED = 50
REVERSE_S = 0.8

SENSOR_NAMES = {"bl": "bottom left", "bm": "bottom middle", "br": "bottom right",
                "head": "head"}


class MovementActions:
    def __init__(self, link, tts):
        self.link = link
        self.tts = tts

    def _reachable(self) -> bool:
        if self.link.connected:
            return True
        self.tts.speak("I can't reach my motor controller.")
        return False

    def _lift_position(self) -> int:
        # ESP32 boots with the lift at 0, so "unknown" means 0.
        pos = self.link.state.lift_pos
        return 0 if pos is None else pos

    def _move_lift(self, target: int, limit_text: str):
        if not self._reachable():
            return
        if self.link.lift_allowed is not None and not self.link.lift_allowed(target):
            self.tts.speak("I can't move my head while I'm charging.")
            return
        current = self._lift_position()
        if target == current:
            self.tts.speak(limit_text)
            return
        self.link.lift(target)
        if not self.link.wait_lift(LIFT_MOVE_TIMEOUT_S):
            self.tts.speak("My lift didn't finish moving.")

    def look_up(self):
        target = min(LIFT_MAX_DEG, self._lift_position() + LOOK_STEP_DEG)
        self._move_lift(target, "I'm already looking as high as I can.")

    def look_down(self):
        target = max(LIFT_MIN_DEG, self._lift_position() - LOOK_STEP_DEG)
        self._move_lift(target, "I'm already as low as I can go.")

    def _drive(self, left: int, right: int, seconds: float):
        blocked = self.link.drive_for(left, right, seconds)
        if blocked == "reverse":
            self.tts.speak("I stopped backing up, I can't see behind me.")
        elif blocked:
            self.tts.speak(f"I can't go that way, my {SENSOR_NAMES.get(blocked, blocked)} "
                           f"sensor sees something.")

    def turn_left(self):
        if self._reachable():
            self._drive(-TURN_SPEED, TURN_SPEED, TURN_90_S)

    def turn_right(self):
        if self._reachable():
            self._drive(TURN_SPEED, -TURN_SPEED, TURN_90_S)

    def forward(self):
        if not self._reachable():
            return
        if self.link.obstacle_ahead():
            self.tts.speak("I can't go forward, there's something in front of me.")
            return
        self._drive(FORWARD_SPEED, FORWARD_SPEED, FORWARD_S)

    def back_up(self):
        if self._reachable():
            self._drive(-REVERSE_SPEED, -REVERSE_SPEED, REVERSE_S)


if __name__ == "__main__":
    import time
    from esp32_link import ESP32Link

    class _PrintTTS:
        def speak(self, text: str):
            print(f"Qwen: {text}")

    tts = _PrintTTS()
    link = ESP32Link(speak=tts.speak)
    link.start()
    time.sleep(0.5)
    actions = MovementActions(link, tts)
    commands = {"u": actions.look_up, "d": actions.look_down, "l": actions.turn_left,
                "r": actions.turn_right, "f": actions.forward, "b": actions.back_up}
    print("ROBOT ON BLOCKS. u = look up, d = look down, l = turn left, r = turn right, "
          "f = forward, b = back up, q = quit")
    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "q":
                break
            if cmd in commands:
                commands[cmd]()
                print(f"lift={link.state.lift_pos} ir={link.state.ir}")
            else:
                print("unknown command")
    finally:
        link.close()
