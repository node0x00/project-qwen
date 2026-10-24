"""
Project Qwen - ORB-SLAM3 live view + remote control (port 8080, same ?key= as
the normal vision stream).

Runs mono_live (ORB-SLAM3 + dock AprilTag) on the camera and serves its
annotated frames, plus a control panel: hold W/A/S/D to drive, Space or the
red button to stop, "Explore" to let the robot search for the dock tag. Any
key or STOP takes over from the robot immediately.

    venv/bin/python slam_view.py              # ORB-SLAM + tag
    venv/bin/python slam_view.py --tag-only   # tag only, SLAM off (docking)
"""

import http.server
import threading
import json
import sys
import os
import secrets
import signal
import subprocess
import threading
import time
import urllib.parse

from robot_control import RobotController
from stream_server import _stream_key

SLAM_DIR = os.path.expanduser("~/slam")
VIEW = "/dev/shm/slam_view.jpg"
PORT = 8080
STREAM_FPS = 10
TAG_ONLY = "--tag-only" in sys.argv      # docking: skip ORB-SLAM (much less CPU / heat)

PAGE = """<!doctype html><html><head><title>Qwen control</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#101214;--panel:#1a1d21;--line:#2c3137;--text:#d7dce2;--dim:#8a939d;
--accent:#5fb3ff;--ok:#56d364;--warn:#f0b429;--bad:#ff5a52}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,-apple-system,sans-serif;
padding:12px 16px;display:flex;flex-direction:column;align-items:center;gap:12px}
.wrap{width:100%;max-width:980px;display:grid;grid-template-columns:minmax(0,640px) minmax(0,1fr);gap:12px}
@media (max-width:860px){.wrap{grid-template-columns:1fr}}
img{width:100%;height:auto;border-radius:6px;background:#000;display:block}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;
display:flex;flex-direction:column;gap:12px}
h2{margin:0;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);font-weight:600}
#mode{font-size:20px;font-weight:700}
#activity{color:var(--dim);min-height:2.6em}
button{font:inherit;border-radius:6px;border:1px solid var(--line);background:#23272c;color:var(--text);
padding:10px 12px;cursor:pointer}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
#stop{background:var(--bad);border-color:var(--bad);color:#fff;font-weight:800;font-size:18px;padding:14px}
#explore{border-color:var(--accent);color:var(--accent)}
.pad{display:grid;grid-template-columns:repeat(3,56px);grid-template-rows:repeat(2,56px);gap:6px;justify-content:center}
.pad button{padding:0;font-weight:700;font-size:16px;touch-action:none;user-select:none}
.pad button.on{background:var(--accent);color:#000;border-color:var(--accent)}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.chip{padding:3px 8px;border-radius:999px;border:1px solid var(--line);font-size:12px;font-variant-numeric:tabular-nums}
.chip.hit{background:var(--bad);border-color:var(--bad);color:#fff}
.chip.good{border-color:var(--ok);color:var(--ok)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 10px;font-variant-numeric:tabular-nums}
.kv span:nth-child(odd){color:var(--dim)}
input[type=range]{flex:1}
.hint{color:var(--dim);font-size:12px}
</style></head><body>
<div class="wrap">
  <img src="/stream.mjpg?key=__KEY__" alt="Robot camera with ORB-SLAM3 features">
  <div class="panel">
    <div><h2>Mode</h2><div id="mode">-</div><div id="activity"></div></div>
    <button id="stop">STOP (Space)</button>
    <button id="dock" style="font-weight:700">DOCK (find, approach, turn, back on)</button>
    <div><h2>Video lines</h2>
      <div class="row"><button id="say-intro">&ldquo;And I'm Qwen&rdquo;</button>
      <button id="say-outro">&ldquo;Thanks for watching&rdquo;</button>
      <button id="demo-full">Demo: battery full &rarr; drive off</button></div></div>
    <div><h2>Security</h2>
      <div class="row"><button id="arm">Arm security mode</button>
      <button id="alarm">Alarm: off</button></div>
      <div class="hint" id="sechint"></div></div>
    <div><h2>Volume</h2>
      <div class="row"><input id="vol" type="range" min="0" max="100" step="1" value="50">
        <span id="volv" class="chip">50%</span></div></div>
    <div><h2>Lift</h2>
      <div class="row"><input id="lift" type="range" min="0" max="130" step="1" value="0">
        <span id="liftv" class="chip">0&deg;</span></div>
      <div class="hint" id="lifthint">Drag to move the scissor lift (0&ndash;130&deg;).</div></div>
    <div><h2>Dock step by step</h2>
      <div class="row"><button id="s1">1 Find + line up</button><button id="s2">2 Forward to turn spot</button>
      <button id="s3">3 Turn 180</button><button id="s4">4 Back onto dock</button></div></div>
    <div class="row"><button id="find">Find dock (360 scan)</button>
      <button id="explore">Explore: find dock tag</button>
      <button id="centre">Centre on tag</button>
      <button id="approach">Approach dock</button></div>
    <div class="row"><button id="teach">Teach dock view (on dock)</button>
      <button id="back">Back onto dock</button>
      <button id="push">Final push (3-4 cm out)</button>
      <button id="prepush">Save pre-push spot</button></div>
    <div>
      <h2>Drive (hold keys)</h2>
      <div class="pad">
        <span></span><button data-k="w">W</button><span></span>
        <button data-k="a">A</button><button data-k="s">S</button><button data-k="d">D</button>
      </div>
    </div>
    <div class="row"><h2>Speed</h2><input id="speed" type="range" min="1" max="30" value="1">
      <span id="speedv" class="chip">1</span></div>
    <div><h2>Front IR</h2><div class="row" id="ir"></div></div>
    <div><h2>Dock tag</h2><div class="kv" id="tag"><span>status</span><span>not seen</span></div></div>
    <div class="hint">Releasing the keys stops the robot within 0.35 s. Any key or STOP takes over from
      Explore. S backs up blind (no rear sensors), at most 1.5 s at a time.</div>
  </div>
</div>
<script>
const KEY = "__KEY__";
const held = new Set();
const post = (q) => fetch(`/cmd?key=${KEY}&${q}`, {method: "POST"}).catch(() => {});
const sendKeys = () => post(`keys=${[...held].join("")}`);
function setHeld(k, on) {
  if (on) held.add(k); else held.delete(k);
  document.querySelectorAll(`.pad button[data-k="${k}"]`).forEach(b => b.classList.toggle("on", on));
  sendKeys();
}
addEventListener("keydown", e => {
  const k = e.key.toLowerCase();
  if (e.code === "Space") { e.preventDefault(); held.clear(); post("stop=1"); return; }
  if ("wasd".includes(k) && k.length === 1) { e.preventDefault(); if (!held.has(k)) setHeld(k, true); }
});
addEventListener("keyup", e => { const k = e.key.toLowerCase(); if (held.has(k)) setHeld(k, false); });
addEventListener("blur", () => { [...held].forEach(k => setHeld(k, false)); });
document.addEventListener("visibilitychange", () => { if (document.hidden) [...held].forEach(k => setHeld(k, false)); });
// Hidden tab: close the video stream, so a forgotten tab does not keep the robot awake
// (idle mode stays awake while anyone is watching).
const cam = document.querySelector("img"), camSrc = cam.src;
document.addEventListener("visibilitychange", () => {
  if (document.hidden) cam.src = "data:,";
  else cam.src = camSrc + "&t=" + Date.now();
});
setInterval(() => { if (held.size) sendKeys(); }, 100);   // keep-alive for the dead-man timer
document.querySelectorAll(".pad button").forEach(b => {
  const k = b.dataset.k;
  b.addEventListener("pointerdown", e => { e.preventDefault(); b.setPointerCapture(e.pointerId); setHeld(k, true); });
  ["pointerup", "pointercancel", "lostpointercapture"].forEach(ev => b.addEventListener(ev, () => setHeld(k, false)));
});
document.getElementById("stop").onclick = () => { held.clear(); post("stop=1"); };
document.getElementById("explore").onclick = () => post("explore=1");
document.getElementById("find").onclick = () => post("find=1");
document.getElementById("s1").onclick = () => post("step=1");
document.getElementById("s2").onclick = () => post("step=2");
document.getElementById("s3").onclick = () => post("step=3");
document.getElementById("s4").onclick = () => post("back=1");
document.getElementById("dock").onclick = () => post("dock=1");
document.getElementById("centre").onclick = () => post("centre=1");
document.getElementById("approach").onclick = () => post("approach=1");
document.getElementById("teach").onclick = () => post("teach=1");
document.getElementById("back").onclick = () => post("back=1");
document.getElementById("push").onclick = () => post("push=1");
document.getElementById("prepush").onclick = () => post("prepush=1");
document.getElementById("arm").onclick = () => post("arm=1");
document.getElementById("alarm").onclick = () => post("alarm=toggle");
const vl = document.getElementById("vol"), vlv = document.getElementById("volv");
let volTimer = null;
vl.oninput = () => {
  vlv.textContent = `${vl.value}%`;
  if (!volTimer) volTimer = setTimeout(() => { volTimer = null; post(`volume=${vl.value}`); }, 200);
};
vl.onchange = () => { clearTimeout(volTimer); volTimer = null; post(`volume=${vl.value}`); };
document.getElementById("say-intro").onclick = () => post("say=intro");
document.getElementById("demo-full").onclick = () => post("demo_full=1");
document.getElementById("say-outro").onclick = () => post("say=outro");
const lf = document.getElementById("lift"), lfv = document.getElementById("liftv");
let liftTimer = null, liftSent = null;
const sendLift = () => { if (lf.value !== liftSent) { liftSent = lf.value; post(`lift=${lf.value}`); } };
lf.oninput = () => {
  lfv.textContent = `${lf.value}\u00b0`;
  if (!liftTimer) liftTimer = setTimeout(() => { liftTimer = null; sendLift(); }, 250);
};
lf.onchange = () => { clearTimeout(liftTimer); liftTimer = null; sendLift(); };
const sp = document.getElementById("speed"), spv = document.getElementById("speedv");
sp.oninput = () => { spv.textContent = sp.value; post(`speed=${sp.value}`); };
const names = {bl: "left", bm: "middle", br: "right"};
async function poll() {
  try {
    const s = await (await fetch(`/status?key=${KEY}`)).json();
    const m = document.getElementById("mode");
    m.textContent = {idle: "Stopped", manual: "You're driving", auto: "Robot exploring"}[s.mode] || s.mode;
    m.style.color = s.mode === "auto" ? "var(--accent)" : s.mode === "manual" ? "var(--ok)" : "var(--text)";
    document.getElementById("activity").textContent =
      (s.esp32 ? "" : "ESP32 not connected! ") + (s.blocked ? `Blocked: ${s.blocked}. ` : "") + s.activity;
    if (document.activeElement !== sp) { sp.value = s.speed; spv.textContent = s.speed; }
    if (document.activeElement !== lf && s.lift !== null && s.lift !== undefined && liftTimer === null) {
      lf.value = s.lift; lfv.textContent = `${s.lift}\u00b0`;
    }
    lf.disabled = !!s.lift_locked;
    if (s.security !== undefined) {
      const armed = s.security !== "off";
      const ab = document.getElementById("arm");
      ab.textContent = armed ? `Security: ${s.security.replace("_", " ")}` : "Arm security mode";
      ab.disabled = armed;
      ab.style.borderColor = armed ? "var(--warn)" : "";
      document.getElementById("sechint").textContent = armed
        ? "Disarm with your face or the keypad code." : "";
      const al = document.getElementById("alarm");
      al.textContent = s.alarm ? "Alarm: ON (press to stop)" : "Alarm: off";
      al.style.background = s.alarm ? "var(--bad)" : "";
      al.style.color = s.alarm ? "#fff" : "";
    }
    if (s.volume !== undefined && s.volume !== null && document.activeElement !== vl && volTimer === null) {
      vl.value = s.volume; vlv.textContent = `${s.volume}%`;
    }
    document.getElementById("lifthint").textContent = s.lift_locked
      ? `Lift locked: ${s.lift_locked}.` : "Drag to move the scissor lift (0\u2013130\u00b0).";
    document.getElementById("ir").innerHTML = Object.entries(s.ir).map(([k, v]) =>
      `<span class="chip ${v ? "hit" : ""}">${names[k] || k}: ${v ? "OBSTACLE" : "clear"}</span>`).join("");
    const t = s.tag;
    document.getElementById("tag").innerHTML = t
      ? `<span>distance</span><span>${t.dist_cm.toFixed(1)} cm</span><span>sideways</span><span>${t.x_cm.toFixed(1)} cm</span>` +
        `<span>bearing</span><span>${t.bearing.toFixed(1)}&deg;</span><span>tag yaw</span><span>${t.yaw.toFixed(1)}&deg;</span>`
      : `<span>status</span><span>not seen</span>`;
  } catch (e) {
    document.getElementById("activity").textContent = "Lost connection to the robot";
  }
  setTimeout(poll, 300);
}
poll();
</script></body></html>"""


VIDEO_LINES = {"intro": "And I'm Qwen.", "outro": "Thanks for watching!"}


def handle_cmd(robot, q: dict):
    """One /cmd request (parsed query) -> robot action. Shared with assistant.py's web page."""
    if "say" in q:                       # fixed lines for filming (speech runs in the background)
        text = VIDEO_LINES.get(q["say"][0])
        if text and getattr(robot, "say", None):
            threading.Thread(target=robot.say, args=(text,), daemon=True).start()
        return
    hooks = getattr(robot, "page_hooks", None) or {}
    for name in ("arm", "alarm", "volume", "demo_full"):
        if name in q and name in hooks:
            hooks[name](q[name][0])
            return
    if "lift" in q:                      # lift=<deg> 0-130, the same command the potentiometer overrides
        deg = max(0, min(130, int(float(q["lift"][0]))))
        if robot.link.lift(deg) is False:
            robot._caption(f"lift: can't move now ({robot.lift_lock_reason() or 'locked'})")
        return
    if "stop" in q:
        robot.stop("STOP pressed")
    elif "explore" in q:
        robot.start_explore()
    elif "find" in q:
        robot.start_find_dock()
    elif "step" in q:
        {"1": robot.start_step_lineup, "2": robot.start_step_forward,
         "3": robot.start_step_turn, "4": robot.start_dock_back}.get(q["step"][0], lambda: None)()
    elif "move" in q:                    # move=<mm> straight on odometry (+ forward)
        robot.start_move(max(-400.0, min(800.0, float(q["move"][0]))))
    elif "turn" in q:                    # turn=<deg> in place on odometry, + = left
        robot.start_turn(max(-360.0, min(360.0, float(q["turn"][0]))))
    elif "spincal" in q:
        robot.start_spin_calibration()
    elif "dock" in q:                     # the DOCK button: steps 1-4 back to back
        robot.start_dock_sequence()
    elif "centre" in q:
        robot.start_centre()
    elif "approach" in q:
        robot.start_approach()
    elif "teach" in q:
        robot.start_teach(force=q["teach"][0] == "force")   # teach=force: you confirm it's docked
    elif "back" in q:
        robot.start_dock_back()
    elif "prepush" in q:
        robot.save_prepush()
    elif "push" in q:
        robot.start_final_push()
    elif "speed" in q:
        robot.set_speed(int(q["speed"][0]))
    elif "test" in q:                    # test=left,right,seconds (bench checks)
        l, r, sec = q["test"][0].split(",")
        l, r = int(l), int(r)
        cap = 20.0 if l + r == 0 else 3.0      # spins in place may run longer
        robot.test_motors(l, r, min(cap, float(sec)))
    elif "sweep" in q:                   # sweep=left|right,1|-1
        wheel, d = q["sweep"][0].split(",")
        robot.sweep(wheel, int(d))
    elif "keys" in q:
        robot.manual(q["keys"][0])


def main():
    key = _stream_key()
    robot = RobotController()
    robot.start()
    open(VIEW + ".txt", "w").close()
    robot.stop("ready - stopped")
    stop_event = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _auth(self, url) -> bool:
            k = urllib.parse.parse_qs(url.query).get("key", [""])[0]
            if not secrets.compare_digest(k, key):
                self.send_error(403, "missing or wrong ?key=")
                return False
            return True

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urllib.parse.urlparse(self.path)
            if not self._auth(url):
                return
            if url.path == "/":
                self._send(PAGE.replace("__KEY__", key).encode(), "text/html")
            elif url.path == "/status":
                self._send(json.dumps(robot.status()).encode(), "application/json")
            elif url.path == "/snapshot.jpg":
                try:
                    self._send(open(VIEW, "rb").read(), "image/jpeg")
                except OSError:
                    self.send_error(503, "no frame yet")
            elif url.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while not stop_event.is_set():
                        t = time.time()
                        try:
                            jpg = open(VIEW, "rb").read()
                        except OSError:
                            jpg = None
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(max(0.0, 1.0 / STREAM_FPS - (time.time() - t)))
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_error(404)

        def do_POST(self):
            url = urllib.parse.urlparse(self.path)
            if not self._auth(url):
                return
            q = urllib.parse.parse_qs(url.query)
            if url.path != "/cmd":
                self.send_error(404)
                return
            handle_cmd(robot, q)
            self._send(b"ok", "text/plain")

    def info_loop():
        """Battery % and Pi temperature for the bottom-right corner of the live view."""
        while not stop_event.is_set():
            parts = []
            b = robot.battery()
            if b:
                parts.append(f"batt {b[1]:.0f}%{' charging' if b[2] else ''} ({b[0]:.1f} V)")
            try:
                parts.append(f"Pi {int(open('/sys/class/thermal/thermal_zone0/temp').read()) / 1000:.0f} C")
            except OSError:
                pass
            try:
                with open(VIEW + ".info", "w") as f:
                    f.write("  ".join(parts))
            except OSError:
                pass
            stop_event.wait(5)
    threading.Thread(target=info_loop, daemon=True).start()

    server = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[control] http://raspberrypi.local:{PORT}/?key={key}", flush=True)

    slam = subprocess.Popen(
        [f"{SLAM_DIR}/qwen_slam/build/mono_live", f"{SLAM_DIR}/ORB_SLAM3/Vocabulary/ORBvoc.txt",
         f"{SLAM_DIR}/qwen_slam/c920e.yaml", "100000", VIEW] + (["tagonly"] if TAG_ONLY else []),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    signal.signal(signal.SIGTERM, lambda *_: slam.send_signal(signal.SIGINT))
    try:
        for line in slam.stdout:
            if line.startswith(("t ", "loaded", "New Map", "Shutdown")):
                print(time.strftime("%H:%M:%S"), line.rstrip(), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()
        stop_event.set()
        slam.send_signal(signal.SIGINT)
        try:
            slam.wait(timeout=20)
        except subprocess.TimeoutExpired:
            slam.kill()
        server.shutdown()


if __name__ == "__main__":
    main()
