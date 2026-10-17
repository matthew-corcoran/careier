# Careier

**Safer lifting for carers today; training data for care robots tomorrow.**

Third-person camera coach for a care-lift demo. It scores the **carer** with a REBA-style number, and treats a **person** differently from a **box**. The patient neck uses stable bands: 0–35° aligned, 35–55° caution, over 55° head drop.

Not a clinical assessment. The camera is the fast trial. The product story is a sensor suit + first-person video.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python coach.py --iphone
```

```bash
python coach.py --list-cameras
python coach.py --camera 1
```

Unlock the iPhone, keep it nearby, same Apple ID.

## Exact setup

1. Put the iPhone in **landscape**, on a chair/tripod at waist height, 2.5–3 m away.
2. Use a **45° side/front view**. Do not use portrait, selfie, or a dead-side silhouette.
3. Keep the whole carer visible from head to feet.
4. In HUMAN mode, start with the carer on the **left** and patient on the **right**. YOLO detects both people in one full-frame pass and follows their box positions.
5. Keep the patient's head and both shoulders visible. Do not let the carer's torso cover the patient's face.
6. If your positions are reversed, press `S`.

The app uses YOLO26s-pose plus ByteTrack for native multi-person detection and persistent identities. Keypoints and neck/trunk angles are smoothed across frames.

## Demo (30 seconds)

### Take 1 — box

- Press `1`.
- Only the carer needs to be visible; lift a bag.
- Screen says `BOX LIFT — NO NECK TO PROTECT`.

### Take 2 — human, bad

- Press `2`.
- Carer left, patient right.
- Patient sits/reclines while keeping head and shoulders visible.
- Lift the torso while letting the head visibly lag backward.
- Purple torso/head lines turn red above 55°; screen shows the measured angle and `HUMAN FAIL — HEAD DROP`.

### Take 3 — human, good

- Stay in HUMAN mode.
- Repeat while supporting the head so it remains aligned with the torso.
- Purple lines turn green; screen shows `HUMAN PASS — HEAD UP`.

If HUMAN mode sees only one pose, it says `PATIENT NOT VISIBLE`; it never reclassifies the take as a box.

Yellow skeleton = carer (REBA). Purple skeleton = patient (neck angle).

The carer torso line shows **trunk lean**, not spinal curvature. A camera cannot measure actual spinal curvature reliably; the future sensor suit handles that.

Keys:

- `N` next camera (does **not** quit)
- `S` swap left/right carer and patient zones
- `F` flip image
- `W` toggle fullscreen
- `Esc` return to the normal 1280×720 window
- `1` BOX mode
- `2` HUMAN mode
- `R` reset peak
- `Q` quit

Recorded clip:

```bash
python coach.py --video floor-transfer.mp4
python coach.py --video floor-transfer.mp4 --export exports/floor-transfer-careier.mp4
```

The person lower in the frame is the patient. `SPACE` pauses. `,` and `.` jump one second. `0` restarts. `S` swaps carer and patient.

Backup clip without a webcam:

```bash
python coach.py --demo
python coach.py --demo --record demo.mp4
```

```bash
python reba.py
```

## Send angles to the Careier hub

`coach.py` sends the carer's trunk angle, REBA score and the patient's neck angle over UDP to the laptop hub (default `127.0.0.1:5005`), so the capture screen shows the camera next to the wearable sensor.

```bash
python coach.py --iphone                       # sends to the hub on this laptop
python coach.py --iphone --hub 10.217.1.155:5005  # hub on another laptop
python coach.py --iphone --no-hub              # don't send
```

If no hub is running, packets are dropped and the coach works as before.

### Run the hub and capture screen

The hub receives the wearable sensor (UDP port 5005) and the camera coach, logs every message to `hub/data/`, and serves a live capture screen. Needs Node 18+, no npm install.

```bash
node hub/hub.mjs
```

Open http://localhost:8090. It shows the coach's annotated camera view, the carer's REBA score (with trunk, neck, legs and arm parts), the patient's neck status, the vest sensor's tilt, both trunk angles on one chart, beeps and a score per lift.

The sensor sends to this laptop's IP (`ipconfig getifaddr en0`) on port 5005. To test without hardware:

```bash
node hub/simulate.mjs
```

### Website with a live posture demo

The hub also serves the project website at http://localhost:8090/site/. When opened from the hub, "Try a posture" follows the live sensor: the figure bends as the wearer bends. Click "Live sensor" to pause it and use the slider.

## Pitch (20 seconds)

A box going up is success. A person going up with a 60-degree neck bend is failure. We measure both the carer's injury risk and the patient's neck alignment. This trial uses a third-person camera; the product uses a sensor suit and first-person video.
