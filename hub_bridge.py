"""Send the camera coach's scores and annotated view to the Careier laptop hub over UDP.

The hub (hub/hub.mjs) already receives the wearable's packets on port 5005, so
camera and sensor land on one timeline in the capture screen. Fire-and-forget:
if no hub is running, packets are simply dropped and the coach keeps working.
"""

from __future__ import annotations

import base64
import json
import socket
import time
from pathlib import Path

CHUNK = 7000  # base64 characters per packet, under macOS's default 9,216-byte UDP limit


def camera_message(
    *,
    mode: str,
    carer_trunk_deg: float | None,
    reba_score: int | None,
    reba_band: str | None,
    patient_neck_deg: float | None,
    patient_neck_status: str | None,
    patient_head_ok: bool | None,
    people: int,
    peak: int | None = None,
    parts: dict[str, int] | None = None,
    status: str | None = None,
    detail: str | None = None,
) -> dict:
    return {
        "type": "camera",
        "source": "camera",
        "mode": mode,
        "people": people,
        "carer_trunk_deg": None if carer_trunk_deg is None else round(float(carer_trunk_deg), 1),
        "reba": reba_score,
        "reba_band": reba_band,
        "reba_peak": peak,
        "reba_parts": parts,
        "patient_neck_deg": None if patient_neck_deg is None else round(float(patient_neck_deg), 1),
        "patient_neck_status": patient_neck_status,
        "patient_head_ok": patient_head_ok,
        "status": status,
        "detail": detail,
    }


def _shrink(frame, long_side: int):
    import cv2  # coach.py already depends on OpenCV

    h, w = frame.shape[:2]
    scale = long_side / max(w, h, 1)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


class ClipRecorder:
    """Record what the hub would receive while scoring a clip.

    Writes <name>.feed.mp4 (the annotated view only) and <name>.json (the
    camera scores against video time), so the dashboard's Demo video switch
    can replay a recording with its real scores and no laptop pipeline.
    """

    def __init__(self, export_path: str | Path, fps: float) -> None:
        base = Path(export_path)
        self.video_path = base.with_name(base.stem + ".feed.mp4")
        self.json_path = base.with_suffix(".json")
        self.fps = fps
        self.t = 0.0
        self.writer = None
        self.timeline: list[dict] = []

    def send_frame(self, frame) -> None:
        import cv2

        frame = _shrink(frame, 1280)
        if self.writer is None:
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            size = (frame.shape[1], frame.shape[0])
            # avc1 is the codec browsers can play; mp4v is the fallback if this OpenCV lacks it.
            for codec in ("avc1", "mp4v"):
                self.writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*codec), self.fps, size)
                if self.writer.isOpened():
                    break
                self.writer.release()
                self.writer = None
            if self.writer is None:
                raise SystemExit(f"Could not open video writer for {self.video_path}")
        self.writer.write(frame)

    def send(self, **fields) -> None:
        self.timeline.append({"t": round(self.t, 3), **camera_message(**fields)})

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        duration = self.timeline[-1]["t"] if self.timeline else 0.0
        self.json_path.write_text(json.dumps({
            "video": self.video_path.name,
            "fps": self.fps,
            "duration": duration,
            "timeline": self.timeline,
        }))
        print(f"Wrote {self.video_path} and {self.json_path} for the dashboard's demo switch")


class HubBridge:
    def __init__(self, target: str | None = "127.0.0.1:5005", rate_hz: float = 15.0, frame_hz: float = 10.0) -> None:
        self.addr = None
        self.sock = None
        self.seq = 0
        self.min_dt = 1.0 / rate_hz
        self.frame_dt = 1.0 / frame_hz
        self.last = 0.0
        self.last_frame = 0.0
        if target:
            host, _, port = target.rpartition(":")
            self.addr = (host or "127.0.0.1", int(port or 5005))
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"Sending camera angles to hub at {self.addr[0]}:{self.addr[1]}")

    def _send(self, msg: dict) -> None:
        try:
            self.sock.sendto(json.dumps(msg).encode("utf-8"), self.addr)
        except OSError:
            pass

    def send(self, **fields) -> None:
        if self.sock is None:
            return
        now = time.time()
        if now - self.last < self.min_dt:
            return
        self.last = now
        self._send({**camera_message(**fields), "seq": self.seq, "t_ms": int(now * 1000)})
        self.seq += 1

    def send_frame(self, frame) -> None:
        """Send a small JPEG of the annotated camera view, ten times a second."""
        if self.sock is None:
            return
        now = time.time()
        if now - self.last_frame < self.frame_dt:
            return
        self.last_frame = now
        import cv2  # coach.py already depends on OpenCV

        small = _shrink(frame, 640)
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ok:
            return
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        # macOS caps UDP datagrams at ~9 KB by default, so send the frame in chunks the hub reassembles.
        parts = [b64[i : i + CHUNK] for i in range(0, len(b64), CHUNK)]
        frame_id = self.frame_id = getattr(self, "frame_id", 0) + 1
        for n, part in enumerate(parts):
            self._send({"type": "camera_chunk", "source": "camera", "id": frame_id, "n": n, "of": len(parts), "data": part})
