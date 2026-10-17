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

CHUNK = 7000  # base64 characters per packet, under macOS's default 9,216-byte UDP limit


class HubBridge:
    def __init__(self, target: str | None = "127.0.0.1:5005", rate_hz: float = 15.0, frame_hz: float = 6.0) -> None:
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

    def send(
        self,
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
    ) -> None:
        if self.sock is None:
            return
        now = time.time()
        if now - self.last < self.min_dt:
            return
        self.last = now
        self._send({
            "type": "camera",
            "source": "camera",
            "seq": self.seq,
            "t_ms": int(now * 1000),
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
        })
        self.seq += 1

    def send_frame(self, frame) -> None:
        """Send a small JPEG of the annotated camera view, a few times a second."""
        if self.sock is None:
            return
        now = time.time()
        if now - self.last_frame < self.frame_dt:
            return
        self.last_frame = now
        import cv2  # coach.py already depends on OpenCV

        h, w = frame.shape[:2]
        scale = 480 / max(w, 1)
        small = cv2.resize(frame, (480, max(1, int(h * scale))), interpolation=cv2.INTER_AREA) if w > 480 else frame
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        if not ok:
            return
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        # macOS caps UDP datagrams at ~9 KB by default, so send the frame in chunks the hub reassembles.
        parts = [b64[i : i + CHUNK] for i in range(0, len(b64), CHUNK)]
        frame_id = self.frame_id = getattr(self, "frame_id", 0) + 1
        for n, part in enumerate(parts):
            self._send({"type": "camera_chunk", "source": "camera", "id": frame_id, "n": n, "of": len(parts), "data": part})
