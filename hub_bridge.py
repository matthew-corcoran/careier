"""Send the camera coach's angles to the Careier laptop hub over UDP.

The hub (hub/hub.mjs) already receives the wearable's packets on port 5005, so
camera and sensor land on one timeline in the capture screen. Fire-and-forget:
if no hub is running, packets are simply dropped and the coach keeps working.
"""

from __future__ import annotations

import json
import socket
import time


class HubBridge:
    def __init__(self, target: str | None = "127.0.0.1:5005", rate_hz: float = 15.0) -> None:
        self.addr = None
        self.sock = None
        self.seq = 0
        self.min_dt = 1.0 / rate_hz
        self.last = 0.0
        if target:
            host, _, port = target.rpartition(":")
            self.addr = (host or "127.0.0.1", int(port or 5005))
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"Sending camera angles to hub at {self.addr[0]}:{self.addr[1]}")

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
    ) -> None:
        if self.sock is None:
            return
        now = time.time()
        if now - self.last < self.min_dt:
            return
        self.last = now
        msg = {
            "type": "camera",
            "source": "camera",
            "seq": self.seq,
            "t_ms": int(now * 1000),
            "mode": mode,
            "people": people,
            "carer_trunk_deg": None if carer_trunk_deg is None else round(float(carer_trunk_deg), 1),
            "reba": reba_score,
            "reba_band": reba_band,
            "patient_neck_deg": None if patient_neck_deg is None else round(float(patient_neck_deg), 1),
            "patient_neck_status": patient_neck_status,
            "patient_head_ok": patient_head_ok,
        }
        self.seq += 1
        try:
            self.sock.sendto(json.dumps(msg).encode("utf-8"), self.addr)
        except OSError:
            pass
