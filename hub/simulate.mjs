// Fake sensor for testing the hub without hardware: sends a calibration, a bad lift and a good lift
// as UDP JSON packets at 50 Hz, shaped like the Axiometa firmware's output. Run: node hub/simulate.mjs
import dgram from "node:dgram";

const HOST = process.argv[2] || "127.0.0.1";
const PORT = 5005;
const sock = dgram.createSocket("udp4");
let seq = 0, t0 = Date.now();
const send = (o) => sock.send(JSON.stringify({ ...o, seq: seq++, t_ms: Date.now() - t0 }), PORT, HOST);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Fake camera coach (what coach.py's hub_bridge sends), roughly 15 Hz, a few degrees off the sensor.
let camSeq = 0, headDrop = false, i = 0;
const sendCam = (tilt) => sock.send(JSON.stringify({
  type: "camera", source: "camera", seq: camSeq++, t_ms: Date.now(), mode: "human", people: 2,
  carer_trunk_deg: +(tilt - 4 + Math.random() * 3).toFixed(1),
  reba: tilt > 60 ? 10 : tilt > 20 ? 6 : 3, reba_band: tilt > 60 ? "HIGH" : tilt > 20 ? "MEDIUM" : "LOW",
  patient_neck_deg: headDrop ? 64 : 14, patient_neck_status: headDrop ? "danger" : "safe", patient_head_ok: !headDrop,
}), PORT, HOST);

async function phase(ms, fn) {
  for (let t = 0; t < ms; t += 20) {
    const f = fn(t / ms);
    send({ type: "frame", ...f });
    if (i++ % 3 === 0) sendCam(f.trunk_flexion);
    await sleep(20);
  }
}
const frame = (tilt, twist = null) => ({ trunk_flexion: +tilt.toFixed(1), twist, lumbar_flexion: null, strain: null, battery_pct: null, calibrated: true });

async function lift(peak, twistPeak, label) {
  console.log("lift:", label);
  send({ type: "lift_start" });
  let beeped = false, twisted = false, start = Date.now();
  await phase(3500, (p) => {
    const tilt = 22 + (peak - 22) * Math.sin(Math.PI * p);
    const tw = twistPeak * Math.max(0, Math.sin(Math.PI * (p - 0.4) / 0.5));
    if (tilt > 60 && !beeped) { beeped = true; send({ type: "alert", kind: "round_back" }); }
    if (tw > 20 && !twisted) { twisted = true; send({ type: "alert", kind: "twist" }); }
    return frame(tilt, +tw.toFixed(1));
  });
  await phase(800, (p) => frame(14 - 10 * p, 0));
  send({ type: "lift_end", duration_ms: Date.now() - start, peak_trunk_flexion: peak, peak_twist: twistPeak, score: null });
}

send({ type: "calibrated" });
await phase(2000, () => frame(2 + Math.random() * 2, 0));
headDrop = true;
await lift(72, 28, "bad: stooped and twisting, patient head drops");
headDrop = false;
await phase(2500, () => frame(3 + Math.random() * 2, 0));
await lift(38, 6, "good: squat, feet turned");
await phase(1500, () => frame(3, 0));
sock.close();
console.log("done");
