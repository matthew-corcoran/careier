// Careier laptop hub: receives the wearable's UDP JSON packets on :5005 and streams them to the
// capture screen over Server-Sent Events. Zero dependencies. Run: node hub/hub.mjs
// Every message is also appended to hub/data/session-<time>.jsonl as the raw dataset.
import dgram from "node:dgram";
import { createServer } from "node:http";
import { readFile, mkdir } from "node:fs/promises";
import { createWriteStream } from "node:fs";
import { networkInterfaces } from "node:os";
import { fileURLToPath } from "node:url";
import { join } from "node:path";

const UDP_PORT = Number(process.env.UDP_PORT || 5005);
const HTTP_PORT = Number(process.env.HTTP_PORT || 8090);
const dir = fileURLToPath(new URL(".", import.meta.url));

await mkdir(join(dir, "data"), { recursive: true });
const logPath = join(dir, "data", `session-${new Date().toISOString().replace(/[:.]/g, "-")}.jsonl`);
const log = createWriteStream(logPath, { flags: "a" });

const clients = new Set();
const lastSeqBySender = new Map();
const stats = { packets: 0, bad: 0, lastSeq: null, dropped: 0, lastFrom: null, lastAt: 0, cameraAt: 0 };

function broadcast(obj) {
  const line = `data: ${JSON.stringify(obj)}\n\n`;
  for (const res of clients) res.write(line);
}

const udp = dgram.createSocket("udp4");
udp.on("message", (buf, rinfo) => {
  const receivedAt = Date.now();
  let msg;
  try { msg = JSON.parse(buf.toString("utf8")); } catch { stats.bad++; return; }
  const rec = { ...msg, hub_ms: receivedAt };
  log.write(JSON.stringify(rec) + "\n");
  broadcast(rec);
  if (msg.source === "camera") { stats.cameraAt = receivedAt; return; } // camera has its own seq; don't mix it into sensor gaps
  stats.packets++;
  stats.lastFrom = `${rinfo.address}:${rinfo.port}`;
  stats.lastAt = receivedAt;
  if (typeof msg.seq === "number") {
    // Track sequence per sender so a second device (or the simulator) doesn't look like packet loss.
    const key = msg.device_id || rinfo.address;
    const last = lastSeqBySender.get(key);
    if (last != null && msg.seq > last + 1 && msg.seq - last < 10000) stats.dropped += msg.seq - last - 1;
    lastSeqBySender.set(key, msg.seq);
    stats.lastSeq = msg.seq;
  }
  if (msg.type === "status") stats.status = msg;
  if (msg.type && msg.type !== "frame" && msg.type !== "frames") console.log(`[${msg.type}]`, JSON.stringify(msg));
});
udp.on("error", (e) => { console.error("UDP error:", e.message); process.exit(1); });
udp.bind(UDP_PORT);

setInterval(() => broadcast({ type: "hub_stats", ...stats, clients: clients.size }), 1000);

createServer(async (req, res) => {
  if (req.url === "/events") {
    res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" });
    res.write(": connected\n\n");
    clients.add(res);
    req.on("close", () => clients.delete(res));
    return;
  }
  if (req.url === "/" || req.url === "/index.html") {
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    return res.end(await readFile(join(dir, "capture.html")));
  }
  res.writeHead(404).end();
}).listen(HTTP_PORT, () => {
  const ips = Object.values(networkInterfaces()).flat().filter((i) => i && i.family === "IPv4" && !i.internal).map((i) => i.address);
  console.log(`Careier hub`);
  console.log(`  UDP listening on port ${UDP_PORT}  (set HUB_IP on the sensor to one of: ${ips.join(", ") || "?"})`);
  console.log(`  Capture screen: http://localhost:${HTTP_PORT}`);
  console.log(`  Logging to ${logPath}`);
});
