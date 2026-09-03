import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const require = createRequire(import.meta.url);
const WebSocket = require("../apps/web/node_modules/next/dist/compiled/ws");

const edgePath = process.env.EDGE_PATH ??
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const outputDir = resolve("docs/project-overview");
const profileDir = resolve("data/interim/edge-project-overview-profile");
const port = 9223;

mkdirSync(outputDir, { recursive: true });

const edge = spawn(edgePath, [
  "--headless=new",
  "--disable-gpu",
  "--hide-scrollbars",
  `--remote-debugging-port=${port}`,
  "--remote-allow-origins=*",
  `--user-data-dir=${profileDir}`,
  "--window-size=1890,900",
  "about:blank",
], { stdio: "ignore", windowsHide: true });

const sleep = (ms) => new Promise((resolvePromise) => setTimeout(resolvePromise, ms));

async function devtoolsTarget() {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/json/list`);
      const targets = await response.json();
      const page = targets.find((target) => target.type === "page");
      if (page) return page;
    } catch {
      // Edge may still be starting.
    }
    await sleep(250);
  }
  throw new Error("Could not connect to the Edge DevTools endpoint.");
}

const target = await devtoolsTarget();
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolvePromise, reject) => {
  socket.once("open", resolvePromise);
  socket.once("error", reject);
});

let nextId = 1;
const pending = new Map();
socket.on("message", (raw) => {
  const message = JSON.parse(raw.toString());
  if (!message.id || !pending.has(message.id)) return;
  const { resolve: resolvePromise, reject } = pending.get(message.id);
  pending.delete(message.id);
  if (message.error) reject(new Error(message.error.message));
  else resolvePromise(message.result);
});

function command(method, params = {}) {
  const id = nextId++;
  return new Promise((resolvePromise, reject) => {
    pending.set(id, { resolve: resolvePromise, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
}

await command("Page.enable");
await command("Runtime.enable");
await command("Emulation.setDeviceMetricsOverride", {
  width: 1890,
  height: 900,
  deviceScaleFactor: 1,
  mobile: false,
});

async function capture(filename, pathname, interaction) {
  await command("Page.navigate", { url: `http://localhost:3000${pathname}` });
  await sleep(3000);
  if (interaction) {
    await command("Runtime.evaluate", { expression: interaction });
    await sleep(2500);
  }
  const result = await command("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: false,
  });
  writeFileSync(resolve(outputDir, filename), Buffer.from(result.data, "base64"));
  console.log(`Captured ${filename}`);
}

try {
  await capture("01-overview-dashboard.png", "/");
  await capture("02-comment-explorer.png", "/comments");
  await capture(
    "03-live-prediction.png",
    "/predict",
    "Array.from(document.querySelectorAll('button')).find((button) => button.textContent.includes('Analyze'))?.click()",
  );
  await capture(
    "04-review-queue.png",
    "/review",
    "document.querySelector('main .divide-y button')?.click()",
  );
} finally {
  socket.close();
  edge.kill();
}
