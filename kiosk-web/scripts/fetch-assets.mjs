// Copies MediaPipe wasm into public/ and downloads the face landmarker model, so the kiosk works offline.
import { cpSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const wasmSrc = join(root, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const wasmDst = join(root, "public", "mediapipe");
mkdirSync(wasmDst, { recursive: true });
cpSync(wasmSrc, wasmDst, { recursive: true });
console.log("copied mediapipe wasm ->", wasmDst);

const modelDir = join(root, "public", "models");
const model = join(modelDir, "face_landmarker.task");
mkdirSync(modelDir, { recursive: true });
if (!existsSync(model)) {
  const url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
  console.log("downloading", url);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`model download failed: ${res.status}`);
  writeFileSync(model, Buffer.from(await res.arrayBuffer()));
  console.log("saved", model);
} else {
  console.log("model present", model);
}
