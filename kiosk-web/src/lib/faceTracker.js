// MediaPipe Face Landmarker running in the browser: quality gate + challenge action detection.
// All geometry is on RAW (unmirrored) video coordinates, matching CONTRACT.md.
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const NOSE = 1;
const IRIS_IMG_LEFT = 468; // iris centres (478-landmark model)
const IRIS_IMG_RIGHT = 473;
const EYE_IMG_LEFT = 33; // fallback: outer eye corners
const EYE_IMG_RIGHT = 263;

export const QUALITY = {
  minFacePx: 110, // a bit stricter than the server (90) so we rarely get rejected there
  minBrightness: 0.2,
  maxBrightness: 0.9,
  maxAbsYaw: 0.3,
  minBlur: 40,
};
export const ACTION_THRESH = { yaw: 0.25, blink: 0.5, smile: 0.5 };

let landmarker = null;

export async function initFaceTracker() {
  if (landmarker) return landmarker;
  const fileset = await FilesetResolver.forVisionTasks("/mediapipe");
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: "/models/face_landmarker.task", delegate },
    runningMode: "VIDEO",
    numFaces: 2,
    outputFaceBlendshapes: true,
  });
  try {
    landmarker = await FaceLandmarker.createFromOptions(fileset, opts("GPU"));
  } catch {
    landmarker = await FaceLandmarker.createFromOptions(fileset, opts("CPU"));
  }
  return landmarker;
}

const sample = document.createElement("canvas");
sample.width = sample.height = 64;
const sctx = sample.getContext("2d", { willReadFrequently: true });

// brightness (0..1) + Laplacian variance on a 64x64 grayscale crop of the face
function faceStats(video, box) {
  const W = video.videoWidth, H = video.videoHeight;
  sctx.drawImage(video, box.x * W, box.y * H, box.w * W, box.h * H, 0, 0, 64, 64);
  const d = sctx.getImageData(0, 0, 64, 64).data;
  const g = new Float32Array(64 * 64);
  let sum = 0;
  for (let i = 0; i < 64 * 64; i++) {
    g[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
    sum += g[i];
  }
  let lsum = 0, lsq = 0, n = 0;
  for (let y = 1; y < 63; y++)
    for (let x = 1; x < 63; x++) {
      const i = y * 64 + x;
      const l = g[i - 64] + g[i + 64] + g[i - 1] + g[i + 1] - 4 * g[i];
      lsum += l; lsq += l * l; n++;
    }
  const mean = lsum / n;
  return { brightness: sum / (64 * 64) / 255, blur: lsq / n - mean * mean };
}

function blend(result, name) {
  const cats = result.faceBlendshapes?.[0]?.categories || [];
  return cats.find((c) => c.categoryName === name)?.score ?? 0;
}

/** Analyze the current video frame. Returns null if no face. */
export function analyzeFrame(video, timestampMs) {
  if (!landmarker || video.readyState < 2) return null;
  const res = landmarker.detectForVideo(video, timestampMs);
  const faces = res.faceLandmarks || [];
  if (!faces.length) return { faceCount: 0 };
  const lm = faces[0];
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of lm) {
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  }
  const box = { x: Math.max(0, minX), y: Math.max(0, minY), w: Math.min(1, maxX) - Math.max(0, minX), h: Math.min(1, maxY) - Math.max(0, minY) };
  const W = video.videoWidth, H = video.videoHeight;
  const eL = lm[IRIS_IMG_LEFT] || lm[EYE_IMG_LEFT];
  const eR = lm[IRIS_IMG_RIGHT] || lm[EYE_IMG_RIGHT];
  const eyeMidX = ((eL.x + eR.x) / 2) * W;
  const eyeDist = Math.hypot((eR.x - eL.x) * W, (eR.y - eL.y) * H) + 1e-6;
  const yaw = (lm[NOSE].x * W - eyeMidX) / eyeDist; // >0: user turned to THEIR left
  const stats = faceStats(video, box);
  return {
    faceCount: faces.length,
    box, // normalized, raw coordinates
    facePx: Math.min(box.w * W, box.h * H),
    yaw,
    blink: Math.min(blend(res, "eyeBlinkLeft"), blend(res, "eyeBlinkRight")),
    smile: Math.min(blend(res, "mouthSmileLeft"), blend(res, "mouthSmileRight")),
    ...stats,
  };
}

/** Returns [] if good to capture, else a list of human-readable guidance strings. */
export function qualityIssues(a) {
  if (!a || a.faceCount === 0) return ["Look at the camera"];
  const out = [];
  if (a.faceCount > 1) out.push("One person at a time");
  if (a.facePx < QUALITY.minFacePx) out.push("Move closer");
  if (a.brightness < QUALITY.minBrightness) out.push("More light needed");
  if (a.brightness > QUALITY.maxBrightness) out.push("Too bright");
  if (Math.abs(a.yaw) > QUALITY.maxAbsYaw) out.push("Face the camera");
  if (a.blur < QUALITY.minBlur) out.push("Hold still");
  return out;
}

export function actionDone(action, a) {
  if (!a || !a.faceCount) return false;
  switch (action) {
    case "blink": return a.blink > ACTION_THRESH.blink;
    case "smile": return a.smile > ACTION_THRESH.smile;
    case "turn_left": return a.yaw > ACTION_THRESH.yaw;
    case "turn_right": return a.yaw < -ACTION_THRESH.yaw;
    default: return false;
  }
}

export const ACTION_LABEL = {
  blink: "Blink your eyes",
  smile: "Smile",
  turn_left: "Turn your head to your LEFT",
  turn_right: "Turn your head to your RIGHT",
};
