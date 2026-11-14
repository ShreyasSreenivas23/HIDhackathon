// Grab RAW (unmirrored) frames from the <video> as base64 JPEG.
const canvas = document.createElement("canvas");

export function grabFrame(video, quality = 0.9) {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  return canvas.toDataURL("image/jpeg", quality);
}

export async function burst(video, n = 5, intervalMs = 100) {
  const frames = [];
  for (let i = 0; i < n; i++) {
    frames.push(grabFrame(video));
    if (i < n - 1) await new Promise((r) => setTimeout(r, intervalMs));
  }
  return frames;
}
