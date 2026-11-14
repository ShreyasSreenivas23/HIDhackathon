// Camera source. Normally the webcam; for testing without a camera use a URL param:
//   ?src=/test/face.jpg   (image, gently jittered so it is not perfectly static)
//   ?src=/test/clip.mp4   (looping video file, e.g. a recorded replay attack)
export async function startCamera(video) {
  const src = new URLSearchParams(location.search).get("src");
  if (!src) {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    await video.play().catch(() => {});
    return "webcam";
  }
  if (/\.(mp4|webm|mov)$/i.test(src)) {
    video.srcObject = null;
    video.src = src;
    video.loop = true;
    await video.play();
    return `file:${src}`;
  }
  const img = new Image();
  img.src = src;
  await img.decode();
  const c = document.createElement("canvas");
  c.width = img.width;
  c.height = img.height;
  const ctx = c.getContext("2d");
  let t = 0;
  setInterval(() => {
    t += 1;
    ctx.fillStyle = "#282828";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(img, Math.sin(t / 7) * 4, Math.cos(t / 9) * 3);
  }, 1000 / 30);
  video.srcObject = c.captureStream(30);
  await video.play();
  return `image:${src}`;
}
