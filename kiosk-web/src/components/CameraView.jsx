// Mirrored live preview with corner-bracket face box. Box colour reflects state.
export default function CameraView({ videoRef, analysis, tone = "idle", caption }) {
  const box = analysis?.box;
  // preview is mirrored with CSS, so mirror the box x as well
  const style = box
    ? { left: `${(1 - box.x - box.w) * 100}%`, top: `${box.y * 100}%`, width: `${box.w * 100}%`, height: `${box.h * 100}%` }
    : null;
  return (
    <div className="camera">
      <video ref={videoRef} autoPlay playsInline muted className="mirror" />
      {style && (
        <div className={`facebox tone-${tone}`} style={style}>
          <span className="c tl" /><span className="c tr" /><span className="c bl" /><span className="c br" />
        </div>
      )}
      {caption && <div className="caption">{caption}</div>}
    </div>
  );
}
