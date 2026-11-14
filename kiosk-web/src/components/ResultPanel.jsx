const REASON_TEXT = {
  phone_detected: "Face shown on a phone / screen",
  texture_spoof: "Print or screen texture detected",
  planar_motion: "Face is flat (no 3D depth)",
  low_liveness: "Liveness too low",
  identity_mismatch: "Face does not match claimed identity",
  unknown_person: "Person not enrolled",
  liveness_uncertain: "Liveness uncertain",
  identity_uncertain: "Identity uncertain",
  low_quality: "Image quality too low",
  too_dark: "Too dark",
  too_bright: "Too bright",
  blurry: "Image blurry",
  face_too_small: "Move closer",
  face_not_frontal: "Face the camera",
  no_face: "No face detected",
  multiple_faces: "More than one person",
  challenge_failed: "Additional check failed",
  actions_not_completed: "Actions not completed",
  head_turn_not_seen: "Head turn not seen",
  otp_invalid: "Invalid code",
};

const SIGNAL_LABEL = { minifas: "Texture CNN", dino: "DINOv2 head", device: "Screen device", planarity: "3D parallax" };

const BADGE = {
  ALLOW: { cls: "allow", icon: "✓", text: "ALLOW" },
  DENY: { cls: "deny", icon: "✕", text: "DENY" },
  REQUEST_VERIFICATION: { cls: "verify", icon: "!", text: "VERIFY" },
};

export default function ResultPanel({ result, clientMs }) {
  if (!result) return null;
  const b = BADGE[result.decision];
  const signals = result.liveness?.signals || {};
  return (
    <div className="result">
      <div className={`badge ${b.cls}`}>
        <span className="icon">{b.icon}</span>
        <span>{b.text}</span>
      </div>
      <p className="message">{result.message}</p>
      {result.identity?.user_id && result.decision === "ALLOW" && <p className="who">Welcome, {result.identity.user_id}</p>}

      <div className="stats">
        <div><span className="k">⏱ Decision time</span><span className="v">{(clientMs / 1000).toFixed(2)} s</span></div>
        <div><span className="k">🛡 Confidence</span><span className="v">{result.confidence.toFixed(2)}</span></div>
        <div><span className="k">Server time</span><span className="v">{Math.round(result.timings_ms?.total ?? 0)} ms</span></div>
        {result.identity?.similarity != null && (
          <div><span className="k">Match score</span><span className="v">{result.identity.similarity.toFixed(2)}</span></div>
        )}
      </div>

      {result.reasons?.length > 0 && (
        <ul className="reasons">
          {result.reasons.map((r) => <li key={r}>{REASON_TEXT[r] || r}</li>)}
        </ul>
      )}

      <div className="signals">
        {Object.entries(signals).map(([k, v]) => (
          <div className="signal" key={k}>
            <span className="name">{SIGNAL_LABEL[k] || k}</span>
            <div className="bar">{v != null && <div className={`fill ${v >= 0.5 ? "ok" : "bad"}`} style={{ width: `${v * 100}%` }} />}</div>
            <span className="val">{v == null ? "n/a" : v.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
