const REASON_TEXT = {
  phone_detected: "Face shown on a phone / screen",
  signature_invalid: "Decision signature invalid — door stays closed",
  texture_disagreement: "Texture checks disagree — quick check needed",
  screen_texture: "Phone / screen detected (screen texture)",
  texture_spoof: "Image texture looks unusual — quick check needed",
  planar_motion: "Face looks flat (photo or screen)",
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
  outside_schedule: "Outside permitted hours",
  permission_disabled: "Campus permission disabled",
  no_campus_permission: "No campus permission",
  no_cached_permission: "No cached campus permission",
  offline_no_local_biometric: "Secure local biometric verification unavailable",
};

const SIGNAL_LABEL = { minifas: "Texture CNN", dino: "DINOv2 head", device: "Screen device", planarity: "3D parallax" };

const BADGE = {
  ALLOW: { cls: "allow", icon: "✓", text: "ALLOW" },
  DENY: { cls: "deny", icon: "✕", text: "DENY" },
  REQUEST_VERIFICATION: { cls: "verify", icon: "!", text: "VERIFY" },
};

/** Who the system thinks this is: recognised (>= identity_hi), possible (gray zone), or not recognised. */
export function identityView(identity, thresholds) {
  const name = identity?.user_id, sim = identity?.similarity;
  const lo = thresholds?.identity_lo ?? 0.28, hi = thresholds?.identity_hi ?? 0.4;
  if (!name || sim == null) return { cls: "none", text: "Not recognised", sub: "no enrolled face matched" };
  if (identity.matched || sim >= hi) return { cls: "yes", text: name, sub: `recognised · match ${sim.toFixed(2)}` };
  if (sim >= lo) return { cls: "maybe", text: `${name}?`, sub: `uncertain match ${sim.toFixed(2)}` };
  return { cls: "none", text: "Not recognised", sub: `closest enrolled: ${name} (${sim.toFixed(2)})` };
}

export default function ResultPanel({ result, clientMs, thresholds }) {
  if (!result) return null;
  const b = BADGE[result.decision];
  const signals = result.liveness?.signals || {};
  const who = identityView(result.identity, thresholds);
  return (
    <div className="result">
      <div className={`badge ${b.cls}`}>
        <span className="icon">{b.icon}</span>
        <span>{b.text}</span>
      </div>
      <div className={`identity ${who.cls}`}>
        <span className="avatar">{who.cls === "none" ? "?" : who.text.slice(0, 1).toUpperCase()}</span>
        <span><span className="name">{who.text}</span><span className="sub">{who.sub}</span></span>
      </div>
      <p className="message">{result.message}</p>
      {result.offline?.local === false && <p className="offline-note">Decision made locally: fail-closed offline policy</p>}
      {result.schedule && (
        <p className={`schedule-note ${result.schedule.authorized ? "scheduled" : "not-scheduled"}`}>
          {result.schedule.authorized ? "Campus schedule: authorized" : `Campus schedule: ${result.schedule.reason.replaceAll("_", " ")}`}
          {" · "}{result.schedule.timezone}
        </p>
      )}
      {who.cls === "yes" && result.decision === "ALLOW" && <p className="who">Welcome, {who.text}</p>}

      <div className="stats">
        <div><span className="k">⏱ Decision time</span><span className="v">{(clientMs / 1000).toFixed(2)} s</span></div>
        <div><span className="k">🛡 Confidence</span><span className="v">{result.confidence.toFixed(2)}</span></div>
        <div><span className="k">Server time</span><span className="v">{Math.round(result.timings_ms?.total ?? 0)} ms</span></div>
        {result.identity?.similarity != null && (
          <div><span className="k">Match score</span><span className="v">{result.identity.similarity.toFixed(2)}</span></div>
        )}
      </div>

      {result.pqc && (
        <div className={`pqc ${result.pqc.ok ? "ok" : "bad"}`}>
          {result.pqc.ok
            ? <>🔐 Post-quantum signed · {result.pqc.alg} verified in {result.pqc.ms} ms · key {result.pqc.key_id?.slice(0, 8)}</>
            : <>⚠ Signature check: {result.pqc.reason}</>}
        </div>
      )}

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
