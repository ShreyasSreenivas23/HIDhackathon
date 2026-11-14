import { useCallback, useEffect, useRef, useState } from "react";
import CameraView from "./components/CameraView";
import ChallengePanel from "./components/ChallengePanel";
import EnrollPanel from "./components/EnrollPanel";
import OtpPanel from "./components/OtpPanel";
import ResultPanel from "./components/ResultPanel";
import { api } from "./lib/api";
import { burst, grabFrame } from "./lib/capture";
import { actionDone, analyzeFrame, initFaceTracker, qualityIssues } from "./lib/faceTracker";

const STABLE_MS = 600;        // quality must be OK this long before auto-capture
const RESULT_HOLD_MS = 4000;  // how long a decision stays on screen
const CHALLENGE_S = 10;
const CHALLENGE_MAX_FRAMES = 14;

// phase: loading | scanning | verifying | result | challenge | otp
export default function App() {
  const videoRef = useRef(null);
  const analysisRef = useRef(null);
  const phaseRef = useRef("loading");
  const goodSinceRef = useRef(null);
  const challengeRef = useRef(null);

  const [tab, setTab] = useState("kiosk");
  const tabRef = useRef(tab);
  useEffect(() => { tabRef.current = tab; }, [tab]);
  const [phase, setPhaseState] = useState("loading");
  const [analysis, setAnalysis] = useState(null);
  const [result, setResult] = useState(null);
  const [clientMs, setClientMs] = useState(0);
  const [challengeUi, setChallengeUi] = useState(null);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState("");

  const setPhase = (p) => { phaseRef.current = p; setPhaseState(p); };

  const showResult = useCallback((r, ms) => {
    setResult(r);
    setClientMs(ms);
    if (r.decision === "REQUEST_VERIFICATION" && r.challenge) return startChallenge(r.challenge);
    setPhase("result");
    const offerOtp = r.decision === "DENY" && r.reasons?.includes("challenge_failed");
    setTimeout(() => {
      if (phaseRef.current !== "result") return;
      if (offerOtp) setPhase("otp");
      else { goodSinceRef.current = null; setPhase("scanning"); }
    }, offerOtp ? 2500 : RESULT_HOLD_MS);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------------------------------------------------------------- verify
  const runVerify = useCallback(async () => {
    setPhase("verifying");
    const t0 = performance.now();
    try {
      const a = analysisRef.current || {};
      const frames = await burst(videoRef.current, 5, 100);
      const r = await api.verify(frames, { blur: a.blur, brightness: a.brightness, face_px: a.facePx, yaw: a.yaw });
      showResult(r, performance.now() - t0);
    } catch (e) {
      setError(String(e));
      setPhase("scanning");
    }
  }, [showResult]);

  // ---------------------------------------------------------------- challenge
  const startChallenge = (ch) => {
    challengeRef.current = { ...ch, index: 0, events: [], periodic: [], actionFrames: [], start: performance.now(), lastGrab: 0, done: false };
    setChallengeUi({ actions: ch.actions, index: 0, secondsLeft: CHALLENGE_S });
    setPhase("challenge");
  };

  const finishChallenge = async () => {
    const c = challengeRef.current;
    if (!c || c.done) return;
    c.done = true;
    setPhase("verifying");
    const t0 = performance.now();
    try {
      // frames that show each action + the most recent periodic frames (server needs the head turns)
      const frames = [...c.periodic, ...c.actionFrames].slice(-CHALLENGE_MAX_FRAMES);
      const r = await api.challenge(c.nonce, frames, c.events);
      showResult(r, performance.now() - t0);
    } catch (e) {
      setError(String(e));
      showResult({ decision: "DENY", confidence: 1, reasons: ["challenge_failed"], message: "Challenge expired — use OTP", liveness: {}, identity: {}, timings_ms: {} }, 0);
    }
  };

  const tickChallenge = (a, now) => {
    const c = challengeRef.current;
    if (!c || c.done) return;
    const elapsed = now - c.start;
    if (now - c.lastGrab > 400) {
      c.periodic.push(grabFrame(videoRef.current, 0.85));
      if (c.periodic.length > 10) c.periodic.shift();
      c.lastGrab = now;
    }
    if (c.index < c.actions.length && actionDone(c.actions[c.index], a)) {
      c.events.push({ action: c.actions[c.index], t_ms: Math.round(elapsed) });
      c.actionFrames.push(grabFrame(videoRef.current, 0.85));     // keep the frame showing the action
      c.index += 1;
    }
    setChallengeUi({ actions: c.actions, index: c.index, secondsLeft: Math.max(0, Math.ceil(CHALLENGE_S - elapsed / 1000)) });
    if (c.index >= c.actions.length) setTimeout(finishChallenge, 300);
    else if (elapsed > CHALLENGE_S * 1000) finishChallenge();
  };

  // ---------------------------------------------------------------- camera + tracking loop
  useEffect(() => {
    let raf, stopped = false, lastUi = 0;
    api.health().then(setHealth).catch(() => setHealth({ ok: false }));
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: "user" }, audio: false });
        videoRef.current.srcObject = stream;
        await initFaceTracker();
        setPhase("scanning");
      } catch (e) {
        setError(`Camera / model init failed: ${e}`);
        return;
      }
      const loop = () => {
        if (stopped) return;
        const now = performance.now();
        const a = analyzeFrame(videoRef.current, now);
        analysisRef.current = a;
        if (now - lastUi > 80) { setAnalysis(a); lastUi = now; }
        const p = phaseRef.current;
        if (p === "scanning" && tabRef.current === "kiosk") {
          if (qualityIssues(a).length === 0) {
            goodSinceRef.current ??= now;
            if (now - goodSinceRef.current > STABLE_MS) runVerify();
          } else goodSinceRef.current = null;
        } else if (p === "challenge") tickChallenge(a, now);
        raf = requestAnimationFrame(loop);
      };
      loop();
    })();
    return () => { stopped = true; cancelAnimationFrame(raf); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------------------------------------------------------------- render
  const issues = qualityIssues(analysis);
  const tone = phase === "result" || phase === "otp"
    ? { ALLOW: "allow", DENY: "deny", REQUEST_VERIFICATION: "verify" }[result?.decision] || "idle"
    : phase === "challenge" ? "verify" : issues.length === 0 ? "ready" : "idle";
  const caption = phase === "scanning" ? (issues[0] || "Hold still…")
    : phase === "verifying" ? "Checking…" : phase === "loading" ? "Starting camera…" : null;

  return (
    <div className="app">
      <header>
        <div className="brand"><span className="hid">HID</span> Liveness Check</div>
        <nav>
          <button className={tab === "kiosk" ? "on" : ""} onClick={() => setTab("kiosk")}>Kiosk</button>
          <button className={tab === "enroll" ? "on" : ""} onClick={() => setTab("enroll")}>Enroll</button>
        </nav>
        <div className={`health ${health?.ok ? "ok" : "bad"}`}>
          {health ? (health.ok ? (health.stub_mode ? "API: stub mode" : `API ok · ${health.enrolled} enrolled`) : "API offline") : "…"}
        </div>
      </header>

      <main>
        <CameraView videoRef={videoRef} analysis={analysis} tone={tone} caption={tab === "kiosk" ? caption : null} />
        <aside>
          {tab === "enroll" ? (
            <EnrollPanel videoRef={videoRef} analysisRef={analysisRef} />
          ) : phase === "challenge" && challengeUi ? (
            <ChallengePanel {...challengeUi} />
          ) : phase === "otp" ? (
            <OtpPanel
              suggestedUser={result?.identity?.user_id}
              onResult={(r) => showResult(r, 0)}
              onCancel={() => { goodSinceRef.current = null; setPhase("scanning"); }}
            />
          ) : phase === "result" ? (
            <ResultPanel result={result} clientMs={clientMs} />
          ) : (
            <div className="idle">
              <h1>Step up to the camera</h1>
              <p>Look straight at the screen. Verification is automatic.</p>
              {error && <p className="err">{error}</p>}
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}
