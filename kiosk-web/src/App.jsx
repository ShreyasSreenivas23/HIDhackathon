import { useCallback, useEffect, useRef, useState } from "react";
import CameraView from "./components/CameraView";
import ChallengePanel from "./components/ChallengePanel";
import EnrollPanel from "./components/EnrollPanel";
import OtpPanel from "./components/OtpPanel";
import ResultPanel from "./components/ResultPanel";
import { api, API_BASE, cacheSchedules, queueOfflineEvent, syncOfflineEvents } from "./lib/api";
import { startCamera } from "./lib/camera";
import { burst, grabFrame } from "./lib/capture";
import { actionDone, analyzeFrame, initFaceTracker, QUALITY, qualityIssues } from "./lib/faceTracker";

const DEBUG = new URLSearchParams(location.search).has("debug");

const STABLE_MS = 600;        // quality must be OK this long before auto-capture
const RESULT_HOLD_MS = 4000;  // how long a decision stays on screen
const NEXT_PERSON_MS = 6000;  // re-verify the same person only after this long
const CHALLENGE_S = 10;
const CHALLENGE_MAX_FRAMES = 14;

// phase: loading | scanning | verifying | result | challenge | otp
export default function App() {
  const videoRef = useRef(null);
  const analysisRef = useRef(null);
  const phaseRef = useRef("loading");
  const goodSinceRef = useRef(null);
  const challengeRef = useRef(null);
  const awaitNextRef = useRef(null);   // after a decision, wait for the person to step away (or timeout)
  const awayRef = useRef(null);

  const [tab, setTab] = useState("kiosk");
  const tabRef = useRef(tab);
  useEffect(() => { tabRef.current = tab; }, [tab]);
  const [phase, setPhaseState] = useState("loading");
  const [analysis, setAnalysis] = useState(null);
  const [result, setResult] = useState(null);
  const resultRef = useRef(null);
  const [clientMs, setClientMs] = useState(0);
  const [challengeUi, setChallengeUi] = useState(null);
  const [health, setHealth] = useState(null);
  const [online, setOnline] = useState(true);
  const [pendingSync, setPendingSync] = useState(0);
  const [error, setError] = useState("");

  const setPhase = (p) => { phaseRef.current = p; setPhaseState(p); };

  const showResult = useCallback((r, ms) => {
    resultRef.current = r;
    setResult(r);
    setClientMs(ms);
    if (r.decision === "REQUEST_VERIFICATION" && r.challenge) return startChallenge(r.challenge);
    setPhase("result");
    if (r.offline?.local === false) return;
    const offerOtp = r.decision === "DENY" && r.reasons?.includes("challenge_failed");
    setTimeout(() => {
      if (phaseRef.current !== "result") return;
      if (offerOtp) setPhase("otp");
      else { goodSinceRef.current = null; awaitNextRef.current = performance.now(); setPhase("scanning"); }
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
      setError("");
      setOnline(true);
      showResult(r, performance.now() - t0);
    } catch (e) {
      if (e.offline) {
        setOnline(false);
        const eventId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
        queueOfflineEvent({
          event_id: eventId,
          user_id: null,
          decision: "DENY",
          occurred_at: new Date().toISOString(),
          payload: { reason: "central_service_unavailable" },
        });
        setPendingSync((count) => count + 1);
        showResult({
          decision: "DENY",
          confidence: 1,
          reasons: ["offline_no_local_biometric"],
          message: "OFFLINE MODE — access denied until secure local verification is available",
          liveness: {},
          identity: {},
          quality: { ok: true },
          timings_ms: {},
          offline: { local: false, sync_status: "pending", event_id: eventId },
        }, performance.now() - t0);
        return;
      }
      setError(e.message || String(e));
      setPhase("error");
      setTimeout(() => { goodSinceRef.current = null; if (phaseRef.current === "error") setPhase("scanning"); }, 3000);
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
    const checkHealth = () => api.health().then((value) => {
      setHealth(value);
      setOnline(true);
      cacheSchedules(value.schedules || []);
      if (phaseRef.current === "result" && resultRef.current?.offline?.local === false) {
        setResult(null);
        resultRef.current = null;
        goodSinceRef.current = null;
        setPhase("scanning");
      }
      return syncOfflineEvents().then((sync) => setPendingSync(sync.pending || value.offline_events_pending || 0)).catch(() => {});
    }).catch(() => { setOnline(false); setHealth({ ok: false }); });
    checkHealth();
    const healthTimer = setInterval(checkHealth, 5000);
    (async () => {
      try {
        await startCamera(videoRef.current);
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
        if (awaitNextRef.current) {
          // person must be gone for 0.7 s (one missed detection is not "gone"), or the timeout passes
          awayRef.current = a?.faceCount ? null : (awayRef.current ?? now);
          if ((awayRef.current && now - awayRef.current > 700) || now - awaitNextRef.current > NEXT_PERSON_MS) {
            awaitNextRef.current = null;
            awayRef.current = null;
          }
        }
        if (p === "scanning" && tabRef.current === "kiosk" && !awaitNextRef.current) {
          if (qualityIssues(a).length === 0) {
            goodSinceRef.current ??= now;
            if (now - goodSinceRef.current > STABLE_MS) runVerify();
          } else goodSinceRef.current = null;
        } else if (p === "challenge") tickChallenge(a, now);
        raf = requestAnimationFrame(loop);
      };
      loop();
    })();
    return () => { stopped = true; cancelAnimationFrame(raf); clearInterval(healthTimer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------------------------------------------------------------- render
  const issues = qualityIssues(analysis);
  const tone = phase === "result" || phase === "otp"
    ? { ALLOW: "allow", DENY: "deny", REQUEST_VERIFICATION: "verify" }[result?.decision] || "idle"
    : phase === "challenge" ? "verify" : issues.length === 0 ? "ready" : "idle";
  const caption = phase === "scanning" ? (awaitNextRef.current ? "Next person, please" : issues[0] || "Hold still…")
    : phase === "verifying" ? "Checking…" : phase === "loading" ? "Starting camera…"
    : phase === "error" ? "Can't reach the server — retrying…" : null;

  return (
    <div className="app">
      <header>
        <div className="brand"><span className="hid">HID</span> Liveness Check</div>
        <nav>
          <button className={tab === "kiosk" ? "on" : ""} onClick={() => setTab("kiosk")}>Kiosk</button>
          <button className={tab === "enroll" ? "on" : ""} onClick={() => setTab("enroll")}>Enroll</button>
        </nav>
        <div className={`health ${online && health?.ok ? "ok" : "bad"}`}>
          {online && health?.ok ? (health.stub_mode ? "ONLINE · stub mode" : `ONLINE · ${health.enrolled} enrolled`) : `OFFLINE MODE (${API_BASE})`}
          {pendingSync > 0 && <span className="sync"> · {pendingSync} pending sync</span>}
        </div>
      </header>

      <main>
        <div className="left">
          <CameraView videoRef={videoRef} analysis={analysis} tone={tone} caption={tab === "kiosk" ? caption : null} />
          {DEBUG && (
            <pre className="debug">
              {JSON.stringify({
                phase,
                faces: analysis?.faceCount ?? null,
                facePx: analysis?.facePx && Math.round(analysis.facePx),
                brightness: analysis?.brightness?.toFixed(2),
                blur: analysis?.blur && Math.round(analysis.blur),
                yaw: analysis?.yaw?.toFixed(2),
                blink: analysis?.blink?.toFixed(2),
                smile: analysis?.smile?.toFixed(2),
                issues,
                limits: QUALITY,
              }, null, 1)}
            </pre>
          )}
        </div>
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
