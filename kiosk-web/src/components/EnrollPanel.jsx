import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { grabFrame } from "../lib/capture";
import { qualityIssues } from "../lib/faceTracker";

export default function EnrollPanel({ videoRef, analysisRef }) {
  const [user, setUser] = useState("");
  const [status, setStatus] = useState("");
  const [users, setUsers] = useState([]);
  const [busy, setBusy] = useState(false);

  const refresh = () => api.users().then((r) => setUsers(r.users)).catch(() => {});
  useEffect(() => { refresh(); }, []);

  const enroll = async () => {
    setBusy(true);
    setStatus("Hold still…");
    const frames = [];
    const deadline = Date.now() + 6000;
    while (frames.length < 3 && Date.now() < deadline) {
      if (qualityIssues(analysisRef.current).length === 0) frames.push(grabFrame(videoRef.current));
      await new Promise((r) => setTimeout(r, 300));
    }
    if (frames.length < 3) {
      setStatus(`Could not get clear frames: ${qualityIssues(analysisRef.current).join(", ")}`);
      setBusy(false);
      return;
    }
    try {
      const r = await api.enroll(user.trim(), frames);
      setStatus(r.ok ? `✓ Enrolled ${r.user_id} (${r.frames_used} frames)` : `✕ ${r.message}`);
      refresh();
    } catch (e) {
      setStatus(String(e));
    }
    setBusy(false);
  };

  return (
    <div className="enroll">
      <h2>Enroll a person</h2>
      <input placeholder="Employee ID / name" value={user} onChange={(e) => setUser(e.target.value)} />
      <button disabled={!user.trim() || busy} onClick={enroll}>{busy ? "Capturing…" : "Capture & enroll"}</button>
      <p className="status">{status}</p>
      <h3>Enrolled ({users.length})</h3>
      <ul className="users">{users.map((u) => <li key={u}>{u}</li>)}</ul>
    </div>
  );
}
