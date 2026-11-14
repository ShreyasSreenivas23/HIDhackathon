import { useEffect, useState } from "react";
import { api, setAdminPin } from "../lib/api";
import { grabFrame } from "../lib/capture";
import { qualityIssues } from "../lib/faceTracker";

export default function EnrollPanel({ videoRef, analysisRef }) {
  const [user, setUser] = useState("");
  const [status, setStatus] = useState("");
  const [users, setUsers] = useState([]);
  const [authority, setAuthority] = useState("student");
  const [active, setActive] = useState(true);
  const [duration, setDuration] = useState("");
  const [schedule, setSchedule] = useState("24x7");
  const [pin, setPin] = useState(sessionStorage.getItem("hid-admin-pin") || "");
  const [busy, setBusy] = useState(false);
  const [now, setNow] = useState(Date.now());

  const refresh = () => api.userDetails().then(setUsers).catch(() => {});
  useEffect(() => { refresh(); }, []);
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);

  const enroll = async () => {
    // admin authentication only for the enroll ACTION (the page itself is open)
    let adminPin = pin;
    if (!adminPin) {
      adminPin = window.prompt("Admin PIN required to enroll") || "";
      if (!adminPin) return;
      setPin(adminPin);
      setAdminPin(adminPin);
    }
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
      const r = await api.enroll(user.trim(), frames, {
        authority,
        schedule_preset: schedule,
        active,
        ...(duration ? { expires_in_seconds: Number(duration) } : {}),
      });
      setStatus(r.ok ? `✓ Enrolled ${r.user_id} (${r.frames_used} frames)` : `✕ ${r.message}`);
      refresh();
    } catch (e) {
      if (String(e).includes("401")) { setPin(""); setAdminPin(""); }   // wrong PIN: ask again next time
      setStatus(String(e));
    }
    setBusy(false);
  };

  const removeUser = async (userId) => {
    if (!window.confirm(`Remove ${userId} from enrollment?`)) return;
    try {
      await api.deleteUser(userId);
      setStatus(`Removed ${userId}`);
      refresh();
    } catch (e) {
      setStatus(String(e));
    }
  };

  const toggleUser = async (profile) => {
    try {
      await api.updateUser(profile.user_id, { active: !profile.active });
      setStatus(`${profile.user_id} is now ${profile.active ? "inactive" : "active"}`);
      refresh();
    } catch (e) {
      setStatus(String(e));
    }
  };

  const clearExpiry = async (profile) => {
    try {
      await api.updateUser(profile.user_id, { expires_in_seconds: 0 });
      setStatus(`Expiry cleared for ${profile.user_id}`);
      refresh();
    } catch (e) {
      setStatus(String(e));
    }
  };

  return (
    <div className="enroll">
      <h2>Enroll a person</h2>
      <input placeholder="Employee ID / name" value={user} onChange={(e) => setUser(e.target.value)} />
      <label>Authority
        <select value={authority} onChange={(e) => setAuthority(e.target.value)}>
          <option value="student">Student</option>
          <option value="teacher">Teacher</option>
          <option value="admin">Admin</option>
        </select>
      </label>
      <label className="check"><input type="checkbox" checked={active} onChange={(e) => setActive(e.target.checked)} /> Active immediately</label>
      <label>Enrollment lifetime
        <select value={duration} onChange={(e) => setDuration(e.target.value)}>
          <option value="">No expiry</option>
          <option value="60">1 minute</option>
          <option value="3600">1 hour</option>
          <option value="86400">1 day</option>
        </select>
      </label>
      <label>Campus access
        <select value={schedule} onChange={(e) => setSchedule(e.target.value)}>
          <option value="24x7">Any time (24/7)</option>
          <option value="campus_hours">Campus hours (Mon–Fri 08:00–18:00)</option>
          <option value="none">No access (recognised but denied)</option>
        </select>
      </label>
      <button disabled={!user.trim() || busy} onClick={enroll}>{busy ? "Capturing…" : "Capture & enroll"}</button>
      <p className="status">{status}</p>
      <h3>Enrolled ({users.length})</h3>
      <ul className="users">{users.map((u) => {
        const expired = u.expires_at && new Date(u.expires_at).getTime() <= now;
        return <li key={u.user_id}>
        <span><strong>{u.user_id}</strong> · {u.authority} · {u.active ? (expired ? "expired" : "active") : "inactive"}{u.expires_at ? ` · expires ${new Date(u.expires_at).toLocaleString()}` : ""}</span>
        <button onClick={() => toggleUser(u)}>{u.active ? "Deactivate" : "Activate"}</button>
        {u.expires_at && <button onClick={() => clearExpiry(u)}>Clear expiry</button>}
        <button className="remove-user" onClick={() => removeUser(u.user_id)}>Remove</button>
      </li>;
      })}</ul>
    </div>
  );
}
