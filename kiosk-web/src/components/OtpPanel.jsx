import { useState } from "react";
import { api } from "../lib/api";

// Simulated OTP fallback. Production: code is sent by SMS (Amazon SNS) to the enrolled phone.
export default function OtpPanel({ suggestedUser, onResult, onCancel }) {
  const [user, setUser] = useState(suggestedUser || "");
  const [demoCode, setDemoCode] = useState(null);
  const [code, setCode] = useState("");
  const [err, setErr] = useState("");

  const request = async () => {
    setErr("");
    try {
      const r = await api.otpRequest(user);
      setDemoCode(r.demo_code);
    } catch (e) { setErr(String(e)); }
  };
  const verify = async () => {
    try { onResult(await api.otpVerify(user, code)); } catch (e) { setErr(String(e)); }
  };

  return (
    <div className="otp">
      <h2>Verify with one-time code</h2>
      {!demoCode ? (
        <>
          <input placeholder="Employee ID" value={user} onChange={(e) => setUser(e.target.value)} />
          <button disabled={!user} onClick={request}>Send code</button>
        </>
      ) : (
        <>
          <p className="sms">📱 Simulated SMS: your code is <b>{demoCode}</b></p>
          <input placeholder="6-digit code" value={code} maxLength={6} onChange={(e) => setCode(e.target.value)} />
          <button disabled={code.length !== 6} onClick={verify}>Verify</button>
        </>
      )}
      <button className="link" onClick={onCancel}>Cancel</button>
      {err && <p className="err">{err}</p>}
    </div>
  );
}
