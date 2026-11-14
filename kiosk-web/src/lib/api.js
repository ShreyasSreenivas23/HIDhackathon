// Thin client for CONTRACT.md endpoints. All paths go through the Vite proxy (/api -> :8000).
const BASE = import.meta.env.VITE_API_BASE || "/api";

async function call(path, body, method = "POST") {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status} ${await res.text()}`);
  return res.json();
}

export const api = {
  health: () => call("/health", null, "GET"),
  users: () => call("/users", null, "GET"),
  enroll: (user_id, frames) => call("/enroll", { user_id, frames }),
  verify: (frames, client_quality, claimed_user_id = null) => call("/verify", { frames, client_quality, claimed_user_id }),
  challenge: (nonce, frames, client_events) => call("/challenge", { nonce, frames, client_events }),
  otpRequest: (user_id) => call("/otp/request", { user_id }),
  otpVerify: (user_id, code) => call("/otp/verify", { user_id, code }),
};
