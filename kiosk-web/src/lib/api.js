// Thin client for CONTRACT.md endpoints.
// Calls the API directly (CORS is open on the API) instead of through the Vite proxy: the proxy
// silently stops working after Vite auto-restarts and then returns index.html for /api/*.
// Point at another backend (e.g. AWS) with:  VITE_API_BASE=http://<host>:8000 npm run dev
const BASE = import.meta.env.VITE_API_BASE || `${location.protocol}//${location.hostname}:8000`;
const TIMEOUT_MS = 10000;

async function call(path, body, method = "POST") {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  let res;
  try {
    res = await fetch(`${BASE}${path}`, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
      signal: ctrl.signal,
    });
  } catch (e) {
    throw new Error(e.name === "AbortError" ? `API timed out (${BASE}${path})` : `API unreachable at ${BASE} — is it running?`);
  } finally {
    clearTimeout(timer);
  }
  const text = await res.text();
  if (!res.ok) {
    let detail = text;
    try { detail = JSON.parse(text).detail ?? text; } catch { /* not json */ }
    throw new Error(`API ${res.status}: ${String(detail).slice(0, 200)}`);
  }
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`API at ${BASE} returned non-JSON (wrong URL?)`);
  }
}

export const API_BASE = BASE;
export const api = {
  health: () => call("/health", null, "GET"),
  users: () => call("/users", null, "GET"),
  enroll: (user_id, frames) => call("/enroll", { user_id, frames }),
  verify: (frames, client_quality, claimed_user_id = null) => call("/verify", { frames, client_quality, claimed_user_id }),
  challenge: (nonce, frames, client_events) => call("/challenge", { nonce, frames, client_events }),
  otpRequest: (user_id) => call("/otp/request", { user_id }),
  otpVerify: (user_id, code) => call("/otp/verify", { user_id, code }),
};
