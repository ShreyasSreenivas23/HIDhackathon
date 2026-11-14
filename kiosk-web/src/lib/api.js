// Thin client for CONTRACT.md endpoints.
// Calls the API directly (CORS is open on the API) instead of through the Vite proxy: the proxy
// silently stops working after Vite auto-restarts and then returns index.html for /api/*.
// Point at another backend (e.g. AWS) with:  VITE_API_BASE=http://<host>:8000 npm run dev
// VITE_API_BASE="" (empty) = same origin: used when the API serves the kiosk itself (cloud / HTTPS deploy).
const BASE = import.meta.env.VITE_API_BASE ?? `${location.protocol}//${location.hostname}:8000`;
const OFFLINE_BASE = import.meta.env.VITE_OFFLINE_API_BASE || BASE;
const TIMEOUT_MS = 10000;

async function call(path, body, method = "POST", base = BASE) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  let res;
  try {
    res = await fetch(`${base}${path}`, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
      signal: ctrl.signal,
    });
  } catch (e) {
    const error = new Error(e.name === "AbortError" ? `API timed out (${base}${path})` : `API unreachable at ${base} — is it running?`);
    error.offline = true;
    throw error;
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
    throw new Error(`API at ${base} returned non-JSON (wrong URL?)`);
  }
}

import { enforce } from "./pqc";

// Every decision passes the post-quantum door-controller check (ML-DSA-65) before the UI sees it.
const pqcKey = () => call("/pqc/public-key", null, "GET");
let stubMode = false;
export function setStubMode(v) { stubMode = !!v; }
const signed = (p) => p.then((r) => enforce(r, pqcKey, { stub: stubMode }));

export const API_BASE = BASE;
export const OFFLINE_API_BASE = OFFLINE_BASE;
export const api = {
  health: () => call("/health", null, "GET"),
  users: () => call("/users", null, "GET"),
  userDetails: () => call("/users/details", null, "GET"),
  updateUser: (user_id, changes) => call(`/users/${encodeURIComponent(user_id)}`, changes, "PATCH"),
  deleteUser: (user_id) => call(`/users/${encodeURIComponent(user_id)}`, null, "DELETE"),
  schedules: () => call("/access/schedules", null, "GET"),
  schedule: (user_id) => call(`/access/schedules/${encodeURIComponent(user_id)}`, null, "GET"),
  enroll: (user_id, frames, options = {}) => call("/enroll", { user_id, frames, ...options }),
  verify: (frames, client_quality, claimed_user_id = null) => signed(call("/verify", { frames, client_quality, claimed_user_id })),
  challenge: (nonce, frames, client_events) => signed(call("/challenge", { nonce, frames, client_events })),
  otpRequest: (user_id) => call("/otp/request", { user_id }),
  otpVerify: (user_id, code) => signed(call("/otp/verify", { user_id, code })),
  offlineVerify: (payload) => signed(call("/offline/verify", payload, "POST", OFFLINE_BASE)),
  pqcPublicKey: pqcKey,
  offlineStatus: () => call("/offline/status", null, "GET", OFFLINE_BASE),
  offlineSync: (events) => call("/offline/sync", { events }, "POST", OFFLINE_BASE),
};

export function cacheSchedules(schedules) {
  localStorage.setItem("hid-access-schedules", JSON.stringify({ savedAt: Date.now(), schedules }));
}

export function cachedSchedules() {
  try { return JSON.parse(localStorage.getItem("hid-access-schedules") || "{}").schedules || []; }
  catch { return []; }
}

export function queueOfflineEvent(event) {
  const key = "hid-offline-events";
  let events = [];
  try { events = JSON.parse(localStorage.getItem(key) || "[]"); } catch { events = []; }
  if (!events.some((item) => item.event_id === event.event_id)) events.push(event);
  localStorage.setItem(key, JSON.stringify(events.slice(-100)));
}

export async function syncOfflineEvents() {
  let events = [];
  try { events = JSON.parse(localStorage.getItem("hid-offline-events") || "[]"); } catch { events = []; }
  if (!events.length) return { accepted: [], rejected: [], pending: 0 };
  const result = await api.offlineSync(events);
  const rejected = new Set((result.rejected || []).map((item) => item.event_id));
  const accepted = new Set(result.accepted || []);
  localStorage.setItem("hid-offline-events", JSON.stringify(events.filter((item) => !accepted.has(item.event_id) && !rejected.has(item.event_id))));
  return { ...result, pending: events.filter((item) => !accepted.has(item.event_id) && !rejected.has(item.event_id)).length };
}
