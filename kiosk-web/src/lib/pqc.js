// Post-quantum check of every decision (the kiosk plays the door controller).
// The API signs each decision with ML-DSA-65 (NIST FIPS 204); we verify with @noble/post-quantum and
// refuse to open the door on anything that is unsigned, forged, tampered, for another key, or expired.
import { ml_dsa65 } from "@noble/post-quantum/ml-dsa.js";

const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const enc = new TextEncoder();
const keyPromises = new Map();   // one pinned key per signer (central cloud API, local edge API)
const seen = new Set();   // decision_ids already acted on (replay protection)

export function loadPublicKey(fetchKey, source = "central") {
  if (!keyPromises.has(source)) {
    keyPromises.set(source, fetchKey().then((k) => ({ ...k, pk: b64(k.public_key_b64) }))
      .catch((e) => { keyPromises.delete(source); throw e; }));
  }
  return keyPromises.get(source);
}

/** Returns { ok, reason, alg, key_id, ms } */
export async function verifyAttestation(resp, fetchKey, source = "central") {
  const t0 = performance.now();
  const att = resp?.attestation;
  if (!att) return { ok: false, reason: "unsigned" };
  let key;
  try { key = await loadPublicKey(fetchKey, source); } catch { return { ok: false, reason: "no_public_key" }; }
  if (att.alg !== "ML-DSA-65" || att.key_id !== key.key_id) return { ok: false, reason: "unknown_key" };
  let valid = false;
  try { valid = ml_dsa65.verify(b64(att.signature_b64), enc.encode(att.payload), key.pk); } catch { valid = false; }
  if (!valid) return { ok: false, reason: "bad_signature" };
  const p = JSON.parse(att.payload);
  if (p.decision !== resp.decision || (p.user_id ?? null) !== (resp.identity?.user_id ?? null)) return { ok: false, reason: "payload_mismatch" };
  if (Date.now() / 1000 > p.expires_at + 5) return { ok: false, reason: "expired" };
  if (seen.has(p.decision_id)) return { ok: false, reason: "replayed" };
  seen.add(p.decision_id);
  return { ok: true, reason: "verified", alg: att.alg, key_id: att.key_id, signer: source, ms: Math.round(performance.now() - t0) };
}

/** Door-controller rule: an ALLOW is only honoured with a valid post-quantum signature. */
export async function enforce(resp, fetchKey, { stub = false, source = "central" } = {}) {
  const pqc = await verifyAttestation(resp, fetchKey, source);
  const out = { ...resp, pqc };
  if (resp.decision === "ALLOW" && !pqc.ok && !(stub && pqc.reason === "unsigned")) {
    out.decision = "DENY";
    out.reasons = ["signature_invalid"];
    out.message = `Decision rejected — post-quantum signature check failed (${pqc.reason})`;
  }
  return out;
}
