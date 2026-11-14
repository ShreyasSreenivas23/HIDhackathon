// Offline PQC check: decisions signed by the LOCAL edge API must verify against the edge key, not the cloud key.
//   node scripts/pqc_offline_e2e.mjs
import fs from "fs";
import path from "path";
import { pathToFileURL } from "url";

const { enforce } = await import(pathToFileURL(path.resolve("src/lib/pqc.js")).href);
const EDGE = process.env.EDGE || "http://localhost:8000";
const CENTRAL = process.env.CENTRAL || "https://duuh07ci1fhe6.cloudfront.net";
const get = async (base, p) => (await fetch(base + p)).json();
const post = async (base, p, body) =>
  (await fetch(base + p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
const dir = process.argv[2] || "../data/debug/1794638316320_verify_ALLOW";
const frames = fs.readdirSync(dir).filter((f) => f.startsWith("frame_")).sort().map((f) => fs.readFileSync(path.join(dir, f)).toString("base64"));

const off = await post(EDGE, "/offline/verify-frames", { frames });
const ck = await get(CENTRAL, "/pqc/public-key"), ek = await get(EDGE, "/pqc/public-key");
console.log("keys: central", ck.key_id, "| edge", ek.key_id, "| decision signed by", off.attestation?.key_id);
const old = await enforce(structuredClone(off), () => Promise.resolve(ck), { source: "central-only" });
console.log("BEFORE fix (cloud key only):", old.decision, old.reasons, "| pqc:", old.pqc.reason);
const fixed = await enforce(structuredClone(off), () => Promise.resolve(ek), { source: "edge" });
console.log("AFTER fix (edge key)       :", fixed.decision, fixed.reasons, "| pqc:", fixed.pqc.reason, `${fixed.pqc.ms} ms`, "| signer:", fixed.pqc.signer);
const forged = structuredClone(off);
forged.attestation.payload = forged.attestation.payload.replace(/"decision":"[A-Z_]+"/, '"decision":"ALLOW"'); forged.decision = "ALLOW";
const f2 = await enforce(forged, () => Promise.resolve(ek), { source: "edge" });
console.log("offline forged/replayed    :", f2.decision, "| pqc:", f2.pqc.reason);
