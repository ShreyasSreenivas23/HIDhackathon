// End-to-end check of the post-quantum door-controller logic against the running API.
//   node scripts/pqc_e2e.mjs [path/to/real_burst_dir] [path/to/attack_burst_dir]
import fs from "fs";
import path from "path";
import { pathToFileURL } from "url";

const { enforce } = await import(pathToFileURL(path.resolve("src/lib/pqc.js")).href);
const API = process.env.API || "http://localhost:8000";
const call = async (p, body) =>
  (await fetch(API + p, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {})).json();
const key = () => call("/pqc/public-key");
const burst = (d) => fs.readdirSync(d).filter((f) => f.startsWith("frame_")).sort().map((f) => fs.readFileSync(path.join(d, f)).toString("base64"));

const realDir = process.argv[2] || "../data/debug/1794638316320_verify_ALLOW";
const atkDir = process.argv[3] || "../data/debug/1794638369366_verify_REQUEST_VERIFICATION";

const real = await call("/verify", { frames: burst(realDir) });
const r1 = await enforce(real, key);
console.log("real person          :", r1.decision, "| pqc:", r1.pqc.reason, `${r1.pqc.ms} ms`, "| server sign ms:", real.timings_ms.pqc_sign);
const r2 = await enforce(real, key);
console.log("same decision replay :", r2.decision === "ALLOW" ? "ALLOW (BAD)" : r2.decision, "| pqc:", r2.pqc.reason);

const atk = await call("/verify", { frames: burst(atkDir) });
const forged = structuredClone(atk);
forged.decision = "ALLOW";
forged.attestation.payload = forged.attestation.payload.replace(/"decision":"[A-Z_]+"/, '"decision":"ALLOW"');
const r3 = await enforce(forged, key);
console.log("phone attack         :", atk.decision, "-> forged to ALLOW ->", r3.decision, r3.reasons, "| pqc:", r3.pqc.reason);

const stripped = structuredClone(real);
delete stripped.attestation;
const r4 = await enforce(stripped, key);
console.log("ALLOW, sig stripped  :", r4.decision, "| pqc:", r4.pqc.reason);

console.log("server-side verify   :", await call("/pqc/verify", { payload: real.attestation.payload, signature_b64: real.attestation.signature_b64 }));
