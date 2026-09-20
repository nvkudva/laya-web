// The bundler emits the asyncify and jsep wasm variants but not the plain threaded one
// the wasm-only build actually loads, so a production build hangs forever looking for a
// file that was never written. Serve ORT's runtime files ourselves and point env.wasm at them.
import { copyFile, mkdir, readdir } from "node:fs/promises";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const dist = dirname(createRequire(import.meta.url).resolve("onnxruntime-web"));
const out = new URL("../public/ort/", import.meta.url).pathname;
await mkdir(out, { recursive: true });
// Both the plain and the jsep runtime: which one ORT asks for depends on which
// onnxruntime-web entry the bundler resolved, and asking for a missing one surfaces
// as "no available backend found" rather than anything that names the file.
const want = (f) => /^ort-wasm-simd-threaded(\.jsep)?\.(wasm|mjs)$/.test(f);
const files = (await readdir(dist)).filter(want);
if (files.length !== 4) throw new Error(`expected 4 ORT runtime files, found ${files}`);
for (const f of files) await copyFile(join(dist, f), join(out, f));
console.log(`copied ${files.join(", ")} -> public/ort/`);
