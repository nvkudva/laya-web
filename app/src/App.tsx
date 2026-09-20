import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DEFAULT_REQUEST, PRESETS, nonLatinFraction } from "./presets";
import { LayaSession, type LoadProgress } from "./laya/session";
import { Distribution } from "./Distribution";
import { JsonEditor, JsonView } from "./JsonCode";
import { Answers } from "./Answers";
import { Icon } from "./Icon";
import type { LayaConfig, LayaResponse } from "./laya/types";

const TOTAL_BYTES = 524_100_000;
const mb = (n: number) => (n / 1e6).toFixed(1);

interface Loading { files: Record<string, LoadProgress>; done: boolean }

export default function App() {
  const laya = useRef<LayaSession | null>(null);
  const [load, setLoad] = useState<Loading>({ files: {}, done: false });
  const [cfg, setCfg] = useState<LayaConfig | null>(null);
  const [request, setRequest] = useState(DEFAULT_REQUEST);
  const [response, setResponse] = useState<LayaResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ms, setMs] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [preset, setPreset] = useState(PRESETS[0].id);
  const [theme, setTheme] = useState<"system" | "light" | "dark">(() => {
    try { return (localStorage.getItem("laya-theme") as "light" | "dark") ?? "system"; } catch { return "system"; }
  });

  // Following prefers-color-scheme alone leaves no way out of an OS setting the
  // reader does not want for this page.
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    try {
      if (theme === "system") localStorage.removeItem("laya-theme");
      else localStorage.setItem("laya-theme", theme);
    } catch { /* private mode */ }
  }, [theme]);

  useEffect(() => {
    let live = true;
    const threads = Number(new URLSearchParams(location.search).get("threads")) || undefined;
    LayaSession.load(import.meta.env.VITE_MODELS_BASE || "/models/v1", (p: LoadProgress) => {
      if (live) setLoad((s) => ({ ...s, files: { ...s.files, [p.file]: p } }));
    }, threads).then((s) => {
      if (!live) return;
      laya.current = s;
      setCfg(s.cfg);
      setLoad((st) => ({ ...st, done: true }));
    }).catch((e) => live && setError(String(e?.message ?? e)));
    return () => { live = false; };
  }, []);

  const loaded = useMemo(
    () => Object.values(load.files).reduce((a, f) => a + f.loaded, 0),
    [load.files],
  );
  const fromCache = Object.values(load.files).some((f) => f.cached);

  const parsed = useMemo(() => {
    try {
      const v = JSON.parse(request);
      if (!v || typeof v !== "object" || !("questions" in v)) throw new Error("needs a questions object");
      return { ok: true as const, v };
    } catch (e) {
      return { ok: false as const, why: (e as Error).message };
    }
  }, [request]);

  const scriptWarning = useMemo(() => {
    if (!parsed.ok) return null;
    const s = (parsed.v as { state?: unknown }).state;
    const text = typeof s === "string" ? s : JSON.stringify(s ?? "");
    const f = nonLatinFraction(text);
    return f > 0.3 ? Math.round(f * 100) : null;
  }, [parsed]);

  const run = useCallback(async () => {
    if (!parsed.ok || !laya.current) return;
    setBusy(true); setError(null);
    const { state = "", questions } = parsed.v as { state?: any; questions: any };
    const t0 = performance.now();
    try {
      const res = await laya.current.systemOne(state, questions);
      setResponse(res); setMs(performance.now() - t0);
    } catch (e) {
      setError(String((e as Error)?.message ?? e)); setResponse(null);
    } finally {
      setBusy(false);
    }
  }, [parsed]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); run(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [run]);

  const nQuestions = parsed.ok ? Object.keys((parsed.v as { questions: object }).questions ?? {}).length : 0;

  // Switching scenario invalidates whatever is on the right; leaving it there would
  // read as this scenario's answer.
  const choosePreset = (p: (typeof PRESETS)[number]) => {
    setPreset(p.id);
    setRequest(p.request);
    setResponse(null);
    setError(null);
    setMs(null);
  };
  const active = PRESETS.find((p) => p.id === preset);
  const nAnswers = response ? Object.keys(response.answers).length : 0;

  return (
    <div className="shell">
      <header className="bar">
        <div className="bar-inner">
          <div className="bar-head">
            <h1 className="wordmark">Laya — System One decision model — running fully in browser</h1>
            <p className="tagline">
              It is not a chat model: it reads a state, scores the options you enumerate, and
              returns one calibrated distribution per question. It runs on onnxruntime-web over
              WebAssembly, so nothing leaves the browser. Question types: <b>noul</b> answers
              true or false, <b>choice</b> picks one named option, <b>score</b> returns the
              expectation over ordered levels.{" "}
              <a href="https://huggingface.co/nvkudva/laya-web-q8" target="_blank" rel="noreferrer">
                More details<Icon name="external" />
              </a>
            </p>
          </div>
          <dl className="readout">
            <div><dt>Checkpoint</dt><dd>ModernBERT-large</dd></div>
            <div><dt>Weights</dt><dd>8-bit + fp16, 524 MB</dd></div>
            <div><dt>Context</dt><dd>{cfg ? `${cfg.max_len} tok` : "—"}</dd></div>
            <div><dt>Last run</dt><dd>{ms === null ? "—" : `${Math.round(ms)} ms`}</dd></div>
            <button
              className="theme-toggle"
              onClick={() => setTheme((t) => (t === "dark" ? "light" : t === "light" ? "system" : "dark"))}
              title="Switch between dark, light and your system setting"
            >
              {theme}
            </button>
          </dl>
        </div>
      </header>

      {!load.done && (
        <section className="load" aria-live="polite">
          <div className="load-inner">
            <div className="load-line">
              <span>{fromCache ? "Reading weights from cache" : "Downloading weights"}</span>
              <b>{mb(loaded)} / {mb(TOTAL_BYTES)} MB</b>
            </div>
            <div className="rule"><span style={{ width: `${Math.min(100, (loaded / TOTAL_BYTES) * 100)}%` }} /></div>
          </div>
        </section>
      )}

      <main>
        <section className="panel">
          <div className="panel-head">
            <h2>Request <span className="panel-qualifier">(editable)</span></h2>
            <span className="panel-note">{nQuestions} question{nQuestions === 1 ? "" : "s"}</span>
          </div>
          <section className="presets-block" aria-labelledby="presets-heading">
          <h3 className="presets-heading" id="presets-heading">Presets</h3>
          <nav className="presets" aria-label="Example scenarios">
            {PRESETS.map((p) => (
              <button
                key={p.id}
                className="preset"
                aria-pressed={p.id === preset}
                onClick={() => choosePreset(p)}
              >
                {p.label}
              </button>
            ))}
          </nav>
          {active && <p className="preset-note">{active.note}</p>}
          </section>
          <JsonEditor
            value={request}
            onChange={(v: string) => { setRequest(v); setPreset(""); }}
            label="Request JSON"
          />
          {scriptWarning !== null && (
            <p className="notice">
              {scriptWarning}% of this state is non-Latin script. This checkpoint is English-only
              and stays confident while getting those wrong, so its confidence score will not
              warn you. Read the result as unreliable.
            </p>
          )}
          <div className="actions">
            <button onClick={run} disabled={!load.done || busy || !parsed.ok}>
              {busy ? "Running" : "Run request"}
            </button>
            <span className={`status${parsed.ok ? "" : " bad"}`}>
              {!load.done ? "Loading model" : !parsed.ok ? parsed.why : busy ? "One forward pass per question" : "Cmd+Enter to run"}
            </span>
          </div>
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Response — Distribution</h2>
            <span className="panel-note">{response ? `${nAnswers} answered` : ""}</span>
          </div>
          {error ? (
            <pre className="code json-view" style={{ color: "var(--warn)" }}>{error}</pre>
          ) : response ? (
            <div className="dist-scroll">
              <Answers answers={response.answers} />
              {Object.entries(response.answers).map(([id, ans]) => (
                <Distribution key={id} id={id} answer={ans} />
              ))}
            </div>
          ) : (
            <pre className="code json-view empty">Each answer is one probability distribution over the options you listed.</pre>
          )}
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Response — JSON</h2>
            <span className="panel-note">
              {response ? `${response.usage.input_tokens} input tokens` : ""}
            </span>
          </div>
          {response && !error ? (
            <JsonView src={JSON.stringify(response, null, 2)} />
          ) : (
            <pre className="code json-view empty">The raw system_one() response, ready to paste into your own integration.</pre>
          )}
        </section>
      </main>

      <footer>
        <div className="footer-inner">
          <section className="credit-group">
            <h2>Credits</h2>
            <ul className="credits">
              <li>
                <Icon name="person" />
                <a href="https://github.com/NandhaKishorM" target="_blank" rel="noreferrer">Nandakishor M</a>
                <span className="credit-note">Convai Innovations</span>
              </li>
              <li>
                <Icon name="model" />
                <a href="https://huggingface.co/convaiinnovations/laya" target="_blank" rel="noreferrer">convaiinnovations/laya</a>
              </li>
              <li>
                <Icon name="source" />
                <a href="https://github.com/NandhaKishorM/laya" target="_blank" rel="noreferrer">NandhaKishorM/laya</a>
              </li>
              <li>
                <Icon name="scale" />
                <a href="https://www.apache.org/licenses/LICENSE-2.0" target="_blank" rel="noreferrer">Apache 2.0</a>
              </li>
            </ul>
          </section>
          <section className="credit-group">
            <h2>Source</h2>
            <ul className="credits">
              <li>
                <Icon name="person" />
                <a href="https://github.com/nvkudva" target="_blank" rel="noreferrer">nvkudva</a>
                <span className="credit-note">quantization and this page</span>
              </li>
              <li>
                <Icon name="model" />
                <a href="https://huggingface.co/nvkudva/laya-web-q8" target="_blank" rel="noreferrer">nvkudva/laya-web-q8</a>
                <span className="credit-note">8-bit</span>
              </li>
              <li>
                <Icon name="source" />
                <a href="https://github.com/nvkudva/laya-web" target="_blank" rel="noreferrer">nvkudva/laya-web</a>
              </li>
            </ul>
          </section>
        </div>
      </footer>
    </div>
  );
}
