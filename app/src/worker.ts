// The encoder is 2.4s at the 512-token limit. On the main thread that is a frozen tab.
import { LayaSession, type LoadProgress } from "./laya/session";
import type { Questions, State } from "./laya/types";

// Point this at the Hugging Face repo for deploys; the local copy is for development.
const MODELS_BASE = import.meta.env.VITE_MODELS_BASE ?? "/models";

let laya: LayaSession | null = null;

type Req =
  | { id: number; kind: "load" }
  | { id: number; kind: "run"; state: State; questions: Questions };

self.onmessage = async (e: MessageEvent<Req>) => {
  const msg = e.data;
  try {
    if (msg.kind === "load") {
      laya = await LayaSession.load(MODELS_BASE, (p: LoadProgress) =>
        self.postMessage({ kind: "progress", ...p }));
      self.postMessage({ id: msg.id, kind: "loaded", cfg: laya.cfg });
      return;
    }
    if (!laya) throw new Error("model is still loading");
    const t0 = performance.now();
    const res = await laya.systemOne(msg.state, msg.questions);
    self.postMessage({ id: msg.id, kind: "result", res, ms: performance.now() - t0 });
  } catch (err) {
    self.postMessage({ id: msg.id, kind: "error", error: String((err as Error)?.message ?? err) });
  }
};
