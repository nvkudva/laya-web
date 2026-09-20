import { PreTrainedTokenizer } from "@huggingface/transformers";
import type { Tok } from "./sequence";

/** transformers.js tokenizer built straight from tokenizer.json -- no model needed. */
export async function loadTokenizer(base: string): Promise<Tok> {
  const [tj, tc] = await Promise.all([
    fetch(`${base}/tokenizer.json`).then((r) => r.json()),
    fetch(`${base}/tokenizer_config.json`).then((r) => r.json()),
  ]);
  const t = new PreTrainedTokenizer(tj, tc);
  // encode the literal special token rather than reaching into the model internals
  const id = (s: string) => {
    const ids = t.encode(s, { add_special_tokens: false }) as number[];
    if (ids.length !== 1) throw new Error(`${s} did not tokenize to a single id: ${ids}`);
    return ids[0];
  };
  return {
    encode: (text, opts) => t.encode(text, opts) as number[],
    maskToken: "[MASK]",
    maskTokenId: id("[MASK]"),
    clsTokenId: id("[CLS]"),
    sepTokenId: id("[SEP]"),
    padTokenId: id("[PAD]"),
  };
}
