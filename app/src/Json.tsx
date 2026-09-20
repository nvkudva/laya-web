import { useEffect, useRef, type ChangeEvent, type UIEvent } from "react";

/**
 * JSON colouring for both a read-only view and an editable one. The editor is the
 * usual overlay: a transparent textarea sitting exactly on top of a highlighted
 * <pre>, with scroll kept in sync. Both share `.code` so the two layers cannot
 * drift in font, size, padding or line height -- if they do, the caret lands in the
 * wrong place.
 */

type Tok = "key" | "str" | "num" | "lit" | "punct";

// Keys are strings followed by a colon, so the string rule has to look ahead.
const RE = /("(?:\\.|[^"\\])*")(\s*:)?|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|\b(true|false|null)\b|([{}[\],:])/g;

function tokens(src: string): Array<[Tok | null, string]> {
  const out: Array<[Tok | null, string]> = [];
  let last = 0;
  for (const m of src.matchAll(RE)) {
    const i = m.index!;
    if (i > last) out.push([null, src.slice(last, i)]);
    if (m[1] !== undefined) {
      out.push([m[2] ? "key" : "str", m[1]]);
      if (m[2]) out.push(["punct", m[2]]);
    } else if (m[3] !== undefined) out.push(["num", m[3]]);
    else if (m[4] !== undefined) out.push(["lit", m[4]]);
    else out.push(["punct", m[5]]);
    last = i + m[0].length;
  }
  if (last < src.length) out.push([null, src.slice(last)]);
  return out;
}

function Highlighted({ src }: { src: string }) {
  return (
    <>
      {tokens(src).map(([t, text], i) =>
        t ? <span key={i} className={`t-${t}`}>{text}</span> : <span key={i}>{text}</span>,
      )}
    </>
  );
}

export function JsonView({ src }: { src: string }) {
  return <pre className="code json-view"><code><Highlighted src={src} /></code></pre>;
}

export function JsonEditor({
  value, onChange, label,
}: { value: string; onChange: (v: string) => void; label: string }) {
  const pre = useRef<HTMLPreElement>(null);
  const ta = useRef<HTMLTextAreaElement>(null);

  // Keep the painted layer aligned when the value changes from outside (a preset
  // click), not just when the user scrolls.
  useEffect(() => {
    if (pre.current && ta.current) {
      pre.current.scrollTop = ta.current.scrollTop;
      pre.current.scrollLeft = ta.current.scrollLeft;
    }
  }, [value]);

  const sync = (e: UIEvent<HTMLTextAreaElement>) => {
    if (!pre.current) return;
    pre.current.scrollTop = e.currentTarget.scrollTop;
    pre.current.scrollLeft = e.currentTarget.scrollLeft;
  };

  return (
    <div className="editor">
      <pre ref={pre} className="code editor-paint" aria-hidden="true">
        <code>
          <Highlighted src={value} />
          {"\n"}
        </code>
      </pre>
      <textarea
        ref={ta}
        className="code editor-input"
        value={value}
        onChange={(e: ChangeEvent<HTMLTextAreaElement>) => onChange(e.target.value)}
        onScroll={sync}
        spellCheck={false}
        aria-label={label}
      />
    </div>
  );
}
