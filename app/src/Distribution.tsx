import type { Answer } from "./laya/types";

/**
 * One question's answer is one distribution over its options, so each block is a
 * single series: one validated fill on a neutral track, no categorical palette and
 * therefore no legend. Rank is carried by ink weight on the label, never by hue.
 */

interface Row { label: string; p: number; index: number }

function rows(a: Answer): Row[] {
  if (a.type === "noul") {
    return [
      { label: "false", p: 1 - a.noul, index: 0 },
      { label: "true", p: a.noul, index: 1 },
    ];
  }
  if (a.type === "score") {
    // Keep the level number next to its name: the answer is an expectation in level
    // units, so "1.36" is only readable against a numbered scale.
    return Object.entries(a.probabilities).map(([k, p], index) => ({
      label: `${k} ${a.legend[k] ?? ""}`.trim(), p, index,
    }));
  }
  return Object.entries(a.probabilities).map(([label, p], index) => ({ label, p, index }));
}

const pct = (p: number) => (p * 100 >= 9.95 ? (p * 100).toFixed(0) : (p * 100).toFixed(1));

function Bars({ data, winner }: { data: Row[]; winner: number }) {
  return (
    <ol className="bars">
      {data.map((r) => (
        <li
          key={r.label}
          className={r.index === winner ? "bar-row is-top" : "bar-row"}
          title={`${r.label} — ${r.p.toFixed(4)}`}
        >
          <span className="bar-label">{r.label}</span>
          <span className="bar-track">
            {/* min-width keeps a near-zero probability visible as a sliver rather than nothing */}
            <span className="bar-fill" style={{ width: `max(2px, ${r.p * 100}%)` }} />
          </span>
          <span className="bar-value">{pct(r.p)}<span className="pct">%</span></span>
        </li>
      ))}
    </ol>
  );
}

export function Distribution({ id, answer }: { id: string; answer: Answer }) {
  const data = rows(answer);
  const winner = data.reduce((best, r) => (r.p > data[best].p ? r.index : best), 0);
  const summary =
    answer.type === "choice" ? answer.choice
    : answer.type === "score" ? answer.score.toFixed(2)
    : answer.noul.toFixed(4);
  const confidence = answer.type === "noul" ? null : answer.confidence;

  return (
    <section
      className="dist"
      role="img"
      aria-label={`${id}: ${data.map((r) => `${r.label} ${pct(r.p)} percent`).join(", ")}`}
    >
      <header className="dist-head">
        <h3>{id}</h3>
        <span className="dist-type">{answer.type}</span>
        <span className="dist-answer">{summary}</span>
      </header>

      <Bars data={data} winner={winner} />

      <footer className="dist-foot">
        <span>
          {answer.type === "score"
            ? `expectation, ${data.length} levels`
            : answer.type === "noul"
              ? "p(true)"
              : `${data.length} options`}
        </span>
        <span className="dist-meta">
          {confidence !== null && <span>confidence {confidence.toFixed(3)}</span>}
          <span>act {answer.rl_agent.act_probability.toFixed(3)}</span>
        </span>
      </footer>
    </section>
  );
}
