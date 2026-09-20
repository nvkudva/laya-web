import type { Answer } from "./laya/types";

/** The decision, stripped of its distribution: what the model would actually do.
 *  The bars below say how sure it is; this says what it picked. */

function decide(a: Answer): { value: string; detail?: string; confidence: number } {
  if (a.type === "noul") {
    return {
      value: a.noul >= 0.5 ? "true" : "false",
      detail: `p(true) ${a.noul.toFixed(3)}`,
      // a binary at p=0.5 is maximally unsure; at 0 or 1, certain
      confidence: Math.abs(a.noul - 0.5) * 2,
    };
  }
  if (a.type === "score") {
    const nearest = Math.round(a.score);
    return {
      value: a.score.toFixed(2),
      detail: a.legend[String(nearest)] ?? undefined,
      confidence: a.confidence,
    };
  }
  return { value: a.choice, confidence: a.confidence };
}

export function Answers({ answers }: { answers: Record<string, Answer> }) {
  return (
    <section className="answers" aria-labelledby="answers-heading">
      <h3 className="answers-heading" id="answers-heading">Answers</h3>
      <dl className="answer-list">
        {Object.entries(answers).map(([id, a]) => {
          const d = decide(a);
          return (
            <div className="answer" key={id}>
              <dt>
                {id}
                <span className="answer-type">{a.type}</span>
              </dt>
              <dd>
                <span className="answer-value">{d.value}</span>
                {d.detail && <span className="answer-detail">{d.detail}</span>}
                <span
                  className="answer-meter"
                  title={`confidence ${d.confidence.toFixed(3)}`}
                  aria-label={`confidence ${d.confidence.toFixed(3)}`}
                >
                  <span style={{ width: `${Math.max(2, d.confidence * 100)}%` }} />
                </span>
              </dd>
            </div>
          );
        })}
      </dl>
    </section>
  );
}
