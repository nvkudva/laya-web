/** The request shape rl_agent_api.RLAgent.system_one accepts, as editable JSON. */
export const DEFAULT_REQUEST = `{
  "state": "Subject: Action required - your account will be suspended\\nFrom: security@paypa1-verify.com\\n\\nWe detected unusual activity. Confirm your password within 24 hours or your account will be closed permanently.",
  "questions": {
    "phishing": {
      "type": "noul",
      "instructions": "Is this email a phishing attempt?",
      "criteria": null
    },
    "route": {
      "type": "choice",
      "instructions": "Which team should handle this message?",
      "criteria": {
        "security": "fraud, phishing, account compromise",
        "billing": "invoices and payments",
        "support": "everything else"
      }
    },
    "severity": {
      "type": "score",
      "instructions": "How severe is this, if it is a threat?",
      "criteria": ["harmless", "suspicious", "dangerous", "critical"]
    }
  }
}`;

/** The English checkpoint scores 0.000 accuracy at 0.952 confidence on Khmer.
 *  It stays confident while being wrong, so confidence gating cannot catch it. */
export function nonLatinFraction(s: string): number {
  const letters = s.match(/\p{L}/gu);
  if (!letters?.length) return 0;
  const latin = s.match(/\p{Script=Latin}/gu)?.length ?? 0;
  return 1 - latin / letters.length;
}
