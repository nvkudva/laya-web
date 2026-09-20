export type QType = "choice" | "score" | "noul";

/** Jev request shape, as rl_agent_api.RLAgent.system_one accepts it. */
export interface QuestionDef {
  type: QType;
  instructions: string | unknown;
  criteria?: Record<string, string | null> | string[] | null;
}
export type Questions = Record<string, QuestionDef>;
export type State = string | Record<string, unknown> | unknown[];

/** Internal form after _to_internal: criteria normalised, instructions stringified. */
export interface InternalQ {
  t: QType;
  ins: string;
  crit: Record<string, string | null> | string[] | null;
}

export interface LayaConfig {
  max_len: number;
  head_max_len: number;
  temperature: [number, number, number];
  temperature_by_options: Record<string, number>;
}

export type Answer =
  | { type: "choice"; choice: string; probabilities: Record<string, number>; confidence: number; rl_agent: { act_probability: number } }
  | { type: "score"; score: number; legend: Record<string, string>; probabilities: Record<string, number>; confidence: number; rl_agent: { act_probability: number } }
  | { type: "noul"; noul: number; rl_agent: { act_probability: number } };

export interface LayaResponse {
  model: string;
  answers: Record<string, Answer>;
  usage: { input_tokens: number; output_tokens: number };
}

export const QTYPES: Record<QType, number> = { choice: 0, score: 1, noul: 2 };
export const QTYPE_NAMES: QType[] = ["choice", "score", "noul"];
