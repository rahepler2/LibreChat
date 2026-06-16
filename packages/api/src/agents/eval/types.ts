/**
 * Types for the agent evaluation API.
 *
 * The evaluation endpoint runs an agent to completion and returns the full,
 * non-streamed detail of the run: the parameters the model selected for each
 * tool, the raw output each tool produced, any structured artifacts those
 * tools emitted (retrieved chunks, citations, generated files, search
 * results), and the final answer. It is intended for evaluating a tool
 * end-to-end — parameter selection, tool output, and answer creation.
 */
import type { CompletionUsage, ChatMessage } from '../openai/types';

/**
 * Evaluation request. Mirrors the chat-completions request shape (the agent
 * id is passed as `model`) but is always non-streaming.
 */
export interface EvaluationRequest {
  /** Agent ID to evaluate (maps to `model` per OpenAI convention) */
  model: string;
  /** Conversation messages */
  messages: ChatMessage[];
  /** Conversation ID (LibreChat extension) */
  conversation_id?: string;
  /** Parent message ID (LibreChat extension) */
  parent_message_id?: string;
}

/**
 * A structured artifact produced by a tool during the run — e.g. file_search
 * citations/chunks, web_search results, ui_resources, or a generated/code file.
 * The concrete payload is carried under the artifact's `type` key, matching the
 * attachment shape LibreChat already emits to clients.
 */
export interface EvaluationArtifact {
  /** Artifact/attachment discriminator (e.g. 'file_search', 'web_search') */
  type?: string;
  /** Tool call this artifact belongs to */
  toolCallId?: string;
  [key: string]: unknown;
}

/**
 * Full evaluation detail for a single tool invocation.
 */
export interface EvaluationToolCall {
  /** Tool call id assigned by the model */
  id: string;
  /** Tool name the model chose to call */
  name: string;
  /**
   * Parameters the model selected, parsed from the streamed JSON arguments.
   * `null` when the arguments were empty or not valid JSON (the raw string is
   * always preserved in `raw_arguments`).
   */
  arguments: unknown;
  /** Raw, unparsed argument string as emitted by the model */
  raw_arguments: string;
  /** Raw textual output the tool returned, when available */
  output: string | null;
  /** Whether the tool reported an error */
  error: boolean;
  /** Structured artifacts the tool emitted (chunks, citations, files, etc.) */
  artifacts: EvaluationArtifact[];
}

/**
 * The agent's final answer for the run.
 */
export interface EvaluationAnswer {
  /** Final assistant text */
  content: string | null;
  /** Reasoning/thinking content, when the model emitted any */
  reasoning: string | null;
}

/**
 * Non-streaming evaluation response.
 */
export interface EvaluationResponse {
  id: string;
  object: 'agent.evaluation';
  created: number;
  /** Agent id that was evaluated */
  model: string;
  /** Conversation id the run executed under */
  conversation_id: string;
  /** Final answer (parameter-selection + tool output lead here) */
  answer: EvaluationAnswer;
  /** Per-tool-call detail: chosen parameters, raw output, and artifacts */
  tool_calls: EvaluationToolCall[];
  /** Token usage for the run */
  usage: CompletionUsage;
  /** Why the run stopped */
  finish_reason: 'stop' | 'tool_calls';
}
