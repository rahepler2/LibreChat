/**
 * Service helpers for the agent evaluation endpoint.
 *
 * Validation reuses the chat-completions validator (identical request fields,
 * minus streaming). Artifact grouping turns the flat list of resolved tool
 * attachments into a per-tool-call lookup for the response builder.
 */
import { validateRequest, isChatCompletionValidationFailure } from '../openai/service';
import type { EvaluationRequest, EvaluationArtifact } from './types';

export type EvaluationValidationSuccess = { valid: true; request: EvaluationRequest };
export type EvaluationValidationFailure = { valid: false; error: string };
export type EvaluationValidationResult =
  | EvaluationValidationSuccess
  | EvaluationValidationFailure;

/** Type guard for evaluation validation failure. */
export function isEvaluationValidationFailure(
  result: EvaluationValidationResult,
): result is EvaluationValidationFailure {
  return !result.valid;
}

/**
 * Validate an evaluation request. Delegates to the shared chat-completions
 * validator (same `model` + `messages` contract) and drops streaming concerns.
 */
export function validateEvaluationRequest(body: unknown): EvaluationValidationResult {
  const result = validateRequest(body);
  if (isChatCompletionValidationFailure(result)) {
    return { valid: false, error: result.error };
  }
  const { model, messages, conversation_id, parent_message_id } = result.request;
  return {
    valid: true,
    request: { model, messages, conversation_id, parent_message_id },
  };
}

/**
 * Group resolved tool attachments by their `toolCallId` so each evaluated tool
 * call can carry its own artifacts (chunks, citations, files, search results).
 * Attachments without a `toolCallId` are skipped — they can't be attributed to
 * a specific tool invocation.
 */
export function groupArtifactsByToolCall(
  attachments: Array<EvaluationArtifact | null | undefined>,
): Map<string, EvaluationArtifact[]> {
  const grouped = new Map<string, EvaluationArtifact[]>();
  for (const attachment of attachments) {
    const toolCallId = attachment?.toolCallId;
    if (typeof toolCallId !== 'string' || toolCallId === '') {
      continue;
    }
    const existing = grouped.get(toolCallId);
    if (existing) {
      existing.push(attachment as EvaluationArtifact);
      continue;
    }
    grouped.set(toolCallId, [attachment as EvaluationArtifact]);
  }
  return grouped;
}
