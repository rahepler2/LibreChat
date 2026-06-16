/**
 * Aggregator for the agent evaluation endpoint.
 *
 * Unlike the OpenAI-compatible aggregator — which discards tool output — this
 * one captures everything needed to evaluate a tool end-to-end: the parameters
 * the model selected per tool call, the raw output each tool returned, the
 * final answer, reasoning, and token usage. Structured tool artifacts (chunks,
 * citations, files) are resolved separately by the controller and merged in at
 * response-build time.
 */
import type { CompletionUsage } from '../openai/types';
import type {
  EvaluationArtifact,
  EvaluationResponse,
  EvaluationToolCall,
  EvaluationAnswer,
} from './types';

/** Accumulated state for a single tool call as it streams in. */
interface PendingToolCall {
  id: string;
  name: string;
  /** Argument fragments, joined lazily to avoid O(n²) concatenation. */
  argChunks: string[];
}

/** Captured result for a completed tool call, keyed by tool_call_id. */
interface ToolResult {
  content: string | null;
  error: boolean;
}

export interface EvaluationAggregator {
  /** Final answer text fragments */
  textChunks: string[];
  /** Reasoning/thinking fragments */
  reasoningChunks: string[];
  /** Tool calls in arrival order, keyed by their streamed index */
  toolCallsByIndex: Map<number, PendingToolCall>;
  /** Tool outputs keyed by tool_call_id */
  toolResultsById: Map<string, ToolResult>;
  /** Token usage accumulated across model calls */
  usage: {
    promptTokens: number;
    completionTokens: number;
    reasoningTokens: number;
  };
  addText: (text: string) => void;
  addReasoning: (text: string) => void;
  /** Register a tool call's id + name when the model opens it */
  startToolCall: (index: number, id: string, name: string) => void;
  /** Append streamed argument fragments to an open tool call */
  appendToolArgs: (index: number, args: string) => void;
  /** Record a tool's raw output once it finishes */
  recordToolResult: (toolCallId: string, content: string | null, error: boolean) => void;
  getText: () => string;
  getReasoning: () => string;
}

/**
 * Create an evaluation aggregator.
 */
export function createEvaluationAggregator(): EvaluationAggregator {
  const textChunks: string[] = [];
  const reasoningChunks: string[] = [];
  const toolCallsByIndex = new Map<number, PendingToolCall>();
  const toolResultsById = new Map<string, ToolResult>();

  return {
    textChunks,
    reasoningChunks,
    toolCallsByIndex,
    toolResultsById,
    usage: { promptTokens: 0, completionTokens: 0, reasoningTokens: 0 },
    addText: (text) => textChunks.push(text),
    addReasoning: (text) => reasoningChunks.push(text),
    startToolCall: (index, id, name) => {
      if (toolCallsByIndex.has(index)) {
        return;
      }
      toolCallsByIndex.set(index, { id, name, argChunks: [] });
    },
    appendToolArgs: (index, args) => {
      const pending = toolCallsByIndex.get(index);
      if (pending) {
        pending.argChunks.push(args);
      }
    },
    recordToolResult: (toolCallId, content, error) => {
      toolResultsById.set(toolCallId, { content, error });
    },
    getText: () => textChunks.join(''),
    getReasoning: () => reasoningChunks.join(''),
  };
}

/** Parse a streamed JSON argument string, tolerating empty/invalid payloads. */
function parseToolArguments(raw: string): unknown {
  const trimmed = raw.trim();
  if (trimmed === '') {
    return null;
  }
  try {
    return JSON.parse(trimmed);
  } catch {
    return null;
  }
}

/**
 * Build the rich, non-streaming evaluation response from the aggregator plus
 * the resolved tool artifacts (grouped by tool_call_id by the controller).
 */
export function buildEvaluationResponse(
  context: { requestId: string; model: string; created: number; conversationId: string },
  aggregator: EvaluationAggregator,
  artifactsByToolCall: Map<string, EvaluationArtifact[]>,
  usage: CompletionUsage,
): EvaluationResponse {
  const toolCalls: EvaluationToolCall[] = [];

  for (const pending of aggregator.toolCallsByIndex.values()) {
    const rawArguments = pending.argChunks.join('');
    const result = aggregator.toolResultsById.get(pending.id);
    toolCalls.push({
      id: pending.id,
      name: pending.name,
      arguments: parseToolArguments(rawArguments),
      raw_arguments: rawArguments,
      output: result?.content ?? null,
      error: result?.error ?? false,
      artifacts: artifactsByToolCall.get(pending.id) ?? [],
    });
  }

  const text = aggregator.getText();
  const reasoning = aggregator.getReasoning();
  const answer: EvaluationAnswer = {
    content: text === '' ? null : text,
    reasoning: reasoning === '' ? null : reasoning,
  };

  const finishReason = toolCalls.length > 0 && text === '' ? 'tool_calls' : 'stop';

  return {
    id: context.requestId,
    object: 'agent.evaluation',
    created: context.created,
    model: context.model,
    conversation_id: context.conversationId,
    answer,
    tool_calls: toolCalls,
    usage,
    finish_reason: finishReason,
  };
}
