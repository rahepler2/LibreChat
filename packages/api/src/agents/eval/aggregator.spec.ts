import { createEvaluationAggregator, buildEvaluationResponse } from './aggregator';
import type { CompletionUsage } from '../openai/types';
import type { EvaluationArtifact } from './types';

const context = {
  requestId: 'eval-123',
  model: 'agent_test',
  created: 1700000000,
  conversationId: 'convo-1',
};

const emptyUsage: CompletionUsage = {
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
};

describe('createEvaluationAggregator', () => {
  it('accumulates text and reasoning fragments in order', () => {
    const aggregator = createEvaluationAggregator();
    aggregator.addText('Hello');
    aggregator.addText(' world');
    aggregator.addReasoning('thinking');
    aggregator.addReasoning(' more');

    expect(aggregator.getText()).toBe('Hello world');
    expect(aggregator.getReasoning()).toBe('thinking more');
  });

  it('records a tool call once and appends streamed argument fragments', () => {
    const aggregator = createEvaluationAggregator();
    aggregator.startToolCall(0, 'call_1', 'web_search');
    // duplicate start should not reset the call
    aggregator.startToolCall(0, 'other', 'other');
    aggregator.appendToolArgs(0, '{"query":');
    aggregator.appendToolArgs(0, '"cats"}');

    const pending = aggregator.toolCallsByIndex.get(0);
    expect(pending).toBeDefined();
    expect(pending?.id).toBe('call_1');
    expect(pending?.name).toBe('web_search');
    expect(pending?.argChunks.join('')).toBe('{"query":"cats"}');
  });

  it('records tool results keyed by tool_call_id', () => {
    const aggregator = createEvaluationAggregator();
    aggregator.recordToolResult('call_1', 'result text', false);
    aggregator.recordToolResult('call_2', null, true);

    expect(aggregator.toolResultsById.get('call_1')).toEqual({
      content: 'result text',
      error: false,
    });
    expect(aggregator.toolResultsById.get('call_2')).toEqual({
      content: null,
      error: true,
    });
  });
});

describe('buildEvaluationResponse', () => {
  it('merges chosen parameters, raw output, and artifacts per tool call', () => {
    const aggregator = createEvaluationAggregator();
    aggregator.startToolCall(0, 'call_1', 'file_search');
    aggregator.appendToolArgs(0, '{"query":"refunds"}');
    aggregator.recordToolResult('call_1', 'Found 2 sources', false);
    aggregator.addText('Refunds take 5 days.');

    const artifact: EvaluationArtifact = {
      type: 'file_search',
      toolCallId: 'call_1',
      file_search: { sources: [{ fileId: 'f1', relevance: 0.9 }] },
    };
    const artifactsByToolCall = new Map<string, EvaluationArtifact[]>([['call_1', [artifact]]]);

    const usage: CompletionUsage = {
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
    };

    const response = buildEvaluationResponse(context, aggregator, artifactsByToolCall, usage);

    expect(response.object).toBe('agent.evaluation');
    expect(response.id).toBe('eval-123');
    expect(response.conversation_id).toBe('convo-1');
    expect(response.answer.content).toBe('Refunds take 5 days.');
    expect(response.tool_calls).toHaveLength(1);

    const [toolCall] = response.tool_calls;
    expect(toolCall.id).toBe('call_1');
    expect(toolCall.name).toBe('file_search');
    expect(toolCall.arguments).toEqual({ query: 'refunds' });
    expect(toolCall.raw_arguments).toBe('{"query":"refunds"}');
    expect(toolCall.output).toBe('Found 2 sources');
    expect(toolCall.error).toBe(false);
    expect(toolCall.artifacts).toEqual([artifact]);
    expect(response.finish_reason).toBe('stop');
  });

  it('leaves arguments null when the model emitted no/invalid JSON', () => {
    const aggregator = createEvaluationAggregator();
    aggregator.startToolCall(0, 'call_1', 'noop_tool');

    const response = buildEvaluationResponse(context, aggregator, new Map(), emptyUsage);
    const [toolCall] = response.tool_calls;
    expect(toolCall.arguments).toBeNull();
    expect(toolCall.raw_arguments).toBe('');
    expect(toolCall.output).toBeNull();
    expect(toolCall.artifacts).toEqual([]);
  });

  it('reports tool_calls finish_reason when there are tool calls but no answer text', () => {
    const aggregator = createEvaluationAggregator();
    aggregator.startToolCall(0, 'call_1', 'web_search');
    aggregator.appendToolArgs(0, '{"query":"x"}');

    const response = buildEvaluationResponse(context, aggregator, new Map(), emptyUsage);
    expect(response.answer.content).toBeNull();
    expect(response.answer.reasoning).toBeNull();
    expect(response.finish_reason).toBe('tool_calls');
  });

  it('reports stop finish_reason for a plain text answer with no tools', () => {
    const aggregator = createEvaluationAggregator();
    aggregator.addText('Just an answer.');

    const response = buildEvaluationResponse(context, aggregator, new Map(), emptyUsage);
    expect(response.tool_calls).toHaveLength(0);
    expect(response.answer.content).toBe('Just an answer.');
    expect(response.finish_reason).toBe('stop');
  });
});
