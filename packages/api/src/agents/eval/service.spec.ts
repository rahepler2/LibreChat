import {
  groupArtifactsByToolCall,
  validateEvaluationRequest,
  isEvaluationValidationFailure,
} from './service';
import type { EvaluationArtifact } from './types';

describe('validateEvaluationRequest', () => {
  it('accepts a valid request and passes through the LibreChat extension fields', () => {
    const result = validateEvaluationRequest({
      model: 'agent_test',
      messages: [{ role: 'user', content: 'hi' }],
      conversation_id: 'convo-1',
      parent_message_id: 'msg-1',
    });

    expect(isEvaluationValidationFailure(result)).toBe(false);
    if (!isEvaluationValidationFailure(result)) {
      expect(result.request.model).toBe('agent_test');
      expect(result.request.messages).toHaveLength(1);
      expect(result.request.conversation_id).toBe('convo-1');
      expect(result.request.parent_message_id).toBe('msg-1');
    }
  });

  it('rejects a request without a model (agent id)', () => {
    const result = validateEvaluationRequest({
      messages: [{ role: 'user', content: 'hi' }],
    });
    expect(isEvaluationValidationFailure(result)).toBe(true);
    if (isEvaluationValidationFailure(result)) {
      expect(result.error).toMatch(/model/);
    }
  });

  it('rejects a request with an empty messages array', () => {
    const result = validateEvaluationRequest({ model: 'agent_test', messages: [] });
    expect(isEvaluationValidationFailure(result)).toBe(true);
  });

  it('rejects a request with an invalid message role', () => {
    const result = validateEvaluationRequest({
      model: 'agent_test',
      messages: [{ role: 'wizard', content: 'hi' }],
    });
    expect(isEvaluationValidationFailure(result)).toBe(true);
  });
});

describe('groupArtifactsByToolCall', () => {
  it('groups multiple artifacts under their tool call id', () => {
    const a: EvaluationArtifact = { type: 'file_search', toolCallId: 'call_1' };
    const b: EvaluationArtifact = { type: 'image', toolCallId: 'call_1' };
    const c: EvaluationArtifact = { type: 'web_search', toolCallId: 'call_2' };

    const grouped = groupArtifactsByToolCall([a, b, c]);
    expect(grouped.get('call_1')).toEqual([a, b]);
    expect(grouped.get('call_2')).toEqual([c]);
  });

  it('skips null/undefined and unattributed artifacts', () => {
    const a: EvaluationArtifact = { type: 'file_search', toolCallId: 'call_1' };
    const orphan: EvaluationArtifact = { type: 'web_search' };

    const grouped = groupArtifactsByToolCall([a, null, undefined, orphan]);
    expect(grouped.size).toBe(1);
    expect(grouped.get('call_1')).toEqual([a]);
  });
});
