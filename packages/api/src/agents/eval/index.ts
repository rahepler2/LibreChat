/**
 * Agent evaluation API.
 *
 * Runs an agent to completion and returns the full, non-streamed detail of the
 * run — the parameters the model selected per tool call, the raw output each
 * tool returned, the structured artifacts those tools emitted (chunks,
 * citations, files, search results), and the final answer — so a tool can be
 * evaluated end-to-end. Authentication is via an admin-owned agent API key.
 */
export * from './types';
export * from './aggregator';
export * from './service';
