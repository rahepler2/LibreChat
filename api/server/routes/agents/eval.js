/**
 * Agent evaluation API route.
 *
 * Provides a non-streaming endpoint that runs an agent to completion and
 * returns the full detail of the run — the parameters the model selected per
 * tool call, the raw output each tool returned, the structured artifacts those
 * tools emitted (chunks, citations, files, search results), and the final
 * answer — so a tool can be evaluated end-to-end.
 *
 * Authentication: an agent API key (Bearer) owned by an ADMIN user. The key is
 * validated like any agent API key, then the owning user must hold the ADMIN
 * role.
 *
 * Usage:
 *   POST /api/agents/v1/evaluations
 *
 * Request body:
 *   {
 *     "model": "agent_id",        // Required: the agent to evaluate
 *     "messages": [{"role": "user", "content": "Hello!"}],
 *     "conversation_id": "...",   // Optional
 *     "parent_message_id": "..."  // Optional
 *   }
 */
const express = require('express');
const { requireAdmin, createRequireApiKeyAuth } = require('@librechat/api');
const { AgentEvaluationController } = require('~/server/controllers/agents/eval');
const { configMiddleware } = require('~/server/middleware');
const db = require('~/models');

const router = express.Router();

const apiKeyMiddleware = createRequireApiKeyAuth({
  validateAgentApiKey: db.validateAgentApiKey,
  findUser: db.findUser,
});

router.use(apiKeyMiddleware);
router.use(requireAdmin);
router.use(configMiddleware);

/**
 * @route POST /api/agents/v1/evaluations
 * @desc Run an agent and return full tool-output detail for evaluation
 * @access Admin (agent API key owned by an ADMIN user)
 */
router.post('/', AgentEvaluationController);

module.exports = router;
