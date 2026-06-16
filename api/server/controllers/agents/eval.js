const { nanoid } = require('nanoid');
const { logger } = require('@librechat/data-schemas');
const { Callback, ToolEndHandler, formatAgentMessages } = require('@librechat/agents');
const {
  EModelEndpoint,
  ResourceType,
  PermissionBits,
  hasPermissions,
  AgentCapabilities,
} = require('librechat-data-provider');
const {
  createRun,
  buildToolSet,
  loadSkillStates,
  createSafeUser,
  initializeAgent,
  getBalanceConfig,
  injectSkillPrimes,
  createErrorResponse,
  extractManualSkills,
  recordCollectedUsage,
  getTransactionsConfig,
  resolveRecursionLimit,
  findPiiMatchInMessages,
  discoverConnectedAgents,
  groupArtifactsByToolCall,
  getRemoteAgentPermissions,
  buildEvaluationResponse,
  createToolExecuteHandler,
  resolveAgentScopedSkillIds,
  createEvaluationAggregator,
  validateEvaluationRequest,
  isEvaluationValidationFailure,
} = require('@librechat/api');
const {
  buildSummarizationHandlers,
  markSummarizationUsage,
  createToolEndCallback,
  agentLogHandlerObj,
} = require('~/server/controllers/agents/callbacks');
const { loadAgentTools, loadToolsForExecution } = require('~/server/services/ToolService');
const {
  findAccessibleResources,
  getEffectivePermissions,
} = require('~/server/services/PermissionService');
const {
  getSkillToolDeps,
  getSkillDbMethods,
  canAuthorSkillFiles,
  withDeploymentSkillIds,
  buildAgentToolContext,
  enrichLoadedToolsWithAgentContext,
} = require('~/server/services/Endpoints/agents/skillDeps');
const { getModelsConfig } = require('~/server/controllers/ModelController');
const { logViolation } = require('~/cache');
const db = require('~/models');

/**
 * Creates a tool loader function for the agent. Mirrors the OpenAI-compatible
 * controller: returns serializable tool definitions for event-driven execution.
 * @param {AbortSignal} signal - The abort signal
 */
function createToolLoader(signal) {
  return async function loadTools({
    req,
    res,
    tools,
    model,
    agentId,
    provider,
    tool_options,
    tool_resources,
  }) {
    const agent = { id: agentId, tools, provider, model, tool_options };
    try {
      return await loadAgentTools({
        req,
        res,
        agent,
        signal,
        tool_resources,
        definitionsOnly: true,
        streamId: null,
      });
    } catch (error) {
      logger.error('Error loading tools for agent ' + agentId, error);
    }
  };
}

/**
 * Convert content part to internal format
 * @param {Object} part - Content part
 * @returns {Object} Converted part
 */
function convertContentPart(part) {
  if (part.type === 'text') {
    return { type: 'text', text: part.text };
  }
  if (part.type === 'image_url') {
    return { type: 'image_url', image_url: part.image_url };
  }
  return part;
}

/**
 * Convert OpenAI-style messages to internal format
 * @param {Array} messages - Request messages
 * @returns {Array} Internal format messages
 */
function convertMessages(messages) {
  return messages.map((msg) => {
    let content;
    if (typeof msg.content === 'string') {
      content = msg.content;
    } else if (msg.content) {
      content = msg.content.map(convertContentPart);
    } else {
      content = '';
    }

    return {
      role: msg.role,
      content,
      ...(msg.name && { name: msg.name }),
      ...(msg.tool_calls && { tool_calls: msg.tool_calls }),
      ...(msg.tool_call_id && { tool_call_id: msg.tool_call_id }),
    };
  });
}

/**
 * Send an error response
 */
function sendErrorResponse(res, statusCode, message, type = 'invalid_request_error', code = null) {
  res.status(statusCode).json(createErrorResponse(message, type, code));
}

/**
 * Normalize a tool's raw output into a string for evaluation capture.
 * @param {unknown} content
 * @returns {string | null}
 */
function normalizeToolOutput(content) {
  if (content == null) {
    return null;
  }
  if (typeof content === 'string') {
    return content;
  }
  try {
    return JSON.stringify(content);
  } catch {
    return String(content);
  }
}

/**
 * Agent evaluation controller.
 *
 * POST /api/agents/v1/evaluations
 *
 * Runs an agent to completion (non-streaming) and returns the full detail of
 * the run for end-to-end tool evaluation: the parameters the model selected per
 * tool call, the raw output each tool returned, the structured artifacts those
 * tools emitted (chunks, citations, files, search results), the final answer,
 * and token usage.
 *
 * Authentication: admin-owned agent API key (Bearer) + ADMIN role.
 *
 * Request body:
 * {
 *   "model": "agent_id",          // Required: the agent to evaluate
 *   "messages": [...],            // Required: chat messages
 *   "conversation_id": "...",     // Optional: conversation context
 *   "parent_message_id": "..."    // Optional: parent message for threading
 * }
 */
const AgentEvaluationController = async (req, res) => {
  const appConfig = req.config;
  const requestStartTime = Date.now();

  const validation = validateEvaluationRequest(req.body);
  if (isEvaluationValidationFailure(validation)) {
    return sendErrorResponse(res, 400, validation.error);
  }

  const request = validation.request;
  const agentId = request.model;

  const agent = await db.getAgent({ id: agentId });
  if (!agent) {
    return sendErrorResponse(
      res,
      404,
      `Agent not found: ${agentId}`,
      'invalid_request_error',
      'model_not_found',
    );
  }

  const piiHit = findPiiMatchInMessages(request.messages, appConfig?.messageFilter?.pii);
  if (piiHit != null) {
    return sendErrorResponse(
      res,
      400,
      `Message contains a ${piiHit.label}. Remove it and try again.`,
      'invalid_request_error',
      'message_filter_pii_block',
    );
  }

  const responseId = `eval-${nanoid()}`;
  const created = Math.floor(Date.now() / 1000);

  const abortController = new AbortController();
  req.on('close', () => {
    if (!abortController.signal.aborted) {
      abortController.abort();
      logger.debug('[Agent Eval] Client disconnected, aborting');
    }
  });

  try {
    if (request.conversation_id != null) {
      if (!(await db.getConvo(req.user?.id, request.conversation_id))) {
        return sendErrorResponse(res, 404, 'Conversation not found', 'invalid_request_error');
      }
    }

    const conversationId = request.conversation_id ?? nanoid();
    const parentMessageId = request.parent_message_id ?? null;

    const context = { requestId: responseId, model: agentId, created, conversationId };

    const agentsEConfig = appConfig?.endpoints?.[EModelEndpoint.agents];
    const allowedProviders = new Set(agentsEConfig?.allowedProviders);

    const loadTools = createToolLoader(abortController.signal);

    const endpointOption = {
      endpoint: agent.provider,
      model_parameters: agent.model_parameters ?? {},
    };
    const skillDbMethods = getSkillDbMethods();

    const dbMethods = {
      getConvoFiles: db.getConvoFiles,
      getFiles: db.getFiles,
      getUserKey: db.getUserKey,
      getMessages: db.getMessages,
      updateFilesUsage: db.updateFilesUsage,
      getUserKeyValues: db.getUserKeyValues,
      getUserCodeFiles: db.getUserCodeFiles,
      getToolFilesByIds: db.getToolFilesByIds,
      getCodeGeneratedFiles: db.getCodeGeneratedFiles,
      listSkillsByAccess: skillDbMethods.listSkillsByAccess,
      listAlwaysApplySkills: skillDbMethods.listAlwaysApplySkills,
      getSkillByName: skillDbMethods.getSkillByName,
    };

    const enabledCapabilities = new Set(agentsEConfig?.capabilities);
    const skillsCapabilityEnabled = enabledCapabilities.has(AgentCapabilities.skills);
    const ephemeralSkillsToggle = req.body?.ephemeralAgent?.skills === true;
    const accessibleSkillIds = skillsCapabilityEnabled
      ? withDeploymentSkillIds(
          await findAccessibleResources({
            userId: req.user.id,
            role: req.user.role,
            resourceType: ResourceType.SKILL,
            requiredPermissions: PermissionBits.VIEW,
          }),
        )
      : [];
    const editableSkillIds = skillsCapabilityEnabled
      ? await findAccessibleResources({
          userId: req.user.id,
          role: req.user.role,
          resourceType: ResourceType.SKILL,
          requiredPermissions: PermissionBits.EDIT,
        })
      : [];
    const skillCreateAllowed = skillsCapabilityEnabled
      ? await getSkillToolDeps().canCreateSkill({ req })
      : false;

    const { skillStates, defaultActiveOnShare } = await loadSkillStates({
      userId: req.user.id,
      appConfig,
      getUserById: db.getUserById,
      accessibleSkillIds,
    });

    const manualSkills = extractManualSkills(req.body);

    const primaryScopedSkillIds = resolveAgentScopedSkillIds({
      agent,
      accessibleSkillIds,
      skillsCapabilityEnabled,
      ephemeralSkillsToggle,
    });
    const primaryScopedEditableSkillIds = resolveAgentScopedSkillIds({
      agent,
      accessibleSkillIds: editableSkillIds,
      skillsCapabilityEnabled,
      ephemeralSkillsToggle,
    });

    const primaryConfig = await initializeAgent(
      {
        req,
        res,
        loadTools,
        requestFiles: [],
        conversationId,
        parentMessageId,
        agent,
        endpointOption,
        allowedProviders,
        isInitialAgent: true,
        accessibleSkillIds: primaryScopedSkillIds,
        skillAuthoringAvailable: canAuthorSkillFiles({
          agent,
          scopedEditableSkillIds: primaryScopedEditableSkillIds,
          skillCreateAllowed,
          skillsCapabilityEnabled,
          ephemeralSkillsToggle,
        }),
        codeEnvAvailable: enabledCapabilities.has(AgentCapabilities.execute_code),
        skillStates,
        defaultActiveOnShare,
        manualSkills,
      },
      dbMethods,
    );

    /**
     * Per-agent tool-execution context map, keyed by agentId. Lets the
     * ON_TOOL_EXECUTE callback route each (sub-)agent's tool calls to the
     * correct toolRegistry / userMCPAuthMap / tool_resources.
     * @type {Map<string, object>}
     */
    const agentToolContexts = new Map();
    agentToolContexts.set(
      primaryConfig.id,
      buildAgentToolContext({ agent, config: primaryConfig }),
    );

    let handoffAgentConfigs = new Map();
    let discoveredEdges = [];
    let discoveredMCPAuthMap;
    if (primaryConfig.edges?.length) {
      const modelsConfig = await getModelsConfig(req);
      ({
        agentConfigs: handoffAgentConfigs,
        edges: discoveredEdges,
        userMCPAuthMap: discoveredMCPAuthMap,
      } = await discoverConnectedAgents(
        {
          req,
          res,
          primaryConfig,
          endpointOption,
          allowedProviders,
          modelsConfig,
          loadTools,
          requestFiles: [],
          conversationId,
          parentMessageId,
          resourceType: ResourceType.REMOTE_AGENT,
          computeAccessibleSkillIds: (handoffAgent) =>
            resolveAgentScopedSkillIds({
              agent: handoffAgent,
              accessibleSkillIds,
              skillsCapabilityEnabled,
              ephemeralSkillsToggle,
            }),
          computeSkillAuthoringAvailable: (handoffAgent) =>
            canAuthorSkillFiles({
              agent: handoffAgent,
              scopedEditableSkillIds: resolveAgentScopedSkillIds({
                agent: handoffAgent,
                accessibleSkillIds: editableSkillIds,
                skillsCapabilityEnabled,
                ephemeralSkillsToggle,
              }),
              skillCreateAllowed,
              skillsCapabilityEnabled,
              ephemeralSkillsToggle,
            }),
          skillStates,
          defaultActiveOnShare,
          codeEnvAvailable: enabledCapabilities.has(AgentCapabilities.execute_code),
        },
        {
          getAgent: db.getAgent,
          checkPermission: async ({ userId, role, resourceId, requiredPermission }) => {
            const permissions = await getRemoteAgentPermissions(
              { getEffectivePermissions },
              userId,
              role,
              resourceId,
            );
            return hasPermissions(permissions, requiredPermission);
          },
          logViolation,
          db: dbMethods,
          onAgentInitialized: (agentId, handoffAgent, config) => {
            agentToolContexts.set(agentId, buildAgentToolContext({ agent: handoffAgent, config }));
          },
          initializeAgent,
        },
      ));
    }

    primaryConfig.edges = discoveredEdges;

    const aggregator = createEvaluationAggregator();

    const collectedUsage = [];
    /** @type {Promise<import('librechat-data-provider').TAttachment | null>[]} */
    const artifactPromises = [];

    const artifactCallback = createToolEndCallback({
      req,
      res,
      artifactPromises,
      streamId: null,
    });

    /**
     * Capture each tool's raw output for evaluation, then delegate to the
     * shared artifact callback so structured artifacts (chunks, citations,
     * files, search results) are still processed into attachments.
     * @type {import('@librechat/agents').ToolEndCallback}
     */
    const toolEndCallback = async (data, metadata) => {
      const output = data?.output;
      if (output?.tool_call_id) {
        const isError = output.is_error === true || output.status === 'error';
        aggregator.recordToolResult(
          output.tool_call_id,
          normalizeToolOutput(output.content),
          isError,
        );
      }
      await artifactCallback(data, metadata);
    };

    const toolExecuteOptions = {
      loadTools: async (toolNames, agentId) => {
        const ctx = agentToolContexts.get(agentId) ?? agentToolContexts.get(primaryConfig.id) ?? {};
        const result = await loadToolsForExecution({
          req,
          res,
          toolNames,
          agent: ctx.agent ?? agent,
          signal: abortController.signal,
          toolRegistry: ctx.toolRegistry,
          userMCPAuthMap: ctx.userMCPAuthMap,
          tool_resources: ctx.tool_resources,
          actionsEnabled: ctx.actionsEnabled,
        });
        return enrichLoadedToolsWithAgentContext({ result, req, ctx });
      },
      toolEndCallback,
      ...getSkillToolDeps(),
    };

    const summarizationConfig = appConfig?.summarization;

    const internalMessages = convertMessages(request.messages);

    const toolSet = buildToolSet(primaryConfig);
    const formatted = formatAgentMessages(internalMessages, {}, toolSet);
    const formattedMessages = formatted.messages;
    const initialSummary = formatted.summary;
    let indexTokenCountMap = formatted.indexTokenCountMap;

    const manualSkillPrimes = primaryConfig.manualSkillPrimes;
    const alwaysApplySkillPrimes = primaryConfig.alwaysApplySkillPrimes;
    if (
      (manualSkillPrimes && manualSkillPrimes.length > 0) ||
      (alwaysApplySkillPrimes && alwaysApplySkillPrimes.length > 0)
    ) {
      const primeResult = injectSkillPrimes({
        initialMessages: formattedMessages,
        indexTokenCountMap,
        manualSkillPrimes,
        alwaysApplySkillPrimes,
      });
      indexTokenCountMap = primeResult.indexTokenCountMap;
      if (primeResult.alwaysApplyDropped > 0) {
        logger.warn(
          `[Agent Eval] Dropped ${primeResult.alwaysApplyDropped} always-apply prime(s) to stay within MAX_PRIMED_SKILLS_PER_TURN.`,
        );
      }
    }

    /**
     * Event handlers — aggregate the full run for evaluation. No streaming.
     */
    const handlers = {
      on_message_delta: {
        handle: (_event, data) => {
          const content = data?.delta?.content;
          if (Array.isArray(content)) {
            for (const part of content) {
              if (part.type === 'text' && part.text) {
                aggregator.addText(part.text);
              }
            }
          }
        },
      },
      on_reasoning_delta: {
        handle: (_event, data) => {
          const content = data?.delta?.content;
          if (Array.isArray(content)) {
            for (const part of content) {
              const text = part.think || part.text;
              if (text) {
                aggregator.addReasoning(text);
              }
            }
          }
        },
      },
      on_run_step: {
        handle: (_event, data) => {
          const stepDetails = data?.stepDetails;
          if (stepDetails?.type === 'tool_calls' && stepDetails.tool_calls) {
            for (const tc of stepDetails.tool_calls) {
              aggregator.startToolCall(data.index ?? 0, tc.id ?? '', tc.name ?? '');
            }
          }
        },
      },
      on_run_step_delta: {
        handle: (_event, data) => {
          const delta = data?.delta;
          if (delta?.type === 'tool_calls' && delta.tool_calls) {
            for (const tc of delta.tool_calls) {
              const args = tc.args ?? '';
              if (args) {
                aggregator.appendToolArgs(tc.index ?? 0, args);
              }
            }
          }
        },
      },
      on_chat_model_end: {
        handle: (_event, data, metadata) => {
          const usage = data?.output?.usage_metadata;
          if (usage) {
            const taggedUsage = markSummarizationUsage(usage, metadata);
            collectedUsage.push(taggedUsage);
            aggregator.usage.promptTokens += taggedUsage.input_tokens ?? 0;
            aggregator.usage.completionTokens += taggedUsage.output_tokens ?? 0;
            aggregator.usage.reasoningTokens +=
              taggedUsage.output_token_details?.reasoning ??
              taggedUsage.output_token_details?.reasoning_tokens ??
              0;
          }
        },
      },
      on_run_step_completed: { handle: () => {} },
      on_tool_end: new ToolEndHandler(toolEndCallback, logger),
      on_chain_stream: { handle: () => {} },
      on_chain_end: { handle: () => {} },
      on_agent_update: { handle: () => {} },
      on_agent_log: agentLogHandlerObj,
      on_custom_event: { handle: () => {} },
      on_tool_execute: createToolExecuteHandler(toolExecuteOptions),
      ...(summarizationConfig?.enabled !== false
        ? buildSummarizationHandlers({ isStreaming: false, res })
        : {}),
    };

    const userId = req.user?.id ?? 'api-user';
    const userMCPAuthMap = discoveredMCPAuthMap ?? primaryConfig.userMCPAuthMap;
    const runAgents = [primaryConfig, ...handoffAgentConfigs.values()];

    const run = await createRun({
      agents: runAgents,
      messages: formattedMessages,
      indexTokenCountMap,
      initialSummary,
      runId: responseId,
      summarizationConfig,
      appConfig,
      signal: abortController.signal,
      customHandlers: handlers,
      requestBody: {
        messageId: responseId,
        conversationId,
      },
      user: { id: userId },
    });

    if (!run) {
      throw new Error('Failed to create agent run');
    }

    const config = {
      runName: 'AgentRun',
      configurable: {
        thread_id: conversationId,
        user_id: userId,
        user: createSafeUser(req.user),
        requestBody: {
          messageId: responseId,
          conversationId,
        },
        ...(userMCPAuthMap != null && { userMCPAuthMap }),
      },
      recursionLimit: resolveRecursionLimit(agentsEConfig, agent),
      signal: abortController.signal,
      streamMode: 'values',
      version: 'v2',
    };

    await run.processStream({ messages: formattedMessages }, config, {
      callbacks: {
        [Callback.TOOL_ERROR]: (graph, error, toolId) => {
          logger.error(`[Agent Eval] Tool Error "${toolId}"`, error);
        },
      },
    });

    // Record token usage against balance (parity with chat/openai paths)
    const balanceConfig = getBalanceConfig(appConfig);
    const transactionsConfig = getTransactionsConfig(appConfig);
    recordCollectedUsage(
      {
        spendTokens: db.spendTokens,
        spendStructuredTokens: db.spendStructuredTokens,
        pricing: { getMultiplier: db.getMultiplier, getCacheMultiplier: db.getCacheMultiplier },
        bulkWriteOps: { insertMany: db.bulkInsertTransactions, updateBalance: db.updateBalance },
      },
      {
        user: userId,
        conversationId,
        collectedUsage,
        context: 'message',
        messageId: responseId,
        balance: balanceConfig,
        transactions: transactionsConfig,
        model: primaryConfig.model || agent.model_parameters?.model,
      },
    ).catch((err) => {
      logger.error('[Agent Eval] Error recording usage:', err);
    });

    // Wait for structured tool artifacts before building the response
    let resolvedArtifacts = [];
    if (artifactPromises.length > 0) {
      try {
        resolvedArtifacts = await Promise.all(artifactPromises);
      } catch (artifactError) {
        logger.warn('[Agent Eval] Error processing artifacts:', artifactError);
      }
    }

    const usage = {
      prompt_tokens: aggregator.usage.promptTokens,
      completion_tokens: aggregator.usage.completionTokens,
      total_tokens: aggregator.usage.promptTokens + aggregator.usage.completionTokens,
    };
    if (aggregator.usage.reasoningTokens > 0) {
      usage.completion_tokens_details = {
        reasoning_tokens: aggregator.usage.reasoningTokens,
      };
    }

    const artifactsByToolCall = groupArtifactsByToolCall(resolvedArtifacts);
    const response = buildEvaluationResponse(context, aggregator, artifactsByToolCall, usage);

    res.json(response);
    logger.debug(
      `[Agent Eval] Evaluation ${responseId} completed in ${Date.now() - requestStartTime}ms`,
    );
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'An error occurred';
    logger.error('[Agent Eval] Error:', error);

    if (res.headersSent) {
      res.end();
      return;
    }

    const statusCode =
      typeof error?.status === 'number' && error.status >= 400 && error.status < 600
        ? error.status
        : 500;
    const errorType =
      statusCode >= 400 && statusCode < 500 ? 'invalid_request_error' : 'server_error';
    sendErrorResponse(res, statusCode, errorMessage, errorType);
  }
};

module.exports = {
  AgentEvaluationController,
};
