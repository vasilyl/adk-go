**Please ensure you have read the [contribution guide](./CONTRIBUTING.md) before creating a pull request.**

### Link to Issue or Description of Change

**1. Link to an existing issue (if applicable):**

- Closes: N/A

**2. Or, if no issue exists, describe the change:**

**Problem:**
Standard event logging only records basic string/json summaries of agent execution. Rich interactions like human-in-the-loop (HITL) approvals (credentials, confirmations, input), state delta modifications, and agent-to-agent interactions are not captured with structure, making analytical queries difficult.

**Solution:**
Refactor `OnEventCallback` in the analytics plugin to detect and parse structured event types: `STATE_DELTA` updates, `HITL` request events, and Agent-to-Agent (A2A) interaction statistics, serializing them into clear attributes and content blocks.

### Testing Plan

Unit tests verify structured schemas are built properly inside `bigquery_agent_analytics_plugin_test.go` when invoking these callbacks.

### Checklist

- [x] I have read the [CONTRIBUTING.md](./CONTRIBUTING.md) document.
- [x] I have performed a self-review of my own code.
- [x] I have commented my code, particularly in hard-to-understand areas.
- [x] I have added tests that prove my fix is effective or that my feature works.
- [x] New and existing unit tests pass locally with my changes.
- [x] I have manually tested my changes end-to-end.
