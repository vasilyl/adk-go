**Please ensure you have read the [contribution guide](./CONTRIBUTING.md) before creating a pull request.**

### Link to Issue or Description of Change

**1. Link to an existing issue (if applicable):**

- Closes: N/A
- Related: https://github.com/google/adk-go/issues/738

**2. Or, if no issue exists, describe the change:**

**Problem:**
During complex multi-agent executions or nested tool invocations, it is difficult to trace the origin of a tool (e.g., whether it is a local function, an MCP tool, a sub-agent, or a transfer agent). This lack of provenance makes debugging and analytics on tool usage patterns highly challenging.

**Solution:**
Introduce a `getToolOrigin(t tool.Tool) string` helper inside `bigquery_agent_analytics_plugin.go` to inspect the tool's type and return its origin classification (`LOCAL`, `SUB_AGENT`, `MCP`, `A2A`, `TRANSFER_AGENT`, or `UNKNOWN`). Use this to populate a `tool_origin` attribute in telemetry logs for `TOOL_START`, `TOOL_END`, and `TOOL_ERROR` event types.

### Testing Plan

Added a unit test `TestToolProvenance` to verify that `getToolOrigin` correctly identifies `LOCAL` function tools and `SUB_AGENT` agent tools.

### Checklist

- [x] I have read the [CONTRIBUTING.md](./CONTRIBUTING.md) document.
- [x] I have performed a self-review of my own code.
- [x] I have commented my code, particularly in hard-to-understand areas.
- [x] I have added tests that prove my fix is effective or that my feature works.
- [x] New and existing unit tests pass locally with my changes.
- [x] I have manually tested my changes end-to-end.
