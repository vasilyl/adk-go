**Please ensure you have read the [contribution guide](./CONTRIBUTING.md) before creating a pull request.**

### Link to Issue or Description of Change

**1. Link to an existing issue (if applicable):**

- Closes: N/A
- Related: https://github.com/google/adk-go/issues/738

**2. Or, if no issue exists, describe the change:**

**Problem:**
In concurrent or asynchronous agent execution flows, standard flat context logs lack parent-child span structure, making it incredibly difficult to trace nested execution layers (e.g., which agent invocation triggered which tool or LLM call).

**Solution:**
Implement a thread-safe in-memory contextual trace stack (`spanStack` and `traceManager`) inside `bigquery_agent_analytics_plugin.go`. Push nested execution span contexts (`spanRecord`) onto the invocation's trace stack at execution boundaries (Runs, Models, Tools), linking children to parents via `parentSpanID`. Retrieve trace context when logging events, and ensure safe cleanup upon run completion.

### Testing Plan

Added a complete integration unit test `TestContextualTraceStacking` which triggers `BeforeRunCallback`, `BeforeModelCallback`, `AfterModelCallback`, and `AfterRunCallback` sequentially, verifying that:
1. A root span is correctly pushed.
2. Nested spans are linked to the root span as `parentSpanID`.
3. Stack sizes update correctly as spans are popped.
4. The invocation's trace stack is properly cleaned up and removed on execution end.

### Checklist

- [x] I have read the [CONTRIBUTING.md](./CONTRIBUTING.md) document.
- [x] I have performed a self-review of my own code.
- [x] I have commented my code, particularly in hard-to-understand areas.
- [x] I have added tests that prove my fix is effective or that my feature works.
- [x] New and existing unit tests pass locally with my changes.
- [x] I have manually tested my changes end-to-end.
