**Please ensure you have read the [contribution guide](./CONTRIBUTING.md) before creating a pull request.**

### Link to Issue or Description of Change

**1. Link to an existing issue (if applicable):**

- Closes: N/A

**2. Or, if no issue exists, describe the change:**

**Problem:**
The BigQuery event log table consolidates diverse event types into a single schema where content payloads are serialized as raw JSON. Querying this raw table directly for specific event types is tedious and requires writing repetitive JSON extraction SQL queries.

**Solution:**
Add a configuration setting `CreateViews` (default `true`). When enabled, the plugin automatically and asynchronously creates simplified, typed BigQuery Views for each standard event type (e.g., `v_user_message`, `v_model_request`, `v_tool_call`). These views automatically extract JSON structures into clean, typed top-level columns (e.g., extracting prompts or tokens counts) for easy analytical querying.

### Testing Plan

Added unit tests validating view creation queries are formulated correctly and executed asynchronously without blocking the main thread initialization.

### Checklist

- [x] I have read the [CONTRIBUTING.md](./CONTRIBUTING.md) document.
- [x] I have performed a self-review of my own code.
- [x] I have commented my code, particularly in hard-to-understand areas.
- [x] I have added tests that prove my fix is effective or that my feature works.
- [x] New and existing unit tests pass locally with my changes.
- [x] I have manually tested my changes end-to-end.
