**Please ensure you have read the [contribution guide](./CONTRIBUTING.md) before creating a pull request.**

### Link to Issue or Description of Change

**1. Link to an existing issue (if applicable):**

- Closes: N/A

**2. Or, if no issue exists, describe the change:**

**Problem:**
In high-traffic production systems, logging every single event (such as every internal tool call or raw LLM trace) to BigQuery can result in massive amounts of log data, leading to excessive storage and ingestion costs.

**Solution:**
Introduce `EventAllowlist` and `EventDenylist` arrays in the configuration. When logging an event, the plugin checks these lists first. If an event is explicitly denied or not explicitly allowed (when an allowlist is defined), the event is ignored and not pushed to the batch processor.

### Testing Plan

Added `TestEventFiltering` which validates that events are allowed or denied correctly according to the configure filters, preventing unauthorized events from being pushed to the stream.

### Checklist

- [x] I have read the [CONTRIBUTING.md](./CONTRIBUTING.md) document.
- [x] I have performed a self-review of my own code.
- [x] I have commented my code, particularly in hard-to-understand areas.
- [x] I have added tests that prove my fix is effective or that my feature works.
- [x] New and existing unit tests pass locally with my changes.
- [x] I have manually tested my changes end-to-end.
