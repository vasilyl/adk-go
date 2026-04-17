**Please ensure you have read the [contribution guide](./CONTRIBUTING.md) before creating a pull request.**

### Link to Issue or Description of Change

**1. Link to an existing issue (if applicable):**

- Closes: N/A
- Related: https://github.com/google/adk-go/issues/738

**2. Or, if no issue exists, describe the change:**

**Problem:**
Logging very large prompts, tool outputs, or media payloads (e.g., high-res images) directly in-line inside a relational BigQuery row is inefficient, expensive, and hit strict row limits. 

**Solution:**
Introduce a GCS-backed asynchronous offloading strategy inside `bigquery_agent_analytics_plugin.go` via `gcsOffloader` and `hybridContentParser`.
1. When the logged content (text or media) exceeds the configured `inlineTextLimit`, or is binary content, upload it to GCS using the `gcsOffloader`.
2. Record the object storage path as a `GCS_REFERENCE` in BigQuery, and store structured metadata under `object_ref` (including the GCS `uri`, bucket details, and an optional `authorizer` connection ID).
3. Fall back to standard smart truncation if the offloader is not configured or if the upload fails.

### Testing Plan

Added a robust unit test `TestHybridContentParser` with a mock httptest TLS GCS server verifying that:
1. Text payloads below the threshold are successfully kept inline and summarized.
2. Large text and binary media payloads are intercepted, correctly uploaded to the mock GCS bucket, and substituted in the database record with a GCS URI reference containing `object_ref` details.
3. If the GCS offloader is missing or fails, the parser gracefully falls back to local truncation.

### Checklist

- [x] I have read the [CONTRIBUTING.md](./CONTRIBUTING.md) document.
- [x] I have performed a self-review of my own code.
- [x] I have commented my code, particularly in hard-to-understand areas.
- [x] I have added tests that prove my fix is effective or that my feature works.
- [x] New and existing unit tests pass locally with my changes.
- [x] I have manually tested my changes end-to-end.
