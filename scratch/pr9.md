**Please ensure you have read the [contribution guide](./CONTRIBUTING.md) before creating a pull request.**

### Link to Issue or Description of Change

**1. Link to an existing issue (if applicable):**

- Closes: N/A
- Related: https://github.com/google/adk-go/issues/738

**2. Or, if no issue exists, describe the change:**

**Problem:**
When offloading large text or media payloads to Google Cloud Storage (GCS), analytical queries on the logged telemetry database need a secure, unified way to link the external GCS references back to BigQuery. BigQuery Object Tables allow querying external GCS objects securely, but require an explicit BigQuery connection ID.

**Solution:**
Introduce a new configuration parameter `ConnectionID string` to the plugin `Config` struct (in `config.go`), and initialize it to an empty string by default in `DefaultConfig()`. This provides a clean, standard location to pass the BigQuery external connection resource path, preparing the analytics plugin to generate secure metadata references (`authorizer` field in external object refs).

### Testing Plan

Verified that the `ConnectionID` field is successfully defined on the `Config` struct, and that it defaults to an empty string when calling `DefaultConfig()` in testing.

### Checklist

- [x] I have read the [CONTRIBUTING.md](./CONTRIBUTING.md) document.
- [x] I have performed a self-review of my own code.
- [x] I have commented my code, particularly in hard-to-understand areas.
- [x] I have added tests that prove my fix is effective or that my feature works.
- [x] New and existing unit tests pass locally with my changes.
- [x] I have manually tested my changes end-to-end.
