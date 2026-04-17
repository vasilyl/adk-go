**Please ensure you have read the [contribution guide](./CONTRIBUTING.md) before creating a pull request.**

### Link to Issue or Description of Change

**1. Link to an existing issue (if applicable):**

- Closes: N/A

**2. Or, if no issue exists, describe the change:**

**Problem:**
As the Go ADK analytics plugin updates its schema over time, users with pre-existing BigQuery tables will experience insertion errors due to missing columns, unless they manually migrate or recreate their tables.

**Solution:**
Implement automatic schema upgrades on initialization. During `NewBigQueryAgentAnalyticsPluginWithClients`, the plugin retrieves the existing table metadata, compares its schema against the latest `EventsSchema()`, and automatically updates the BigQuery table schema to add any missing fields or record sub-fields safely.

### Testing Plan

* `TestSchemaFieldsMatch` validates that the schema comparison algorithm correctly identifies top-level and nested structural differences.
* `TestNewBigQueryAgentAnalyticsPlugin_TableExists` verifies that existing tables are inspected and dynamically updated without throwing insertion errors.

### Checklist

- [x] I have read the [CONTRIBUTING.md](./CONTRIBUTING.md) document.
- [x] I have performed a self-review of my own code.
- [x] I have commented my code, particularly in hard-to-understand areas.
- [x] I have added tests that prove my fix is effective or that my feature works.
- [x] New and existing unit tests pass locally with my changes.
- [x] I have manually tested my changes end-to-end.
