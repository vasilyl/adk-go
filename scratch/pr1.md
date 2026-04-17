**Please ensure you have read the [contribution guide](./CONTRIBUTING.md) before creating a pull request.**

### Link to Issue or Description of Change

**1. Link to an existing issue (if applicable):**

- Closes: N/A
- Related: https://github.com/google/adk-go/issues/738

**2. Or, if no issue exists, describe the change:**

**Problem:**
Logging large volumes of events into a flat BigQuery table can lead to inefficient query scanning, resulting in slower analytical queries and higher query costs over time.

**Solution:**
Configure daily time partitioning during table creation inside `NewBigQueryAgentAnalyticsPluginWithClients`. The BigQuery table is created with `TimePartitioning` enabled on the `timestamp` field using `bq.DayPartitioningType`.

### Testing Plan

Added unit test `TestNewBigQueryAgentAnalyticsPlugin_CreateTable_WithPartitioning` to verify that table creation requests correctly specify the partitioning parameters (`timePartitioning` with type `DAY` and field `timestamp`).

### Checklist

- [x] I have read the [CONTRIBUTING.md](./CONTRIBUTING.md) document.
- [x] I have performed a self-review of my own code.
- [x] I have commented my code, particularly in hard-to-understand areas.
- [x] I have added tests that prove my fix is effective or that my feature works.
- [x] New and existing unit tests pass locally with my changes.
- [x] I have manually tested my changes end-to-end.
