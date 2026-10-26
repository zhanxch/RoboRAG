// A comparison group is deliberately stricter than a task name.
export const groupKey = r => JSON.stringify([r.task, r.condition, r.protocol, r.model, r.effort, r.split, r.transport, r.bank_version, r.code_version, r.mcp_server, r.mcp_version]);
export function totalTokens(r) {
  return r.input_tokens != null && r.output_tokens != null ? r.input_tokens + r.output_tokens : r.total_tokens;
}
export function summarize(runs) {
  const completed = runs.filter(r => r.status === 'completed');
  const scored = completed.filter(r => typeof r.success === 'boolean');
  const progress = completed.filter(r => r.completion != null);
  const times = completed.filter(r => r.wall_seconds != null);
  const known = runs.filter(r => totalTokens(r) != null);
  return {count: runs.length, completed: completed.length, scored: scored.length,
    successes: scored.filter(r => r.success).length,
    successRate: scored.length ? scored.filter(r => r.success).length / scored.length * 100 : null,
    completion: progress.length ? progress.reduce((a,r) => a + r.completion, 0) / progress.length : null,
    completionN: progress.length,
    wall: times.length ? times.reduce((a,r) => a + r.wall_seconds, 0) / times.length : null, wallN: times.length,
    tokens: known.length ? known.reduce((a,r) => a + totalTokens(r), 0) : null, tokenN: known.length,
    lowerBound: known.length !== runs.length || runs.some(r => r.usage !== 'complete')};
}
