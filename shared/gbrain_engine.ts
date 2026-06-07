/**
 * shared/gbrain_engine.ts — the standard GBrain engine bootstrap, in one place.
 *
 * Introduced by W3.5.96: policy_eval.ts / route_eval.ts / query_policy.ts (and bench_strategies.ts)
 * each repeat this identical ~12-line connect sequence. Later GBrain-based chapters import
 * bootstrapEngine() instead of inlining it — and the hardcoded gbrain source path lives here only.
 *
 *   import { bootstrapEngine } from "/Users/yuxinliu/code/agent-prep/shared/gbrain_engine.ts";
 *   const { engine, runEval } = await bootstrapEngine();
 *
 * GBRAIN_SRC env overrides the gbrain source path (default below).
 *
 * NOTE: a chapter that is *teaching* the bootstrap should still inline it (so the reader sees the
 * sequence); import this only when the bootstrap is plumbing, not the lesson.
 */
const GB = process.env.GBRAIN_SRC ?? "/Users/yuxinliu/code/agent-prep/gbrain/src";

export interface Bootstrapped {
  engine: any;        // GBrain engine (searchKeyword / searchVector / getPage / listPages / …)
  runEval: any;       // core/search/eval.ts runEval — the keyword/vector/hybrid harness
  config: any;        // loaded GBrain config
}

/** Connect to GBrain exactly as the CLI does: load config → create engine → connect → wire gateway. */
export async function bootstrapEngine(): Promise<Bootstrapped> {
  const { loadConfig, toEngineConfig } = await import(`${GB}/core/config.ts`);
  const { createEngine } = await import(`${GB}/core/engine-factory.ts`);
  const { connectWithRetry } = await import(`${GB}/core/db.ts`);
  const { configureGateway, reconfigureGatewayWithEngine } = await import(`${GB}/core/ai/gateway.ts`);
  const { buildGatewayConfig } = await import(`${GB}/core/ai/build-gateway-config.ts`);
  const { runEval } = await import(`${GB}/core/search/eval.ts`);

  const config = loadConfig();
  configureGateway(buildGatewayConfig(config));
  const engine = await createEngine(toEngineConfig(config));
  await connectWithRetry(engine, toEngineConfig(config), { noRetry: true });
  await reconfigureGatewayWithEngine(engine);
  return { engine, runEval, config };
}
