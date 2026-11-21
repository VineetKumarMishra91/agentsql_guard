# AgentSQL-Guard with InvaCard

**A formulation-invariant cardinality correction model and causal evaluation framework for LLM-agent-generated SQL**

Foundation of Data Science – Group 13, VIT Vellore
Umar (23BCE0079) · Sohailur (23BCE0042) · Vineet Kumar Mishra (23BCE0953)
Faculty: Kalyanaraman P

---

## What this project is about

LLM agents now write SQL that runs directly on production databases. Most work checks whether that SQL is *correct*; we study what happens inside the database when a correct agent query runs. Before running a query, the optimizer estimates how many rows each step will produce (the **cardinality estimate**). When that estimate is badly wrong, the optimizer picks a slow plan. Agents phrase the same question differently every time, while existing estimators and correction tools learn from the exact wording of past queries – so what they learn never carries over.

* **InvaCard** corrects the optimizer's row estimates using the *meaning* of each sub-expression instead of its wording: a semantic experience memory, a gated learned residual (split-conformal band per formulation path), and a projection onto coherence laws so that corrected estimates stay logically consistent.
* **SQL-Guard** wraps it in two stages: stage one predicts risk before execution using only SQL text and EXPLAIN; stage two diagnoses after execution. Every claim is judged against **independent, measured labels**, and slowdowns are split into causes with Shapley values from controlled fault injection.

## Repository contents

| Folder | Contents |
|---|---|
| [`1_Paper/`](1_Paper) | **Final report** (PDF and Word): methodology, implementation, results and outcomes. Also the earlier project paper. |
| [`2_Base_Paper/`](2_Base_Paper) | Base paper: *SQLStorm: Taking Database Benchmarking into the LLM Era*, Schmidt, Leis, Boncz, Neumann, PVLDB 18, 2025 (open access, CC BY-NC-ND 4.0). |
| [`3_Reference_Papers/`](3_Reference_Papers) | Full reference list with links to every cited paper. |
| [`4_Dataset/`](4_Dataset) | AgentSQL-FactorBench v0.1: the 240-query matched workload (expert / ORM / LLM-agent wordings) with equivalence checks, TPC-H sample tables (SF 0.01), schema files and links to all larger datasets. |
| [`5_Code/`](5_Code) | `agentsql_guard_demo.py` – the complete pipeline in one file, plus `requirements.txt` and run instructions. |
| [`6_Results/`](6_Results) | Outputs of the full run reported in the final report: HTML report, console output, CSV data (measurements, sub-plan cardinalities, equivalence checks) and figures. |

## How to run

```bash
cd 5_Code
python -m pip install -r requirements.txt
python agentsql_guard_demo.py --quick     # about 2 minutes
python agentsql_guard_demo.py             # full run, about 5–7 minutes
```

No database server is needed: the TPC-H data is generated in memory with DuckDB. Results are printed step by step and saved as an HTML report with charts in `demo_output/`. See [`5_Code/README.md`](5_Code/README.md) for all options.

## Pipeline (11 steps)

1. Build the TPC-H database in memory
2. Write 12 business questions five ways: expert SQL, ORM (SQLAlchemy) SQL, three LLM-agent wordings
3. Verify every wording returns the same rows (main + two randomly mutated databases)
4. Syntactic and semantic query fingerprints (sqlglot) with unit/property tests
5. Measure real latency, estimated vs true rows per plan step, and alternative plans → independent labels
6. Plan of the worst agent wording next to the expert plan
7. Wording-dependent estimation error and the coherence certificate (linear program, checked for soundness)
8. InvaCard vs baselines (native, AQO-style syntactic memory, semantic memory, learned residual) + ablations
9. Stage-one detectors M1–M5 using EXPLAIN-only features (seen-question and new-question protocols)
10. Factorial fault injection (wording × plan × contention × resources), Shapley cause shares, leakage test
11. Mitigation policies replayed on measured runtimes, decision curve

## Key results (full run, TPC-H SF 0.1, 240 queries)

| Result | Value |
|---|---|
| Verified-equivalent queries | 239 of 240 (one agent wording rejected on mutated data) |
| Regression rate (≥2× slower than the best equivalent wording) | agent 11.2 % · expert 0 % · ORM 0 % |
| Worst agent wording | 75.5× slower than the fastest equivalent, same answer |
| Share of agent slowdowns caused by wording (Shapley) | 77 % (plan choice 5 %, contention 13 %, resources 6 %) |
| Same sub-expression estimated ≥2× differently across wordings | 12 of 528 (true counts identical) |
| InvaCard q-error on unseen agent wordings, 90th percentile | 29.6 → 9.6 |
| InvaCard q-error, 99th percentile | 26,189 → 4,733 |
| Key reuse on agent sub-expressions: syntactic vs semantic | 0 % vs 55.5 % |
| Coherence-law violations: native → learned residual alone → InvaCard | 12 → 128 → 0 |
| Stage-one detection AUPRC on completely new questions (base rate 0.067) | 0.52 |
| Coherence certificate | fired on 36 of 239 plans, sound on all |
| Leakage invariance test | pass |
| Rewrite-to-verified-equivalent policy | prevented 16 of 16 regressions, 0 unnecessary interventions |

Limitations (detailed in the final report): the demo uses DuckDB, so corrected estimates cannot be injected into the planner and cold-cache, lock and stale-statistics factors are not tested; InvaCard is weaker than the native estimator on join, semi-join, CTE and grouping steps of new questions; agent wordings come from a single LLM. The full study runs the same pipeline on PostgreSQL 16 with pg_hint_plan.
