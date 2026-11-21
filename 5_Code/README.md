# AgentSQL-Guard with InvaCard – demo

One file, one command:

```
python agentsql_guard_demo.py
```

* Needs Python 3.9 or newer. On the first run the script installs the packages it needs
  (duckdb, numpy, pandas, scikit-learn, scipy, sqlglot, sqlalchemy, matplotlib) with pip.
* No database server is needed: the TPC-H data is generated in memory with DuckDB.
* The full run takes about 5–7 minutes on a laptop. For a fast run (about 2 minutes) use
  `python agentsql_guard_demo.py --quick`.
* Results are printed step by step in the terminal and saved in the `demo_output` folder:
  an HTML report with all tables and charts (opened automatically in the browser),
  six PNG charts, and the CSV data files (measurements, sub-plan cardinalities, equivalence checks).

Useful options

| Option | Meaning |
|---|---|
| `--quick` | 2 parameter draws, 1 timed run per plan (fast demo) |
| `--sf 0.2` | larger database (slower, clearer timings) |
| `--draws 4` | parameter draws per question (default 4 → 240 queries) |
| `--dbgen` | use DuckDB's official TPC-H generator instead of the built-in one (needs internet once) |
| `--no-browser` | do not open the report automatically |
| `--out folder` | where to save the results |

What the 11 steps show

1. Build the TPC-H database
2. Write 12 business questions five ways: expert SQL, ORM (SQLAlchemy) SQL, three LLM-agent wordings
3. Check every wording returns the same answer (main + two randomly mutated databases)
4. Syntactic and semantic query fingerprints, with unit/property tests
5. Measure real latency, estimated vs true rows per plan step, and alternative plans (independent labels)
6. Plan of the worst agent wording next to the expert plan
7. Wording-dependent estimation error and the coherence certificate (linear program, checked for soundness)
8. InvaCard vs baselines (native, AQO-style syntactic memory, semantic memory, learned residual) + ablations
9. Stage-one detectors M1–M5 (EXPLAIN-only features), seen-question and new-question protocols
10. Factorial fault injection (wording × plan × contention × resources), Shapley cause shares, leakage test
11. Mitigation policies replayed on measured runtimes, decision curve

The demo uses DuckDB so that it runs anywhere without installation. The full study described in the
project paper runs the same pipeline on PostgreSQL 16 with pg_hint_plan (Rows-hint injection, cold-cache
and lock factors), which an embedded engine cannot reproduce.
