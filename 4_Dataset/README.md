# 4_Dataset – AgentSQL-FactorBench (release 0.1) and data sources

Project: AgentSQL-Guard with InvaCard (Foundation of Data Science, Group 13)

## What is in this folder

| Path | What it holds |
|---|---|
| `workload/agentsql_workload_v0.1.csv` | 240 query instances: 12 TPC-H intents (Q2, Q3, Q4, Q5, Q7, Q10, Q11, Q16, Q17, Q18, Q20, Q21) x 4 parameter draws x 5 formulations (1 expert TPC-H query, 1 SQLAlchemy Core ORM query, 3 LLM-agent formulations). |
| `workload/equivalence_checks_v0.1.csv` | Result of checking every ORM and agent query against the expert query on four databases. |
| `tpch_sample_sf0.01/*.csv` | The eight TPC-H tables at scale factor 0.01 (about 87,000 rows), generated with the official `dbgen` tool. CSV with header row. |
| `schema/tpch_create_tables.sql` | Official TPC-H table definitions (from `dss.ddl` in the TPC-H kit). |
| `schema/tpch_keys_and_indexes.sql` | Primary keys, foreign keys and the fixed set of secondary indexes used in the study. |
| `Dataset_Sources_and_Links.md` | Download links for all larger datasets used in the paper. |

## Columns of `agentsql_workload_v0.1.csv`

| Column | Meaning |
|---|---|
| query_id | `<intent>_d<draw>_<formulation>`, for example `Q21_d0_agent3` |
| intent | TPC-H query number whose business question is used |
| parameter_draw | 0 to 3; all five formulations of one draw share the same parameter values |
| arm | `expert`, `orm` or `agent` |
| formulation | `expert`, `orm`, `agent1`, `agent2`, `agent3` |
| author | TPC-H specification, SQLAlchemy Core, or the LLM that wrote the query |
| question | The plain-English question given for the intent with its parameters filled in |
| parameters_json | Parameter values drawn following the TPC-H specification |
| sql | The SQL text (PostgreSQL dialect) |
| equivalence_verdict | `reference` (expert query), `equivalent` or `unverified_timeout` |

## Equivalence check

Each ORM and agent query was run next to the expert query on (1) TPC-H at scale factor 0.01, (2) and (3) two randomly altered copies of that database (rows deleted, dates and quantities shifted, comments changed), and (4) TPC-H at scale factor 0.3. Results were compared as multisets after rounding numbers to four decimals.

* ORM: 48 of 48 equivalent on all four databases.
* Agent: 140 of 144 equivalent on all four databases; 4 (all `Q21_d*_agent3`, a correlated-subquery wording) did not finish within the 30 s / 60 s limits even on the smallest database and are marked `unverified_timeout`.

## How to load the sample into PostgreSQL

```sql
CREATE DATABASE tpch_s0;
\c tpch_s0
\i schema/tpch_create_tables.sql
-- for each table, e.g.
\copy region   FROM 'tpch_sample_sf0.01/region.csv'   WITH CSV HEADER
\copy nation   FROM 'tpch_sample_sf0.01/nation.csv'   WITH CSV HEADER
\copy supplier FROM 'tpch_sample_sf0.01/supplier.csv' WITH CSV HEADER
\copy customer FROM 'tpch_sample_sf0.01/customer.csv' WITH CSV HEADER
\copy part     FROM 'tpch_sample_sf0.01/part.csv'     WITH CSV HEADER
\copy partsupp FROM 'tpch_sample_sf0.01/partsupp.csv' WITH CSV HEADER
\copy orders   FROM 'tpch_sample_sf0.01/orders.csv'   WITH CSV HEADER
\copy lineitem FROM 'tpch_sample_sf0.01/lineitem.csv' WITH CSV HEADER
\i schema/tpch_keys_and_indexes.sql
ANALYZE;
```

Larger scale factors (0.3, 1 and 3) are generated with `dbgen -s <SF>` from the TPC-H kit linked in `Dataset_Sources_and_Links.md`.
