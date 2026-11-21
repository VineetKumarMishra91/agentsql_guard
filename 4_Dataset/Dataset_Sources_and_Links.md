# Dataset sources and download links

| Dataset | Type | Role in the study | Link |
|---|---|---|---|
| TPC-H | Synthetic database + 22 query templates, uniform data | Main schema (scale factors 1 and 3); expert arm and business questions for ORM and agent arms | https://www.tpc.org/tpch/ (official); generator kit: https://github.com/gregrahn/tpch-kit |
| DSB | Synthetic, skewed and correlated data based on TPC-DS | Second schema; harder test for estimators | https://github.com/microsoft/dsb |
| IMDB + Join Order Benchmark (JOB) | Real movie database, 113 join queries | Estimator comparison on real data | https://github.com/gregrahn/join-order-benchmark |
| CEB (Cardinality Estimation Benchmark) | 16K+ IMDB queries with true sub-plan row counts | Extra labelled sub-plans; unseen-template tests | https://learnedsystems.mit.edu/cardinality-estimation-benchmark |
| STATS / STATS-CEB | Real Stack Exchange data and workload | Optional generalisation check | https://arxiv.org/abs/2109.05877 |
| SQLStorm (base paper) | 18K+ LLM-generated queries | Realism check for agent SQL | https://www.vldb.org/pvldb/vol18/p4144-schmidt.pdf |
| BIRD | Text-to-SQL benchmark | Question styles and prompt examples only | https://bird-bench.github.io/ |
| Spider | Text-to-SQL benchmark | Question styles and prompt examples only | https://yale-lily.github.io/spider |
| AgentSQL-FactorBench v0.1 (ours) | Matched expert / ORM / agent workload | Training and testing | This folder (`workload/`) |
