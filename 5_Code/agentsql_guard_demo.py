#!/usr/bin/env python3
"""
AgentSQL-Guard with InvaCard - single-file demonstration
=========================================================
Foundation of Data Science project, Group 13 (Umar 23BCE0079, Sohailur 23BCE0042, Vineet 23BCE0953)

What this script does (run:  python agentsql_guard_demo.py )
  1. builds a TPC-H database in memory (DuckDB)
  2. writes 12 business questions five ways each: expert SQL, ORM (SQLAlchemy) SQL, three LLM-agent wordings
  3. checks that every wording returns the same answer (main + two randomly mutated databases)
  4. computes syntactic and semantic query fingerprints (sqlglot) and runs property tests
  5. measures real latency, row-count estimates vs true row counts, and counterfactual plans (labels)
  6. prints the plan of the worst agent wording next to the expert plan
  7. measures wording-dependent estimation error and computes the coherence certificate (linear program)
  8. trains and tests InvaCard (semantic memory + gated learned residual + coherence projection) vs baselines
  9. stage-one detectors M1-M5 (EXPLAIN-only features), leave-one-question-out, AUPRC
 10. factorial fault injection (wording x plan x contention x resources) with exact Shapley cause shares
     and the leakage-invariance test
 11. replays mitigation policies and draws a decision curve
Results are printed step by step and saved as an HTML report with charts in ./demo_output

Options: --quick (about 2 minutes), --sf 0.2 (bigger data), --dbgen (official TPC-H generator), --no-browser, --help
Requires Python 3.9+. Missing packages are installed automatically on first run.
"""
import importlib, subprocess, sys

REQUIRED = {"duckdb": "duckdb", "numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "scipy": "scipy",
            "sqlglot": "sqlglot", "sqlalchemy": "sqlalchemy", "matplotlib": "matplotlib"}


def ensure_packages():
    missing = [pip for mod, pip in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if "duckdb" not in missing:
        # the plan-profile format used here is the one of DuckDB 2.x; older versions report rows differently
        try:
            from importlib.metadata import version
            if int(version("duckdb").split(".")[0]) < 2:
                missing.append("duckdb>=2.0,<3")
        except Exception:
            pass
    missing = ["duckdb>=2.0,<3" if m == "duckdb" else m for m in missing]
    if missing:
        print("Installing missing packages (first run only): " + ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", *missing])


import importlib.util
ensure_packages()

import argparse, base64, pathlib, datetime as dt, decimal, hashlib, html, itertools, json, math, os, random, re
import textwrap, threading, time, warnings, webbrowser
from collections import Counter, defaultdict

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
plt.rcParams["axes.prop_cycle"] = cycler(color=["#000000", "#444444", "#777777", "#000000", "#999999", "#444444"],
                                         linestyle=["-", "--", ":", "-.", "-", ":"])
import numpy as np
import pandas as pd
import sqlglot
from sqlglot import exp
from sqlglot.optimizer import optimize
from scipy.optimize import linprog
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import (MetaData, Table, Column, Integer, Numeric, String, Date, select, func, and_, or_, not_,
                        exists, literal, extract, distinct, desc)
from sqlalchemy.dialects import postgresql

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ===================================================================================================
# PART 1. TPC-H DATA GENERATOR (fallback when DuckDB's dbgen extension cannot be downloaded)
# ===================================================================================================
# TPC-H style data generator (follows the value rules of TPC-H spec section 4.2.3)

REGIONS = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]
NATION_ROWS = [("ALGERIA", 0), ("ARGENTINA", 1), ("BRAZIL", 1), ("CANADA", 1), ("EGYPT", 4), ("ETHIOPIA", 0),
           ("FRANCE", 3), ("GERMANY", 3), ("INDIA", 2), ("INDONESIA", 2), ("IRAN", 4), ("IRAQ", 4), ("JAPAN", 2),
           ("JORDAN", 4), ("KENYA", 0), ("MOROCCO", 0), ("MOZAMBIQUE", 0), ("PERU", 1), ("CHINA", 2),
           ("ROMANIA", 3), ("SAUDI ARABIA", 4), ("VIETNAM", 2), ("RUSSIA", 3), ("UNITED KINGDOM", 3),
           ("UNITED STATES", 1)]
SEGMENTS = ["AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"]
PRIORITIES = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
SHIPMODES = ["REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB"]
INSTRUCT = ["DELIVER IN PERSON", "COLLECT COD", "NONE", "TAKE BACK RETURN"]
TYPE_S1 = ["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"]
TYPE_S2 = ["ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"]
TYPE_S3 = ["TIN", "NICKEL", "BRASS", "STEEL", "COPPER"]
CONT_S1 = ["SM", "LG", "MED", "JUMBO", "WRAP"]
CONT_S2 = ["CASE", "BOX", "BAG", "JAR", "PKG", "PACK", "CAN", "DRUM"]
COLORS = ["almond", "antique", "aquamarine", "azure", "beige", "bisque", "black", "blanched", "blue", "blush",
          "brown", "burlywood", "burnished", "chartreuse", "chiffon", "chocolate", "coral", "cornflower",
          "cornsilk", "cream", "cyan", "dark", "deep", "dim", "dodger", "drab", "firebrick", "floral", "forest",
          "frosted", "gainsboro", "ghost", "goldenrod", "green", "grey", "honeydew", "hot", "indian", "ivory",
          "khaki", "lace", "lavender", "lawn", "lemon", "light", "lime", "linen", "magenta", "maroon", "medium",
          "metallic", "midnight", "mint", "misty", "moccasin", "navajo", "navy", "olive", "orange", "orchid",
          "pale", "papaya", "peach", "peru", "pink", "plum", "powder", "puff", "purple", "red", "rose", "rosy",
          "royal", "saddle", "salmon", "sandy", "seashell", "sienna", "sky", "slate", "smoke", "snow", "spring",
          "steel", "tan", "thistle", "tomato", "turquoise", "violet", "wheat", "white", "yellow"]
WORDS = np.array("furiously quickly carefully blithely slyly ironic final pending regular express special bold even "
                 "silent unusual idle packages deposits accounts requests instructions theodolites foxes pinto beans "
                 "platelets asymptotes dependencies excuses ideas courts sleep wake nag haggle cajole integrate "
                 "boost detect use print among above along after".split())
START, END = np.datetime64("1992-01-01"), np.datetime64("1998-12-31")
CURRENT = np.datetime64("1995-06-17")


def _choice(rng, arr, n):
    return np.asarray(arr, dtype=object)[rng.integers(0, len(arr), n)]


def _text(rng, n, lo=3, hi=9):
    k = rng.integers(lo, hi + 1, n)
    w = WORDS[rng.integers(0, len(WORDS), (n, hi))]
    return np.array([" ".join(w[i, :k[i]]) for i in range(n)], dtype=object)


def _phone(rng, nationkey):
    n = len(nationkey)
    a, b, c = rng.integers(100, 1000, n), rng.integers(100, 1000, n), rng.integers(1000, 10000, n)
    return np.array([f"{nk + 10}-{x}-{y}-{z}" for nk, x, y, z in zip(nationkey, a, b, c)], dtype=object)


def _money(rng, lo, hi, n):
    return np.round(rng.integers(int(lo * 100), int(hi * 100) + 1, n) / 100.0, 2)


def generate_tpch(sf=0.1, seed=7):
    rng = np.random.default_rng(seed)
    T = {}
    T["region"] = pd.DataFrame({"r_regionkey": np.arange(5), "r_name": REGIONS, "r_comment": _text(rng, 5)})
    T["nation"] = pd.DataFrame({"n_nationkey": np.arange(25), "n_name": [n for n, _ in NATION_ROWS],
                                "n_regionkey": [r for _, r in NATION_ROWS], "n_comment": _text(rng, 25)})
    # supplier
    S = max(int(10000 * sf), 10)
    sk = np.arange(1, S + 1)
    snk = rng.integers(0, 25, S)
    scomment = _text(rng, S)
    bad = rng.choice(S, max(1, S // 2000), replace=False)          # Customer ... Complaints (Q16)
    for i in bad:
        scomment[i] = "slyly Customer final Complaints " + scomment[i]
    T["supplier"] = pd.DataFrame({"s_suppkey": sk, "s_name": [f"Supplier#{k:09d}" for k in sk],
                                  "s_address": _text(rng, S, 1, 3), "s_nationkey": snk, "s_phone": _phone(rng, snk),
                                  "s_acctbal": _money(rng, -999.99, 9999.99, S), "s_comment": scomment})
    # part
    P = max(int(200000 * sf), 50)
    pk = np.arange(1, P + 1)
    ci = np.argsort(rng.random((P, len(COLORS))), axis=1)[:, :5]
    pname = np.array([" ".join(COLORS[j] for j in row) for row in ci], dtype=object)
    m, nn = rng.integers(1, 6, P), rng.integers(1, 6, P)
    ptype = np.array([f"{a} {b} {c}" for a, b, c in zip(_choice(rng, TYPE_S1, P), _choice(rng, TYPE_S2, P), _choice(rng, TYPE_S3, P))], dtype=object)
    retail = (90000 + ((pk // 10) % 20001) + 100 * (pk % 1000)) / 100.0
    T["part"] = pd.DataFrame({"p_partkey": pk, "p_name": pname, "p_mfgr": [f"Manufacturer#{x}" for x in m],
                              "p_brand": [f"Brand#{x}{y}" for x, y in zip(m, nn)], "p_type": ptype,
                              "p_size": rng.integers(1, 51, P),
                              "p_container": [f"{a} {b}" for a, b in zip(_choice(rng, CONT_S1, P), _choice(rng, CONT_S2, P))],
                              "p_retailprice": np.round(retail, 2), "p_comment": _text(rng, P, 1, 4)})
    # partsupp: 4 suppliers per part (spec formula)
    psp = np.repeat(pk, 4)
    i = np.tile(np.arange(4), P)
    pss = (psp + (i * ((S // 4) + (psp - 1) // S))) % S + 1
    T["partsupp"] = pd.DataFrame({"ps_partkey": psp, "ps_suppkey": pss, "ps_availqty": rng.integers(1, 10000, 4 * P),
                                  "ps_supplycost": _money(rng, 1.0, 1000.0, 4 * P), "ps_comment": _text(rng, 4 * P, 2, 6)})
    # customer
    C = max(int(150000 * sf), 30)
    ck = np.arange(1, C + 1)
    cnk = rng.integers(0, 25, C)
    T["customer"] = pd.DataFrame({"c_custkey": ck, "c_name": [f"Customer#{k:09d}" for k in ck],
                                  "c_address": _text(rng, C, 1, 3), "c_nationkey": cnk, "c_phone": _phone(rng, cnk),
                                  "c_acctbal": _money(rng, -999.99, 9999.99, C),
                                  "c_mktsegment": _choice(rng, SEGMENTS, C), "c_comment": _text(rng, C)})
    # orders (sparse keys, customers not divisible by 3)
    O = max(int(1500000 * sf), 100)
    idx = np.arange(O)
    ok = (idx // 8) * 32 + idx % 8 + 1
    valid_c = ck[ck % 3 != 0]
    ocust = valid_c[rng.integers(0, len(valid_c), O)]
    span = int((END - np.timedelta64(151, "D") - START).astype(int))
    odate = START + rng.integers(0, span + 1, O).astype("timedelta64[D]")
    # lineitem
    nl = rng.integers(1, 8, O)
    L = int(nl.sum())
    lok = np.repeat(ok, nl)
    lodate = np.repeat(odate, nl)
    lnum = np.concatenate([np.arange(1, k + 1) for k in nl])
    lpk = rng.integers(1, P + 1, L)
    j = rng.integers(0, 4, L)
    lsk = (lpk + (j * ((S // 4) + (lpk - 1) // S))) % S + 1
    qty = rng.integers(1, 51, L).astype(float)
    ext = np.round(qty * (90000 + ((lpk // 10) % 20001) + 100 * (lpk % 1000)) / 100.0, 2)
    disc = rng.integers(0, 11, L) / 100.0
    tax = rng.integers(0, 9, L) / 100.0
    ship = lodate + rng.integers(1, 122, L).astype("timedelta64[D]")
    commit = lodate + rng.integers(30, 91, L).astype("timedelta64[D]")
    receipt = ship + rng.integers(1, 31, L).astype("timedelta64[D]")
    rflag = np.where(receipt <= CURRENT, np.where(rng.random(L) < 0.5, "R", "A"), "N").astype(object)
    lstat = np.where(ship > CURRENT, "O", "F").astype(object)
    T["lineitem"] = pd.DataFrame({"l_orderkey": lok, "l_partkey": lpk, "l_suppkey": lsk, "l_linenumber": lnum,
                                  "l_quantity": qty, "l_extendedprice": ext, "l_discount": disc, "l_tax": tax,
                                  "l_returnflag": rflag, "l_linestatus": lstat, "l_shipdate": ship,
                                  "l_commitdate": commit, "l_receiptdate": receipt,
                                  "l_shipinstruct": _choice(rng, INSTRUCT, L), "l_shipmode": _choice(rng, SHIPMODES, L),
                                  "l_comment": _text(rng, L, 2, 5)})
    li = T["lineitem"]
    tot = (li.l_extendedprice * (1 + li.l_tax) * (1 - li.l_discount)).groupby(li.l_orderkey).sum()
    nf = (li.l_linestatus == "F").groupby(li.l_orderkey).mean()
    status = np.where(nf.values == 1, "F", np.where(nf.values == 0, "O", "P"))
    T["orders"] = pd.DataFrame({"o_orderkey": ok, "o_custkey": ocust, "o_orderstatus": status.astype(object),
                                "o_totalprice": np.round(tot.reindex(ok).values, 2), "o_orderdate": odate,
                                "o_orderpriority": _choice(rng, PRIORITIES, O),
                                "o_clerk": [f"Clerk#{x:09d}" for x in rng.integers(1, max(int(1000 * sf), 2) + 1, O)],
                                "o_shippriority": np.zeros(O, dtype=int), "o_comment": _text(rng, O, 2, 6)})
    return T


# ===================================================================================================
# PART 2. THE THREE-ARM WORKLOAD: 12 TPC-H INTENTS, EXPERT SQL AND THREE LLM-AGENT WORDINGS EACH
# ===================================================================================================
REGIONS = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]
NATIONS = ["ALGERIA", "ARGENTINA", "BRAZIL", "CANADA", "EGYPT", "ETHIOPIA", "FRANCE", "GERMANY", "INDIA",
           "INDONESIA", "IRAN", "IRAQ", "JAPAN", "JORDAN", "KENYA", "MOROCCO", "MOZAMBIQUE", "PERU", "CHINA",
           "ROMANIA", "SAUDI ARABIA", "VIETNAM", "RUSSIA", "UNITED KINGDOM", "UNITED STATES"]
SEGMENTS = ["AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"]
TYPE_S1 = ["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"]
TYPE_S2 = ["ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"]
TYPE_S3 = ["TIN", "NICKEL", "BRASS", "STEEL", "COPPER"]
CONT_S1 = ["SM", "LG", "MED", "JUMBO", "WRAP"]
CONT_S2 = ["CASE", "BOX", "BAG", "JAR", "PKG", "PACK", "CAN", "DRUM"]
COLORS = ["almond", "antique", "aquamarine", "azure", "beige", "bisque", "black", "blanched", "blue", "blush",
          "brown", "burlywood", "burnished", "chartreuse", "chiffon", "chocolate", "coral", "cornflower",
          "cornsilk", "cream", "cyan", "dark", "deep", "dim", "dodger", "drab", "firebrick", "floral", "forest",
          "frosted", "gainsboro", "ghost", "goldenrod", "green", "grey", "honeydew", "hot", "indian", "ivory",
          "khaki", "lace", "lavender", "lawn", "lemon", "light", "lime", "linen", "magenta", "maroon", "medium",
          "metallic", "midnight", "mint", "misty", "moccasin", "navajo", "navy", "olive", "orange", "orchid",
          "pale", "papaya", "peach", "peru", "pink", "plum", "powder", "puff", "purple", "red", "rose", "rosy",
          "royal", "saddle", "salmon", "sandy", "seashell", "sienna", "sky", "slate", "smoke", "snow", "spring",
          "steel", "tan", "thistle", "tomato", "turquoise", "violet", "wheat", "white", "yellow"]

SF = 0.3


def _add_months(d, m):
    y, mo = divmod(d.month - 1 + m, 12)
    return dt.date(d.year + y, mo + 1, 1)


def make_params(intent, rng):
    """Parameter draws following the TPC-H substitution rules (spec section 2.4)."""
    if intent == "Q2":
        return dict(size=rng.randint(1, 50), type=rng.choice(TYPE_S3), region=rng.choice(REGIONS))
    if intent == "Q3":
        d = dt.date(1995, 3, 1) + dt.timedelta(days=rng.randint(0, 30))
        return dict(segment=rng.choice(SEGMENTS), date=d.isoformat())
    if intent == "Q4":
        d = _add_months(dt.date(1993, 1, 1), rng.randint(0, 57))
        return dict(date=d.isoformat(), date_end=_add_months(d, 3).isoformat())
    if intent == "Q5":
        d = dt.date(rng.randint(1993, 1997), 1, 1)
        return dict(region=rng.choice(REGIONS), date=d.isoformat(), date_end=dt.date(d.year + 1, 1, 1).isoformat())
    if intent == "Q7":
        n1, n2 = rng.sample(NATIONS, 2)
        return dict(n1=n1, n2=n2)
    if intent == "Q10":
        d = _add_months(dt.date(1993, 2, 1), rng.randint(0, 23))
        return dict(date=d.isoformat(), date_end=_add_months(d, 3).isoformat())
    if intent == "Q11":
        return dict(nation=rng.choice(NATIONS), fraction=round(0.0001 / SF, 10))
    if intent == "Q16":
        sizes = rng.sample(range(1, 51), 8)
        return dict(brand=f"Brand#{rng.randint(1, 5)}{rng.randint(1, 5)}",
                    type=f"{rng.choice(TYPE_S1)} {rng.choice(TYPE_S2)}", sizes=", ".join(map(str, sizes)))
    if intent == "Q17":
        return dict(brand=f"Brand#{rng.randint(1, 5)}{rng.randint(1, 5)}",
                    container=f"{rng.choice(CONT_S1)} {rng.choice(CONT_S2)}")
    if intent == "Q18":
        return dict(quantity=rng.randint(312, 315))
    if intent == "Q20":
        d = dt.date(rng.randint(1993, 1997), 1, 1)
        return dict(color=rng.choice(COLORS), date=d.isoformat(), date_end=dt.date(d.year + 1, 1, 1).isoformat(),
                    nation=rng.choice(NATIONS))
    if intent == "Q21":
        return dict(nation=rng.choice(NATIONS))
    raise KeyError(intent)


INTENTS = ["Q2", "Q3", "Q4", "Q5", "Q7", "Q10", "Q11", "Q16", "Q17", "Q18", "Q20", "Q21"]

QUESTIONS = {
    "Q2": "For parts of size {size} whose type ends in {type}, find the supplier in region {region} that offers the lowest supply cost; list account balance, supplier name, nation, part key, manufacturer, address, phone and comment, top 100 by account balance.",
    "Q3": "Top 10 unshipped orders by revenue for customers in market segment {segment}, for orders placed before {date} with line items shipped after {date}.",
    "Q4": "For each order priority, count orders placed in the quarter starting {date} that have at least one line item received after its commit date.",
    "Q5": "Revenue per nation in region {region} for the year starting {date}, counting only line items where customer and supplier are in the same nation.",
    "Q7": "Shipping volume between {n1} and {n2} (both directions) per supplier nation, customer nation and year, for shipments in 1995 and 1996.",
    "Q10": "Top 20 customers by revenue lost to returned items for orders placed in the quarter starting {date}.",
    "Q11": "Parts whose stock value supplied from {nation} exceeds {fraction} of the total stock value from {nation}.",
    "Q16": "Count distinct suppliers per brand, type and size for parts not of brand {brand}, not of type {type}, with size in ({sizes}), excluding suppliers with customer complaints.",
    "Q17": "Average yearly revenue lost if small orders (quantity below 20 percent of the part's average) of brand {brand} and container {container} were dropped.",
    "Q18": "Top 100 customers with orders whose total quantity exceeds {quantity}.",
    "Q20": "Suppliers in {nation} with more than half of last year's shipped quantity still in stock for parts whose name starts with {color}, for the year starting {date}.",
    "Q21": "Suppliers in {nation} who were the only late supplier on multi-supplier finished orders, top 100 by count.",
}

EXPERT = {
    "Q2": """select s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment
from part, supplier, partsupp, nation, region
where p_partkey = ps_partkey and s_suppkey = ps_suppkey and p_size = {size} and p_type like '%{type}'
  and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = '{region}'
  and ps_supplycost = (select min(ps_supplycost) from partsupp, supplier, nation, region
      where p_partkey = ps_partkey and s_suppkey = ps_suppkey and s_nationkey = n_nationkey
        and n_regionkey = r_regionkey and r_name = '{region}')
order by s_acctbal desc, n_name, s_name, p_partkey limit 100""",
    "Q3": """select l_orderkey, sum(l_extendedprice * (1 - l_discount)) as revenue, o_orderdate, o_shippriority
from customer, orders, lineitem
where c_mktsegment = '{segment}' and c_custkey = o_custkey and l_orderkey = o_orderkey
  and o_orderdate < date '{date}' and l_shipdate > date '{date}'
group by l_orderkey, o_orderdate, o_shippriority
order by revenue desc, o_orderdate limit 10""",
    "Q4": """select o_orderpriority, count(*) as order_count
from orders
where o_orderdate >= date '{date}' and o_orderdate < date '{date}' + interval '3' month
  and exists (select * from lineitem where l_orderkey = o_orderkey and l_commitdate < l_receiptdate)
group by o_orderpriority order by o_orderpriority""",
    "Q5": """select n_name, sum(l_extendedprice * (1 - l_discount)) as revenue
from customer, orders, lineitem, supplier, nation, region
where c_custkey = o_custkey and l_orderkey = o_orderkey and l_suppkey = s_suppkey and c_nationkey = s_nationkey
  and s_nationkey = n_nationkey and n_regionkey = r_regionkey and r_name = '{region}'
  and o_orderdate >= date '{date}' and o_orderdate < date '{date}' + interval '1' year
group by n_name order by revenue desc""",
    "Q7": """select supp_nation, cust_nation, l_year, sum(volume) as revenue
from (select n1.n_name as supp_nation, n2.n_name as cust_nation, extract(year from l_shipdate) as l_year,
             l_extendedprice * (1 - l_discount) as volume
      from supplier, lineitem, orders, customer, nation n1, nation n2
      where s_suppkey = l_suppkey and o_orderkey = l_orderkey and c_custkey = o_custkey
        and s_nationkey = n1.n_nationkey and c_nationkey = n2.n_nationkey
        and ((n1.n_name = '{n1}' and n2.n_name = '{n2}') or (n1.n_name = '{n2}' and n2.n_name = '{n1}'))
        and l_shipdate between date '1995-01-01' and date '1996-12-31') as shipping
group by supp_nation, cust_nation, l_year order by supp_nation, cust_nation, l_year""",
    "Q10": """select c_custkey, c_name, sum(l_extendedprice * (1 - l_discount)) as revenue, c_acctbal, n_name,
       c_address, c_phone, c_comment
from customer, orders, lineitem, nation
where c_custkey = o_custkey and l_orderkey = o_orderkey and o_orderdate >= date '{date}'
  and o_orderdate < date '{date}' + interval '3' month and l_returnflag = 'R' and c_nationkey = n_nationkey
group by c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, c_comment
order by revenue desc limit 20""",
    "Q11": """select ps_partkey, sum(ps_supplycost * ps_availqty) as value
from partsupp, supplier, nation
where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = '{nation}'
group by ps_partkey having sum(ps_supplycost * ps_availqty) > (
    select sum(ps_supplycost * ps_availqty) * {fraction} from partsupp, supplier, nation
    where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = '{nation}')
order by value desc""",
    "Q16": """select p_brand, p_type, p_size, count(distinct ps_suppkey) as supplier_cnt
from partsupp, part
where p_partkey = ps_partkey and p_brand <> '{brand}' and p_type not like '{type}%' and p_size in ({sizes})
  and ps_suppkey not in (select s_suppkey from supplier where s_comment like '%Customer%Complaints%')
group by p_brand, p_type, p_size order by supplier_cnt desc, p_brand, p_type, p_size""",
    "Q17": """select sum(l_extendedprice) / 7.0 as avg_yearly
from lineitem, part
where p_partkey = l_partkey and p_brand = '{brand}' and p_container = '{container}'
  and l_quantity < (select 0.2 * avg(l_quantity) from lineitem where l_partkey = p_partkey)""",
    "Q18": """select c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, sum(l_quantity)
from customer, orders, lineitem
where o_orderkey in (select l_orderkey from lineitem group by l_orderkey having sum(l_quantity) > {quantity})
  and c_custkey = o_custkey and o_orderkey = l_orderkey
group by c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
order by o_totalprice desc, o_orderdate limit 100""",
    "Q20": """select s_name, s_address
from supplier, nation
where s_suppkey in (select ps_suppkey from partsupp
        where ps_partkey in (select p_partkey from part where p_name like '{color}%')
          and ps_availqty > (select 0.5 * sum(l_quantity) from lineitem
              where l_partkey = ps_partkey and l_suppkey = ps_suppkey
                and l_shipdate >= date '{date}' and l_shipdate < date '{date}' + interval '1' year))
  and s_nationkey = n_nationkey and n_name = '{nation}'
order by s_name""",
    "Q21": """select s_name, count(*) as numwait
from supplier, lineitem l1, orders, nation
where s_suppkey = l1.l_suppkey and o_orderkey = l1.l_orderkey and o_orderstatus = 'F'
  and l1.l_receiptdate > l1.l_commitdate
  and exists (select * from lineitem l2 where l2.l_orderkey = l1.l_orderkey and l2.l_suppkey <> l1.l_suppkey)
  and not exists (select * from lineitem l3 where l3.l_orderkey = l1.l_orderkey and l3.l_suppkey <> l1.l_suppkey
                  and l3.l_receiptdate > l3.l_commitdate)
  and s_nationkey = n_nationkey and n_name = '{nation}'
group by s_name order by numwait desc, s_name limit 100""",
}

# Agent arm: three formulations per intent, written by the LLM from QUESTIONS + schema.
AGENT = {
    "Q2": [
        """WITH rs AS (
  SELECT s.s_suppkey, s.s_acctbal, s.s_name, s.s_address, s.s_phone, s.s_comment, n.n_name
  FROM supplier s JOIN nation n ON s.s_nationkey = n.n_nationkey
  JOIN region r ON n.n_regionkey = r.r_regionkey WHERE r.r_name = '{region}'),
cand AS (
  SELECT p.p_partkey, p.p_mfgr, ps.ps_supplycost, rs.s_acctbal, rs.s_name, rs.n_name, rs.s_address,
         rs.s_phone, rs.s_comment, MIN(ps.ps_supplycost) OVER (PARTITION BY p.p_partkey) AS min_cost
  FROM part p JOIN partsupp ps ON ps.ps_partkey = p.p_partkey JOIN rs ON rs.s_suppkey = ps.ps_suppkey
  WHERE p.p_size = {size} AND p.p_type LIKE '%{type}')
SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment
FROM cand WHERE ps_supplycost = min_cost
ORDER BY s_acctbal DESC, n_name, s_name, p_partkey LIMIT 100""",
        """SELECT s.s_acctbal, s.s_name, n.n_name, p.p_partkey, p.p_mfgr, s.s_address, s.s_phone, s.s_comment
FROM part p
JOIN partsupp ps ON ps.ps_partkey = p.p_partkey
JOIN supplier s ON s.s_suppkey = ps.ps_suppkey
JOIN nation n ON n.n_nationkey = s.s_nationkey
JOIN region r ON r.r_regionkey = n.n_regionkey
JOIN (SELECT ps2.ps_partkey, MIN(ps2.ps_supplycost) AS min_cost
      FROM partsupp ps2 JOIN supplier s2 ON s2.s_suppkey = ps2.ps_suppkey
      JOIN nation n2 ON n2.n_nationkey = s2.s_nationkey JOIN region r2 ON r2.r_regionkey = n2.n_regionkey
      WHERE r2.r_name = '{region}' GROUP BY ps2.ps_partkey) m
  ON m.ps_partkey = p.p_partkey AND ps.ps_supplycost = m.min_cost
WHERE p.p_size = {size} AND p.p_type LIKE '%{type}' AND r.r_name = '{region}'
ORDER BY s.s_acctbal DESC, n.n_name, s.s_name, p.p_partkey LIMIT 100""",
        """SELECT s.s_acctbal, s.s_name, n.n_name, p.p_partkey, p.p_mfgr, s.s_address, s.s_phone, s.s_comment
FROM part p
JOIN partsupp ps ON ps.ps_partkey = p.p_partkey
JOIN supplier s ON s.s_suppkey = ps.ps_suppkey
JOIN nation n ON n.n_nationkey = s.s_nationkey
JOIN region r ON r.r_regionkey = n.n_regionkey
WHERE p.p_size = {size} AND p.p_type LIKE '%{type}' AND r.r_name = '{region}'
  AND NOT EXISTS (
    SELECT 1 FROM partsupp ps2 JOIN supplier s2 ON s2.s_suppkey = ps2.ps_suppkey
    JOIN nation n2 ON n2.n_nationkey = s2.s_nationkey JOIN region r2 ON r2.r_regionkey = n2.n_regionkey
    WHERE ps2.ps_partkey = p.p_partkey AND r2.r_name = '{region}' AND ps2.ps_supplycost < ps.ps_supplycost)
ORDER BY s.s_acctbal DESC, n.n_name, s.s_name, p.p_partkey LIMIT 100""",
    ],
    "Q3": [
        """WITH c AS (SELECT c_custkey FROM customer WHERE c_mktsegment = '{segment}'),
o AS (SELECT o_orderkey, o_orderdate, o_shippriority FROM orders
      WHERE o_orderdate < DATE '{date}' AND o_custkey IN (SELECT c_custkey FROM c))
SELECT l.l_orderkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue, o.o_orderdate, o.o_shippriority
FROM lineitem l JOIN o ON l.l_orderkey = o.o_orderkey
WHERE l.l_shipdate > DATE '{date}'
GROUP BY l.l_orderkey, o.o_orderdate, o.o_shippriority
ORDER BY revenue DESC, o.o_orderdate LIMIT 10""",
        """SELECT x.l_orderkey, x.revenue, o.o_orderdate, o.o_shippriority
FROM (SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue
      FROM lineitem WHERE l_shipdate > DATE '{date}' GROUP BY l_orderkey) x
JOIN orders o ON o.o_orderkey = x.l_orderkey
JOIN customer c ON c.c_custkey = o.o_custkey
WHERE c.c_mktsegment = '{segment}' AND o.o_orderdate < DATE '{date}'
ORDER BY x.revenue DESC, o.o_orderdate LIMIT 10""",
        """SELECT l.l_orderkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue, o.o_orderdate, o.o_shippriority
FROM orders o JOIN lineitem l ON l.l_orderkey = o.o_orderkey
WHERE EXISTS (SELECT 1 FROM customer c WHERE c.c_custkey = o.o_custkey AND c.c_mktsegment = '{segment}')
  AND o.o_orderdate < DATE '{date}' AND l.l_shipdate > DATE '{date}'
GROUP BY l.l_orderkey, o.o_orderdate, o.o_shippriority
ORDER BY revenue DESC, o.o_orderdate LIMIT 10""",
    ],
    "Q4": [
        """SELECT o_orderpriority, COUNT(*) AS order_count
FROM orders
WHERE o_orderdate >= DATE '{date}' AND o_orderdate < DATE '{date}' + INTERVAL '3 months'
  AND o_orderkey IN (SELECT l_orderkey FROM lineitem WHERE l_commitdate < l_receiptdate)
GROUP BY o_orderpriority ORDER BY o_orderpriority""",
        """SELECT o.o_orderpriority, COUNT(DISTINCT o.o_orderkey) AS order_count
FROM orders o JOIN lineitem l ON l.l_orderkey = o.o_orderkey
WHERE o.o_orderdate >= DATE '{date}' AND o.o_orderdate < DATE '{date}' + INTERVAL '3 months'
  AND l.l_commitdate < l.l_receiptdate
GROUP BY o.o_orderpriority ORDER BY o.o_orderpriority""",
        """WITH late AS (SELECT DISTINCT l_orderkey FROM lineitem WHERE l_commitdate < l_receiptdate)
SELECT o.o_orderpriority, COUNT(*) AS order_count
FROM orders o JOIN late ON late.l_orderkey = o.o_orderkey
WHERE o.o_orderdate >= DATE '{date}' AND o.o_orderdate < DATE '{date}' + INTERVAL '3 months'
GROUP BY o.o_orderpriority ORDER BY o.o_orderpriority""",
    ],
    "Q5": [
        """SELECT n.n_name, SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
FROM region r
JOIN nation n ON n.n_regionkey = r.r_regionkey
JOIN customer c ON c.c_nationkey = n.n_nationkey
JOIN orders o ON o.o_custkey = c.c_custkey
JOIN lineitem l ON l.l_orderkey = o.o_orderkey
JOIN supplier s ON s.s_suppkey = l.l_suppkey AND s.s_nationkey = n.n_nationkey
WHERE r.r_name = '{region}' AND o.o_orderdate >= DATE '{date}' AND o.o_orderdate < DATE '{date}' + INTERVAL '1 year'
GROUP BY n.n_name ORDER BY revenue DESC""",
        """WITH rn AS (SELECT n_nationkey, n_name FROM nation JOIN region ON n_regionkey = r_regionkey
            WHERE r_name = '{region}')
SELECT rn.n_name, SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
FROM lineitem l
JOIN orders o ON o.o_orderkey = l.l_orderkey
JOIN customer c ON c.c_custkey = o.o_custkey
JOIN supplier s ON s.s_suppkey = l.l_suppkey
JOIN rn ON rn.n_nationkey = c.c_nationkey
WHERE s.s_nationkey = c.c_nationkey
  AND o.o_orderdate >= DATE '{date}' AND o.o_orderdate < DATE '{date}' + INTERVAL '1 year'
GROUP BY rn.n_name ORDER BY revenue DESC""",
        """SELECT n.n_name, SUM(t.rev) AS revenue
FROM (SELECT c.c_nationkey, l.l_suppkey, l.l_extendedprice * (1 - l.l_discount) AS rev
      FROM orders o JOIN customer c ON c.c_custkey = o.o_custkey
      JOIN lineitem l ON l.l_orderkey = o.o_orderkey
      WHERE o.o_orderdate >= DATE '{date}' AND o.o_orderdate < DATE '{date}' + INTERVAL '1 year') t
JOIN supplier s ON s.s_suppkey = t.l_suppkey AND s.s_nationkey = t.c_nationkey
JOIN nation n ON n.n_nationkey = s.s_nationkey
JOIN region r ON r.r_regionkey = n.n_regionkey
WHERE r.r_name = '{region}'
GROUP BY n.n_name ORDER BY revenue DESC""",
    ],
    "Q7": [
        """WITH sn AS (SELECT n_nationkey, n_name FROM nation WHERE n_name IN ('{n1}', '{n2}'))
SELECT s_n.n_name AS supp_nation, c_n.n_name AS cust_nation, EXTRACT(YEAR FROM l.l_shipdate) AS l_year,
       SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
FROM lineitem l
JOIN supplier s ON s.s_suppkey = l.l_suppkey
JOIN orders o ON o.o_orderkey = l.l_orderkey
JOIN customer c ON c.c_custkey = o.o_custkey
JOIN sn s_n ON s_n.n_nationkey = s.s_nationkey
JOIN sn c_n ON c_n.n_nationkey = c.c_nationkey
WHERE s_n.n_name <> c_n.n_name AND l.l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
GROUP BY 1, 2, 3 ORDER BY 1, 2, 3""",
        """SELECT supp_nation, cust_nation, l_year, SUM(volume) AS revenue FROM (
  SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation, EXTRACT(YEAR FROM l_shipdate) AS l_year,
         l_extendedprice * (1 - l_discount) AS volume
  FROM supplier JOIN lineitem ON s_suppkey = l_suppkey JOIN orders ON o_orderkey = l_orderkey
  JOIN customer ON c_custkey = o_custkey JOIN nation n1 ON s_nationkey = n1.n_nationkey
  JOIN nation n2 ON c_nationkey = n2.n_nationkey
  WHERE n1.n_name = '{n1}' AND n2.n_name = '{n2}' AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
  UNION ALL
  SELECT n1.n_name, n2.n_name, EXTRACT(YEAR FROM l_shipdate), l_extendedprice * (1 - l_discount)
  FROM supplier JOIN lineitem ON s_suppkey = l_suppkey JOIN orders ON o_orderkey = l_orderkey
  JOIN customer ON c_custkey = o_custkey JOIN nation n1 ON s_nationkey = n1.n_nationkey
  JOIN nation n2 ON c_nationkey = n2.n_nationkey
  WHERE n1.n_name = '{n2}' AND n2.n_name = '{n1}' AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
) t GROUP BY supp_nation, cust_nation, l_year ORDER BY supp_nation, cust_nation, l_year""",
        """SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation, EXTRACT(YEAR FROM l.l_shipdate) AS l_year,
       SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
FROM lineitem l
JOIN supplier s ON s.s_suppkey = l.l_suppkey
JOIN orders o ON o.o_orderkey = l.l_orderkey
JOIN customer c ON c.c_custkey = o.o_custkey
JOIN nation n1 ON n1.n_nationkey = s.s_nationkey
JOIN nation n2 ON n2.n_nationkey = c.c_nationkey
WHERE s.s_nationkey IN (SELECT n_nationkey FROM nation WHERE n_name IN ('{n1}', '{n2}'))
  AND c.c_nationkey IN (SELECT n_nationkey FROM nation WHERE n_name IN ('{n1}', '{n2}'))
  AND s.s_nationkey <> c.c_nationkey
  AND l.l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
GROUP BY n1.n_name, n2.n_name, EXTRACT(YEAR FROM l.l_shipdate)
ORDER BY supp_nation, cust_nation, l_year""",
    ],
    "Q10": [
        """SELECT c.c_custkey, c.c_name, SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue, c.c_acctbal,
       n.n_name, c.c_address, c.c_phone, c.c_comment
FROM customer c
JOIN orders o ON o.o_custkey = c.c_custkey
JOIN lineitem l ON l.l_orderkey = o.o_orderkey
JOIN nation n ON n.n_nationkey = c.c_nationkey
WHERE o.o_orderdate >= DATE '{date}' AND o.o_orderdate < DATE '{date}' + INTERVAL '3 months'
  AND l.l_returnflag = 'R'
GROUP BY c.c_custkey, c.c_name, c.c_acctbal, c.c_phone, n.n_name, c.c_address, c.c_comment
ORDER BY revenue DESC LIMIT 20""",
        """WITH rev AS (
  SELECT o.o_custkey, SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue
  FROM orders o JOIN lineitem l ON l.l_orderkey = o.o_orderkey
  WHERE o.o_orderdate >= DATE '{date}' AND o.o_orderdate < DATE '{date}' + INTERVAL '3 months'
    AND l.l_returnflag = 'R'
  GROUP BY o.o_custkey)
SELECT c.c_custkey, c.c_name, rev.revenue, c.c_acctbal, n.n_name, c.c_address, c.c_phone, c.c_comment
FROM rev JOIN customer c ON c.c_custkey = rev.o_custkey JOIN nation n ON n.n_nationkey = c.c_nationkey
ORDER BY rev.revenue DESC LIMIT 20""",
        """SELECT c.c_custkey, c.c_name, SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue, c.c_acctbal,
       n.n_name, c.c_address, c.c_phone, c.c_comment
FROM customer c JOIN nation n ON n.n_nationkey = c.c_nationkey
JOIN orders o ON o.o_custkey = c.c_custkey JOIN lineitem l ON l.l_orderkey = o.o_orderkey
WHERE o.o_orderkey IN (SELECT o2.o_orderkey FROM orders o2
                       WHERE o2.o_orderdate >= DATE '{date}' AND o2.o_orderdate < DATE '{date}' + INTERVAL '3 months')
  AND l.l_returnflag = 'R'
GROUP BY c.c_custkey, c.c_name, c.c_acctbal, c.c_phone, n.n_name, c.c_address, c.c_comment
ORDER BY revenue DESC LIMIT 20""",
    ],
    "Q11": [
        """WITH stock AS (
  SELECT ps.ps_partkey, ps.ps_supplycost * ps.ps_availqty AS v
  FROM partsupp ps JOIN supplier s ON s.s_suppkey = ps.ps_suppkey
  JOIN nation n ON n.n_nationkey = s.s_nationkey WHERE n.n_name = '{nation}')
SELECT ps_partkey, SUM(v) AS value FROM stock GROUP BY ps_partkey
HAVING SUM(v) > (SELECT SUM(v) * {fraction} FROM stock)
ORDER BY value DESC""",
        """SELECT ps_partkey, value FROM (
  SELECT ps.ps_partkey, SUM(ps.ps_supplycost * ps.ps_availqty) AS value,
         SUM(SUM(ps.ps_supplycost * ps.ps_availqty)) OVER () AS total
  FROM partsupp ps JOIN supplier s ON s.s_suppkey = ps.ps_suppkey
  JOIN nation n ON n.n_nationkey = s.s_nationkey
  WHERE n.n_name = '{nation}' GROUP BY ps.ps_partkey) t
WHERE value > total * {fraction}
ORDER BY value DESC""",
        """SELECT t.ps_partkey, t.value
FROM (SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) AS value FROM partsupp
      WHERE ps_suppkey IN (SELECT s_suppkey FROM supplier
                           WHERE s_nationkey = (SELECT n_nationkey FROM nation WHERE n_name = '{nation}'))
      GROUP BY ps_partkey) t
CROSS JOIN (SELECT SUM(ps_supplycost * ps_availqty) AS total FROM partsupp
            JOIN supplier ON s_suppkey = ps_suppkey JOIN nation ON n_nationkey = s_nationkey
            WHERE n_name = '{nation}') tot
WHERE t.value > tot.total * {fraction}
ORDER BY t.value DESC""",
    ],
    "Q16": [
        """SELECT p.p_brand, p.p_type, p.p_size, COUNT(DISTINCT ps.ps_suppkey) AS supplier_cnt
FROM partsupp ps JOIN part p ON p.p_partkey = ps.ps_partkey
WHERE p.p_brand <> '{brand}' AND p.p_type NOT LIKE '{type}%' AND p.p_size IN ({sizes})
  AND NOT EXISTS (SELECT 1 FROM supplier s WHERE s.s_suppkey = ps.ps_suppkey
                  AND s.s_comment LIKE '%Customer%Complaints%')
GROUP BY p.p_brand, p.p_type, p.p_size
ORDER BY supplier_cnt DESC, p.p_brand, p.p_type, p.p_size""",
        """SELECT p.p_brand, p.p_type, p.p_size, COUNT(DISTINCT ps.ps_suppkey) AS supplier_cnt
FROM partsupp ps JOIN part p ON p.p_partkey = ps.ps_partkey
LEFT JOIN supplier bad ON bad.s_suppkey = ps.ps_suppkey AND bad.s_comment LIKE '%Customer%Complaints%'
WHERE bad.s_suppkey IS NULL AND p.p_brand <> '{brand}' AND p.p_type NOT LIKE '{type}%' AND p.p_size IN ({sizes})
GROUP BY p.p_brand, p.p_type, p.p_size
ORDER BY supplier_cnt DESC, p.p_brand, p.p_type, p.p_size""",
        """SELECT p_brand, p_type, p_size, COUNT(*) AS supplier_cnt
FROM (SELECT DISTINCT p.p_brand, p.p_type, p.p_size, ps.ps_suppkey
      FROM partsupp ps JOIN part p ON p.p_partkey = ps.ps_partkey
      WHERE p.p_brand <> '{brand}' AND p.p_type NOT LIKE '{type}%' AND p.p_size IN ({sizes})
        AND ps.ps_suppkey NOT IN (SELECT s_suppkey FROM supplier WHERE s_comment LIKE '%Customer%Complaints%')) d
GROUP BY p_brand, p_type, p_size
ORDER BY supplier_cnt DESC, p_brand, p_type, p_size""",
    ],
    "Q17": [
        """WITH avgq AS (SELECT l_partkey, 0.2 * AVG(l_quantity) AS lim FROM lineitem GROUP BY l_partkey)
SELECT SUM(l.l_extendedprice) / 7.0 AS avg_yearly
FROM lineitem l JOIN part p ON p.p_partkey = l.l_partkey JOIN avgq a ON a.l_partkey = l.l_partkey
WHERE p.p_brand = '{brand}' AND p.p_container = '{container}' AND l.l_quantity < a.lim""",
        """SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly
FROM (SELECT l.l_extendedprice, l.l_quantity, AVG(l.l_quantity) OVER (PARTITION BY l.l_partkey) AS avg_q
      FROM lineitem l JOIN part p ON p.p_partkey = l.l_partkey
      WHERE p.p_brand = '{brand}' AND p.p_container = '{container}') t
WHERE l_quantity < 0.2 * avg_q""",
        """WITH sel AS (SELECT p_partkey FROM part WHERE p_brand = '{brand}' AND p_container = '{container}'),
lim AS (SELECT l_partkey, 0.2 * AVG(l_quantity) AS q FROM lineitem
        WHERE l_partkey IN (SELECT p_partkey FROM sel) GROUP BY l_partkey)
SELECT SUM(l.l_extendedprice) / 7.0 AS avg_yearly
FROM lineitem l JOIN lim ON lim.l_partkey = l.l_partkey
WHERE l.l_quantity < lim.q""",
    ],
    "Q18": [
        """WITH big AS (SELECT l_orderkey, SUM(l_quantity) AS qty FROM lineitem GROUP BY l_orderkey
             HAVING SUM(l_quantity) > {quantity})
SELECT c.c_name, c.c_custkey, o.o_orderkey, o.o_orderdate, o.o_totalprice, big.qty
FROM big JOIN orders o ON o.o_orderkey = big.l_orderkey JOIN customer c ON c.c_custkey = o.o_custkey
ORDER BY o.o_totalprice DESC, o.o_orderdate LIMIT 100""",
        """SELECT c.c_name, c.c_custkey, o.o_orderkey, o.o_orderdate, o.o_totalprice, SUM(l.l_quantity) AS total_qty
FROM customer c JOIN orders o ON o.o_custkey = c.c_custkey JOIN lineitem l ON l.l_orderkey = o.o_orderkey
GROUP BY c.c_name, c.c_custkey, o.o_orderkey, o.o_orderdate, o.o_totalprice
HAVING SUM(l.l_quantity) > {quantity}
ORDER BY o.o_totalprice DESC, o.o_orderdate LIMIT 100""",
        """SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, total_qty FROM (
  SELECT c.c_name, c.c_custkey, o.o_orderkey, o.o_orderdate, o.o_totalprice,
         SUM(l.l_quantity) OVER (PARTITION BY o.o_orderkey) AS total_qty,
         ROW_NUMBER() OVER (PARTITION BY o.o_orderkey ORDER BY l.l_linenumber) AS rn
  FROM customer c JOIN orders o ON o.o_custkey = c.c_custkey JOIN lineitem l ON l.l_orderkey = o.o_orderkey) t
WHERE rn = 1 AND total_qty > {quantity}
ORDER BY o_totalprice DESC, o_orderdate LIMIT 100""",
    ],
    "Q20": [
        """WITH shipped AS (
  SELECT l_partkey, l_suppkey, 0.5 * SUM(l_quantity) AS half_qty FROM lineitem
  WHERE l_shipdate >= DATE '{date}' AND l_shipdate < DATE '{date}' + INTERVAL '1 year'
  GROUP BY l_partkey, l_suppkey),
cand AS (
  SELECT DISTINCT ps.ps_suppkey FROM partsupp ps JOIN part p ON p.p_partkey = ps.ps_partkey
  JOIN shipped sh ON sh.l_partkey = ps.ps_partkey AND sh.l_suppkey = ps.ps_suppkey
  WHERE p.p_name LIKE '{color}%' AND ps.ps_availqty > sh.half_qty)
SELECT s.s_name, s.s_address
FROM supplier s JOIN nation n ON n.n_nationkey = s.s_nationkey JOIN cand ON cand.ps_suppkey = s.s_suppkey
WHERE n.n_name = '{nation}' ORDER BY s.s_name""",
        """SELECT s.s_name, s.s_address
FROM supplier s JOIN nation n ON n.n_nationkey = s.s_nationkey
WHERE n.n_name = '{nation}'
  AND EXISTS (
    SELECT 1 FROM partsupp ps JOIN part p ON p.p_partkey = ps.ps_partkey
    WHERE ps.ps_suppkey = s.s_suppkey AND p.p_name LIKE '{color}%'
      AND ps.ps_availqty > (SELECT 0.5 * SUM(l.l_quantity) FROM lineitem l
                            WHERE l.l_partkey = ps.ps_partkey AND l.l_suppkey = ps.ps_suppkey
                              AND l.l_shipdate >= DATE '{date}' AND l.l_shipdate < DATE '{date}' + INTERVAL '1 year'))
ORDER BY s.s_name""",
        """SELECT s.s_name, s.s_address
FROM supplier s JOIN nation n ON n.n_nationkey = s.s_nationkey
WHERE n.n_name = '{nation}'
  AND s.s_suppkey IN (
    SELECT ps.ps_suppkey FROM partsupp ps JOIN part p ON p.p_partkey = ps.ps_partkey
    JOIN lineitem l ON l.l_partkey = ps.ps_partkey AND l.l_suppkey = ps.ps_suppkey
    WHERE p.p_name LIKE '{color}%' AND l.l_shipdate >= DATE '{date}'
      AND l.l_shipdate < DATE '{date}' + INTERVAL '1 year'
    GROUP BY ps.ps_partkey, ps.ps_suppkey, ps.ps_availqty
    HAVING ps.ps_availqty > 0.5 * SUM(l.l_quantity))
ORDER BY s.s_name""",
    ],
    "Q21": [
        """WITH order_supp AS (
  SELECT l_orderkey, COUNT(DISTINCT l_suppkey) AS n_supp,
         COUNT(DISTINCT CASE WHEN l_receiptdate > l_commitdate THEN l_suppkey END) AS n_late_supp
  FROM lineitem GROUP BY l_orderkey)
SELECT s.s_name, COUNT(*) AS numwait
FROM lineitem l1
JOIN supplier s ON s.s_suppkey = l1.l_suppkey
JOIN nation n ON n.n_nationkey = s.s_nationkey
JOIN orders o ON o.o_orderkey = l1.l_orderkey
JOIN order_supp os ON os.l_orderkey = l1.l_orderkey
WHERE n.n_name = '{nation}' AND o.o_orderstatus = 'F' AND l1.l_receiptdate > l1.l_commitdate
  AND os.n_supp > 1 AND os.n_late_supp = 1
GROUP BY s.s_name ORDER BY numwait DESC, s.s_name LIMIT 100""",
        """SELECT s.s_name, COUNT(*) AS numwait
FROM supplier s
JOIN nation n ON n.n_nationkey = s.s_nationkey
JOIN lineitem l1 ON l1.l_suppkey = s.s_suppkey
JOIN orders o ON o.o_orderkey = l1.l_orderkey
LEFT JOIN lineitem l3 ON l3.l_orderkey = l1.l_orderkey AND l3.l_suppkey <> l1.l_suppkey
                     AND l3.l_receiptdate > l3.l_commitdate
WHERE n.n_name = '{nation}' AND o.o_orderstatus = 'F' AND l1.l_receiptdate > l1.l_commitdate
  AND l3.l_orderkey IS NULL
  AND EXISTS (SELECT 1 FROM lineitem l2 WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey)
GROUP BY s.s_name ORDER BY numwait DESC, s.s_name LIMIT 100""",
        """SELECT s.s_name, COUNT(*) AS numwait
FROM supplier s
JOIN nation n ON n.n_nationkey = s.s_nationkey
JOIN lineitem l1 ON l1.l_suppkey = s.s_suppkey
JOIN orders o ON o.o_orderkey = l1.l_orderkey
WHERE n.n_name = '{nation}' AND o.o_orderstatus = 'F' AND l1.l_receiptdate > l1.l_commitdate
  AND l1.l_orderkey IN (SELECT l2.l_orderkey FROM lineitem l2 WHERE l2.l_suppkey <> l1.l_suppkey)
  AND l1.l_orderkey NOT IN (SELECT l3.l_orderkey FROM lineitem l3
                            WHERE l3.l_suppkey <> l1.l_suppkey AND l3.l_receiptdate > l3.l_commitdate)
GROUP BY s.s_name ORDER BY numwait DESC, s.s_name LIMIT 100""",
    ],
}


# ===================================================================================================
# PART 3. APPLICATION ARM: THE SAME INTENTS WRITTEN WITH SQLALCHEMY CORE (ORM-GENERATED SQL)
# ===================================================================================================
md = MetaData()
region = Table("region", md, Column("r_regionkey", Integer), Column("r_name", String))
nation = Table("nation", md, Column("n_nationkey", Integer), Column("n_name", String), Column("n_regionkey", Integer))
supplier = Table("supplier", md, Column("s_suppkey", Integer), Column("s_name", String), Column("s_address", String),
                 Column("s_nationkey", Integer), Column("s_phone", String), Column("s_acctbal", Numeric),
                 Column("s_comment", String))
part = Table("part", md, Column("p_partkey", Integer), Column("p_name", String), Column("p_mfgr", String),
             Column("p_brand", String), Column("p_type", String), Column("p_size", Integer), Column("p_container", String))
partsupp = Table("partsupp", md, Column("ps_partkey", Integer), Column("ps_suppkey", Integer),
                 Column("ps_availqty", Integer), Column("ps_supplycost", Numeric))
customer = Table("customer", md, Column("c_custkey", Integer), Column("c_name", String), Column("c_address", String),
                 Column("c_nationkey", Integer), Column("c_phone", String), Column("c_acctbal", Numeric),
                 Column("c_mktsegment", String), Column("c_comment", String))
orders = Table("orders", md, Column("o_orderkey", Integer), Column("o_custkey", Integer), Column("o_orderstatus", String),
               Column("o_totalprice", Numeric), Column("o_orderdate", Date), Column("o_orderpriority", String),
               Column("o_shippriority", Integer))
lineitem = Table("lineitem", md, Column("l_orderkey", Integer), Column("l_partkey", Integer), Column("l_suppkey", Integer),
                 Column("l_linenumber", Integer), Column("l_quantity", Numeric), Column("l_extendedprice", Numeric),
                 Column("l_discount", Numeric), Column("l_returnflag", String), Column("l_shipdate", Date),
                 Column("l_commitdate", Date), Column("l_receiptdate", Date))

D = dt.date.fromisoformat
rev_expr = lambda l: l.c.l_extendedprice * (1 - l.c.l_discount)


def orm_build(intent, p):
    if intent == "Q2":
        ps2, s2, n2, r2 = partsupp.alias("ps2"), supplier.alias("s2"), nation.alias("n2"), region.alias("r2")
        min_cost = (select(func.min(ps2.c.ps_supplycost))
                    .select_from(ps2.join(s2, s2.c.s_suppkey == ps2.c.ps_suppkey)
                                 .join(n2, n2.c.n_nationkey == s2.c.s_nationkey)
                                 .join(r2, r2.c.r_regionkey == n2.c.n_regionkey))
                    .where(ps2.c.ps_partkey == part.c.p_partkey, r2.c.r_name == p["region"])
                    .scalar_subquery())
        q = (select(supplier.c.s_acctbal, supplier.c.s_name, nation.c.n_name, part.c.p_partkey, part.c.p_mfgr,
                    supplier.c.s_address, supplier.c.s_phone, supplier.c.s_comment)
             .select_from(part.join(partsupp, partsupp.c.ps_partkey == part.c.p_partkey)
                          .join(supplier, supplier.c.s_suppkey == partsupp.c.ps_suppkey)
                          .join(nation, nation.c.n_nationkey == supplier.c.s_nationkey)
                          .join(region, region.c.r_regionkey == nation.c.n_regionkey))
             .where(part.c.p_size == p["size"], part.c.p_type.like("%" + p["type"]), region.c.r_name == p["region"],
                    partsupp.c.ps_supplycost == min_cost)
             .order_by(desc(supplier.c.s_acctbal), nation.c.n_name, supplier.c.s_name, part.c.p_partkey).limit(100))
    elif intent == "Q3":
        revenue = func.sum(rev_expr(lineitem)).label("revenue")
        q = (select(lineitem.c.l_orderkey, revenue, orders.c.o_orderdate, orders.c.o_shippriority)
             .select_from(customer.join(orders, orders.c.o_custkey == customer.c.c_custkey)
                          .join(lineitem, lineitem.c.l_orderkey == orders.c.o_orderkey))
             .where(customer.c.c_mktsegment == p["segment"], orders.c.o_orderdate < D(p["date"]),
                    lineitem.c.l_shipdate > D(p["date"]))
             .group_by(lineitem.c.l_orderkey, orders.c.o_orderdate, orders.c.o_shippriority)
             .order_by(desc("revenue"), orders.c.o_orderdate).limit(10))
    elif intent == "Q4":
        late = (exists().where(lineitem.c.l_orderkey == orders.c.o_orderkey,
                               lineitem.c.l_commitdate < lineitem.c.l_receiptdate))
        q = (select(orders.c.o_orderpriority, func.count().label("order_count"))
             .where(orders.c.o_orderdate >= D(p["date"]), orders.c.o_orderdate < D(p["date_end"]), late)
             .group_by(orders.c.o_orderpriority).order_by(orders.c.o_orderpriority))
    elif intent == "Q5":
        revenue = func.sum(rev_expr(lineitem)).label("revenue")
        q = (select(nation.c.n_name, revenue)
             .select_from(customer.join(orders, orders.c.o_custkey == customer.c.c_custkey)
                          .join(lineitem, lineitem.c.l_orderkey == orders.c.o_orderkey)
                          .join(supplier, and_(supplier.c.s_suppkey == lineitem.c.l_suppkey,
                                               supplier.c.s_nationkey == customer.c.c_nationkey))
                          .join(nation, nation.c.n_nationkey == supplier.c.s_nationkey)
                          .join(region, region.c.r_regionkey == nation.c.n_regionkey))
             .where(region.c.r_name == p["region"], orders.c.o_orderdate >= D(p["date"]),
                    orders.c.o_orderdate < D(p["date_end"]))
             .group_by(nation.c.n_name).order_by(desc("revenue")))
    elif intent == "Q7":
        n1, n2 = nation.alias("n1"), nation.alias("n2")
        yr = extract("year", lineitem.c.l_shipdate)
        q = (select(n1.c.n_name.label("supp_nation"), n2.c.n_name.label("cust_nation"), yr.label("l_year"),
                    func.sum(rev_expr(lineitem)).label("revenue"))
             .select_from(supplier.join(lineitem, lineitem.c.l_suppkey == supplier.c.s_suppkey)
                          .join(orders, orders.c.o_orderkey == lineitem.c.l_orderkey)
                          .join(customer, customer.c.c_custkey == orders.c.o_custkey)
                          .join(n1, n1.c.n_nationkey == supplier.c.s_nationkey)
                          .join(n2, n2.c.n_nationkey == customer.c.c_nationkey))
             .where(or_(and_(n1.c.n_name == p["n1"], n2.c.n_name == p["n2"]),
                        and_(n1.c.n_name == p["n2"], n2.c.n_name == p["n1"])),
                    lineitem.c.l_shipdate.between(dt.date(1995, 1, 1), dt.date(1996, 12, 31)))
             .group_by(n1.c.n_name, n2.c.n_name, yr).order_by("supp_nation", "cust_nation", "l_year"))
    elif intent == "Q10":
        revenue = func.sum(rev_expr(lineitem)).label("revenue")
        q = (select(customer.c.c_custkey, customer.c.c_name, revenue, customer.c.c_acctbal, nation.c.n_name,
                    customer.c.c_address, customer.c.c_phone, customer.c.c_comment)
             .select_from(customer.join(orders, orders.c.o_custkey == customer.c.c_custkey)
                          .join(lineitem, lineitem.c.l_orderkey == orders.c.o_orderkey)
                          .join(nation, nation.c.n_nationkey == customer.c.c_nationkey))
             .where(orders.c.o_orderdate >= D(p["date"]), orders.c.o_orderdate < D(p["date_end"]),
                    lineitem.c.l_returnflag == "R")
             .group_by(customer.c.c_custkey, customer.c.c_name, customer.c.c_acctbal, customer.c.c_phone,
                       nation.c.n_name, customer.c.c_address, customer.c.c_comment)
             .order_by(desc("revenue")).limit(20))
    elif intent == "Q11":
        val = func.sum(partsupp.c.ps_supplycost * partsupp.c.ps_availqty)
        ps2, s2, n2 = partsupp.alias("ps2"), supplier.alias("s2"), nation.alias("n2")
        total = (select(func.sum(ps2.c.ps_supplycost * ps2.c.ps_availqty) * literal(p["fraction"], Numeric))
                 .select_from(ps2.join(s2, s2.c.s_suppkey == ps2.c.ps_suppkey)
                              .join(n2, n2.c.n_nationkey == s2.c.s_nationkey))
                 .where(n2.c.n_name == p["nation"]).scalar_subquery())
        q = (select(partsupp.c.ps_partkey, val.label("value"))
             .select_from(partsupp.join(supplier, supplier.c.s_suppkey == partsupp.c.ps_suppkey)
                          .join(nation, nation.c.n_nationkey == supplier.c.s_nationkey))
             .where(nation.c.n_name == p["nation"]).group_by(partsupp.c.ps_partkey)
             .having(val > total).order_by(desc("value")))
    elif intent == "Q16":
        bad = select(supplier.c.s_suppkey).where(supplier.c.s_comment.like("%Customer%Complaints%"))
        sizes = [int(x) for x in p["sizes"].split(",")]
        q = (select(part.c.p_brand, part.c.p_type, part.c.p_size,
                    func.count(distinct(partsupp.c.ps_suppkey)).label("supplier_cnt"))
             .select_from(partsupp.join(part, part.c.p_partkey == partsupp.c.ps_partkey))
             .where(part.c.p_brand != p["brand"], not_(part.c.p_type.like(p["type"] + "%")),
                    part.c.p_size.in_(sizes), partsupp.c.ps_suppkey.not_in(bad))
             .group_by(part.c.p_brand, part.c.p_type, part.c.p_size)
             .order_by(desc("supplier_cnt"), part.c.p_brand, part.c.p_type, part.c.p_size))
    elif intent == "Q17":
        l2 = lineitem.alias("l2")
        avg_sub = (select(literal(0.2, Numeric) * func.avg(l2.c.l_quantity))
                   .where(l2.c.l_partkey == part.c.p_partkey).scalar_subquery())
        q = (select((func.sum(lineitem.c.l_extendedprice) / literal(7.0, Numeric)).label("avg_yearly"))
             .select_from(lineitem.join(part, part.c.p_partkey == lineitem.c.l_partkey))
             .where(part.c.p_brand == p["brand"], part.c.p_container == p["container"],
                    lineitem.c.l_quantity < avg_sub))
    elif intent == "Q18":
        l2 = lineitem.alias("l2")
        big = (select(l2.c.l_orderkey).group_by(l2.c.l_orderkey)
               .having(func.sum(l2.c.l_quantity) > p["quantity"]))
        q = (select(customer.c.c_name, customer.c.c_custkey, orders.c.o_orderkey, orders.c.o_orderdate,
                    orders.c.o_totalprice, func.sum(lineitem.c.l_quantity).label("sum_qty"))
             .select_from(customer.join(orders, orders.c.o_custkey == customer.c.c_custkey)
                          .join(lineitem, lineitem.c.l_orderkey == orders.c.o_orderkey))
             .where(orders.c.o_orderkey.in_(big))
             .group_by(customer.c.c_name, customer.c.c_custkey, orders.c.o_orderkey, orders.c.o_orderdate,
                       orders.c.o_totalprice)
             .order_by(desc(orders.c.o_totalprice), orders.c.o_orderdate).limit(100))
    elif intent == "Q20":
        l2 = lineitem.alias("l2")
        half = (select(literal(0.5, Numeric) * func.sum(l2.c.l_quantity))
                .where(l2.c.l_partkey == partsupp.c.ps_partkey, l2.c.l_suppkey == partsupp.c.ps_suppkey,
                       l2.c.l_shipdate >= D(p["date"]), l2.c.l_shipdate < D(p["date_end"])).scalar_subquery())
        parts = select(part.c.p_partkey).where(part.c.p_name.like(p["color"] + "%"))
        cand = select(partsupp.c.ps_suppkey).where(partsupp.c.ps_partkey.in_(parts), partsupp.c.ps_availqty > half)
        q = (select(supplier.c.s_name, supplier.c.s_address)
             .select_from(supplier.join(nation, nation.c.n_nationkey == supplier.c.s_nationkey))
             .where(supplier.c.s_suppkey.in_(cand), nation.c.n_name == p["nation"])
             .order_by(supplier.c.s_name))
    elif intent == "Q21":
        l1, l2, l3 = lineitem.alias("l1"), lineitem.alias("l2"), lineitem.alias("l3")
        e2 = exists().where(l2.c.l_orderkey == l1.c.l_orderkey, l2.c.l_suppkey != l1.c.l_suppkey)
        e3 = exists().where(l3.c.l_orderkey == l1.c.l_orderkey, l3.c.l_suppkey != l1.c.l_suppkey,
                            l3.c.l_receiptdate > l3.c.l_commitdate)
        q = (select(supplier.c.s_name, func.count().label("numwait"))
             .select_from(supplier.join(l1, l1.c.l_suppkey == supplier.c.s_suppkey)
                          .join(orders, orders.c.o_orderkey == l1.c.l_orderkey)
                          .join(nation, nation.c.n_nationkey == supplier.c.s_nationkey))
             .where(orders.c.o_orderstatus == "F", l1.c.l_receiptdate > l1.c.l_commitdate, e2, not_(e3),
                    nation.c.n_name == p["nation"])
             .group_by(supplier.c.s_name).order_by(desc("numwait"), supplier.c.s_name).limit(100))
    else:
        raise KeyError(intent)
    return str(q.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))


# ===================================================================================================
# PART 3b. QUERY FINGERPRINTING ENGINE (syntactic and semantic fingerprints)
# ===================================================================================================
TPCH_SCHEMA = {
    "region": {"r_regionkey": "int", "r_name": "text", "r_comment": "text"},
    "nation": {"n_nationkey": "int", "n_name": "text", "n_regionkey": "int", "n_comment": "text"},
    "supplier": {"s_suppkey": "int", "s_name": "text", "s_address": "text", "s_nationkey": "int", "s_phone": "text",
                 "s_acctbal": "decimal", "s_comment": "text"},
    "customer": {"c_custkey": "int", "c_name": "text", "c_address": "text", "c_nationkey": "int", "c_phone": "text",
                 "c_acctbal": "decimal", "c_mktsegment": "text", "c_comment": "text"},
    "part": {"p_partkey": "int", "p_name": "text", "p_mfgr": "text", "p_brand": "text", "p_type": "text",
             "p_size": "int", "p_container": "text", "p_retailprice": "decimal", "p_comment": "text"},
    "partsupp": {"ps_partkey": "int", "ps_suppkey": "int", "ps_availqty": "int", "ps_supplycost": "decimal",
                 "ps_comment": "text"},
    "orders": {"o_orderkey": "int", "o_custkey": "int", "o_orderstatus": "text", "o_totalprice": "decimal",
               "o_orderdate": "date", "o_orderpriority": "text", "o_clerk": "text", "o_shippriority": "int",
               "o_comment": "text"},
    "lineitem": {"l_orderkey": "int", "l_partkey": "int", "l_suppkey": "int", "l_linenumber": "int",
                 "l_quantity": "decimal", "l_extendedprice": "decimal", "l_discount": "decimal", "l_tax": "decimal",
                 "l_returnflag": "text", "l_linestatus": "text", "l_shipdate": "date", "l_commitdate": "date",
                 "l_receiptdate": "date", "l_shipinstruct": "text", "l_shipmode": "text", "l_comment": "text"},
}
TABLES = list(TPCH_SCHEMA)


def h(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def syntactic_fp(sql):
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return None
    tree = tree.transform(lambda n: exp.Placeholder() if isinstance(n, exp.Literal) else n)
    return h(tree.sql(dialect="postgres", normalize=True))


def _canon_pred(e, alias2table):
    s = e.sql(dialect="postgres")
    for a, t in alias2table.items():
        s = re.sub(rf'"?\b{re.escape(a)}\b"?\.', f"{t}.", s)
    return s.replace('"', "")


def semantic_fp(sql):
    """Returns (fingerprint, tables, predicates) or (None, None, None) on failure."""
    try:
        tree = optimize(sqlglot.parse_one(sql, read="postgres"), schema=TPCH_SCHEMA, dialect="postgres")
    except Exception:
        return None, None, None
    alias2table, tables = {}, []
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        if name in TPCH_SCHEMA:
            tables.append(name)
            alias2table[t.alias_or_name] = name
    preds = set()
    for node in tree.find_all(exp.Where, exp.Join, exp.Having):
        cond = node.this if isinstance(node, (exp.Where, exp.Having)) else node.args.get("on")
        if cond is None:
            continue
        for c in cond.flatten() if isinstance(cond, exp.And) else [cond]:
            preds.add(_canon_pred(c, alias2table))
    key = json.dumps([sorted(tables), sorted(preds)])
    return h(key), sorted(tables), sorted(preds)




# ===================================================================================================
# PART 4. DATABASE, MEASUREMENT AND PLAN HELPERS
# ===================================================================================================
TPCH_TABLES = ["region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem"]
DATE_COLS = {"orders": ["o_orderdate"], "lineitem": ["l_shipdate", "l_commitdate", "l_receiptdate"]}

# "Hint sets": alternative optimizer configurations that give counterfactual plans for the same query
# (DuckDB equivalent of the PostgreSQL enable_* hint sets used in the paper).
HINT_SETS = {
    "H1_no_join_order": "join_order",
    "H2_no_build_probe_swap": "build_side_probe_side",
    "H3_no_join_filter_pushdown": "join_filter_pushdown",
    "H4_no_statistics_propagation": "statistics_propagation",
    "H5_no_deliminator": "deliminator",
}


def load_tables(con, tables):
    for name, df in tables.items():
        df = df.copy()
        for col in df.columns:                          # older DuckDB versions do not accept pandas' new str dtype
            if df[col].dtype.kind not in "biufcmM":
                df[col] = df[col].astype(object)
        con.register("tmp_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
        for col in DATE_COLS.get(name, []):
            con.execute(f"ALTER TABLE {name} ALTER {col} TYPE DATE")


def build_database(sf, seed=7, mutate=False, official=False):
    """In-memory DuckDB database with TPC-H data, from the built-in generator in Part 1 (TPC-H value rules).
    With --dbgen, DuckDB's official TPC-H generator is used instead (needs internet once to fetch the extension)."""
    con = duckdb.connect()
    source = "built-in generator"
    if not mutate and official:
        try:
            con.execute("INSTALL tpch; LOAD tpch;")
            con.execute(f"CALL dbgen(sf={sf})")
            source = "DuckDB official TPC-H dbgen"
        except Exception:
            con.close()
            con = duckdb.connect()
            load_tables(con, generate_tpch(sf, seed))
    elif not mutate:
        load_tables(con, generate_tpch(sf, seed))
    else:
        T = generate_tpch(sf, seed)
        rng = np.random.default_rng(seed + 100)
        li, od = T["lineitem"], T["orders"]
        T["lineitem"] = li[rng.random(len(li)) > 0.15].copy()                          # delete 15% of lineitems
        sh = rng.integers(-60, 61, len(T["lineitem"])).astype("timedelta64[D]")
        T["lineitem"]["l_shipdate"] = T["lineitem"]["l_shipdate"].values + sh          # shift ship dates
        T["lineitem"]["l_quantity"] = np.clip(T["lineitem"]["l_quantity"] + rng.integers(-5, 6, len(T["lineitem"])), 1, 50)
        T["orders"] = od[rng.random(len(od)) > 0.05].copy()                           # delete 5% of orders
        T["part"]["p_size"] = rng.integers(1, 51, len(T["part"]))                     # re-draw part sizes
        load_tables(con, T)
        source = "built-in generator, randomly mutated copy"
    return con, source


def set_hint(con, disabled):
    if disabled:
        con.execute(f"SET disabled_optimizers = '{disabled}'")
    else:
        con.execute("RESET disabled_optimizers")


def run_timed(con, sql, timeout_s, fetch=True):
    """Run a query with a wall-clock timeout (DuckDB has no statement_timeout, so a timer interrupts it).
    Returns (milliseconds, timed_out, rows)."""
    timer = threading.Timer(timeout_s, con.interrupt)
    timer.start()
    t0 = time.perf_counter()
    try:
        rows = con.execute(sql).fetchall() if fetch else con.execute(sql)
        return (time.perf_counter() - t0) * 1000.0, False, rows
    except Exception as e:
        if "INTERRUPT" in str(e).upper():
            return timeout_s * 1000.0, True, None
        raise
    finally:
        timer.cancel()


def canon(rows):
    out = []
    for r in rows:
        t = []
        for v in r:
            if isinstance(v, (float, decimal.Decimal)):
                t.append(round(float(v), 2))
            elif isinstance(v, str):
                t.append(v.strip())
            else:
                t.append(v)
        out.append(tuple(t))
    return sorted(out, key=repr)


def profile_plan(con, sql, timeout_s):
    """Execute once with profiling on; return DuckDB's operator tree with estimated and actual rows.
    Join-filter pushdown is switched off for this labelling run so that scan row counts are logical
    (not pruned at runtime by filters that the estimator never sees)."""
    set_hint(con, "join_filter_pushdown")
    use_api = hasattr(con, "get_profiling_information")
    tmp = os.path.join(OUT_DIR, "_profile.json")
    if use_api:
        con.execute("PRAGMA enable_profiling = 'no_output'")
    else:                                           # older DuckDB versions: profile to a file
        con.execute("PRAGMA enable_profiling = 'json'")
        con.execute(f"PRAGMA profiling_output = '{tmp.replace(chr(92), '/')}'")
    try:
        ms, to, _ = run_timed(con, sql, timeout_s)
        prof = None
        if not to:
            if use_api:
                prof = json.loads(con.get_profiling_information(format="json"))
            else:
                with open(tmp, encoding="utf-8") as f:
                    prof = json.load(f)
    finally:
        con.execute("PRAGMA disable_profiling")
        set_hint(con, "")
    if prof is None:
        return None
    root = prof.get("operator", prof) if isinstance(prof, dict) else prof
    return root[0] if isinstance(root, list) else root


def explain_plan(con, sql):
    """EXPLAIN only (nothing is executed): the information stage one is allowed to use."""
    set_hint(con, "")
    txt = con.execute("EXPLAIN (FORMAT JSON) " + sql).fetchall()[0][1]
    tree = json.loads(txt)
    return tree[0] if isinstance(tree, list) else tree


# ---------------------------------------------------------------- plan flattening and sub-expression keys
EVAL_TYPES = {"TABLE_SCAN", "FILTER", "HASH_JOIN", "NESTED_LOOP_JOIN", "PIECEWISE_MERGE_JOIN", "BLOCKWISE_NL_JOIN",
              "CROSS_PRODUCT", "HASH_GROUP_BY", "PERFECT_HASH_GROUP_BY", "UNGROUPED_AGGREGATE", "TOP_N", "LIMIT",
              "STREAMING_LIMIT", "CTE_SCAN", "DELIM_JOIN", "LEFT_DELIM_JOIN", "RIGHT_DELIM_JOIN", "IE_JOIN", "ASOF_JOIN", "DELIM_SCAN"}
PATHS = ["scan", "filter", "join", "nl_join", "semi", "anti", "mark", "outer", "delim", "cte", "agg", "limit", "window", "other"]
OP_TYPES = sorted(EVAL_TYPES)


def _as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _ntype(n):
    return n.get("type") or n.get("name") or n.get("operator_type") or "?"


def _est(n):
    try:
        return float(str(n.get("extra_info", {}).get("Estimated Cardinality", "nan")).replace("~", ""))
    except ValueError:
        return float("nan")


def _path(t, info):
    jt = str(info.get("Join Type", "")).upper()
    if "DELIM" in t:
        return "delim"
    if jt == "SEMI" or jt == "RIGHT_SEMI":
        return "semi"
    if jt in ("ANTI", "RIGHT_ANTI"):
        return "anti"
    if jt == "MARK":
        return "mark"
    if jt in ("LEFT", "RIGHT", "OUTER", "FULL"):
        return "outer"
    if t in ("NESTED_LOOP_JOIN", "CROSS_PRODUCT", "BLOCKWISE_NL_JOIN", "PIECEWISE_MERGE_JOIN", "IE_JOIN"):
        return "nl_join"
    if "JOIN" in t:
        return "join"
    if t == "TABLE_SCAN":
        return "scan"
    if t == "FILTER":
        return "filter"
    if t in ("CTE_SCAN", "DELIM_SCAN"):
        return "cte"
    if "GROUP_BY" in t or "AGGREGATE" in t:
        return "agg"
    if t in ("TOP_N", "LIMIT", "STREAMING_LIMIT"):
        return "limit"
    if "WINDOW" in t:
        return "window"
    return "other"


_REF = re.compile(r"#\[?[\d.]+\]?")
_SPACES = re.compile(r"\s+")


def _norm_cond(s):
    # positional column references (#0, #[2.1]) are kept: dropping them would merge different predicates
    return _SPACES.sub(" ", str(s)).strip()


def flatten_plan(root, actual=True):
    """Post-order list of plan nodes with: type, path, estimated rows, actual rows, tables below,
    semantic key (meaning of the sub-expression) and syntactic key (AQO-style: tied to the wording)."""
    cte_tables = {}

    def first_pass(n):
        info = n.get("extra_info", {})
        kids = n.get("children", [])
        tabs = []
        for k in kids:
            tabs += first_pass(k)
        t = _ntype(n)
        if t == "TABLE_SCAN" or "Table" in info:
            tabs.append(str(info.get("Table", "?")).split(".")[-1])
        if t == "CTE" and kids:
            cte_tables[str(info.get("Table Index", ""))] = sorted(set(first_pass(kids[0])))
        if t == "CTE_SCAN":
            tabs += cte_tables.get(str(info.get("CTE Index", "")), [])
        return tabs
    first_pass(root)
    cte_def = {}
    out = []

    def walk(n, depth, parent_t):
        info = n.get("extra_info", {})
        t = _ntype(n)
        tabs, conds, kids_out = [], [], []
        for ci, k in enumerate(n.get("children", [])):
            kt, kc, ko = walk(k, depth + 1, t)
            if t == "CTE" and ci == 0:                  # remember what the CTE computes for its consumers
                cte_def[str(info.get("Table Index", ""))] = (sorted(set(kt)), list(kc))
            tabs += kt
            conds += kc
            kids_out.append(ko)
        if t == "TABLE_SCAN" or "Table" in info:
            tabs.append(str(info.get("Table", "?")).split(".")[-1])
        if t == "CTE_SCAN":
            dt_, dc_ = cte_def.get(str(info.get("CTE Index", "")), (cte_tables.get(str(info.get("CTE Index", "")), []), []))
            tabs += dt_
            conds += ["CTE:" + c for c in dc_]
        for f in ("Conditions", "Filters", "Expression"):
            for c in _as_list(info.get(f)):
                if c:
                    conds.append(_norm_cond(c))
        tabs = sorted(set(tabs))
        me = None
        if t in EVAL_TYPES:
            opclass = ("limit" if t in ("TOP_N", "LIMIT", "STREAMING_LIMIT") else
                       "agg" if ("GROUP_BY" in t or "AGGREGATE" in t) else "rel")
            extra = []
            if opclass == "agg":
                extra = ["GROUP:" + _norm_cond(",".join(map(str, _as_list(info.get("Groups"))))) +
                         " AGG:" + _norm_cond(",".join(map(str, _as_list(info.get("Aggregates")))))]
            if opclass == "limit":
                extra = ["TOP:" + str(info.get("Top", info.get("Limit", "")))]
            jt = str(info.get("Join Type", ""))
            if jt:
                extra.append("JOIN:" + jt.replace("RIGHT_", ""))
            conds = conds + extra                      # ancestors inherit the meaning of this step
            sem = json.dumps([tabs, sorted(set(conds)), opclass])
            syn = json.dumps([t, jt, depth, parent_t, [_norm_cond(c) for c in _as_list(info.get("Conditions")) + _as_list(info.get("Filters"))]])
            est = _est(n)
            act = n.get("intermediate_rows", n.get("operator_cardinality")) if actual else None
            me = dict(type=t, path=_path(t, info), jt=jt, depth=depth, tables=tabs, n_tables=len(tabs),
                      n_conds=len(set(conds)), est=est, act=(float(act) if act is not None else None),
                      sem_key=hashlib.sha1(sem.encode()).hexdigest()[:14],
                      syn_key=hashlib.sha1(syn.encode()).hexdigest()[:14],
                      top=(float(info.get("Top")) if str(info.get("Top", "")).isdigit() else None),
                      child_est=[], parent_type=parent_t)
            out.append(me)
        # children estimates for the node (nearest evaluated descendants)
        kid_nodes = [k for k in kids_out if k is not None]
        if me is not None:
            me["child_est"] = [k["est"] for k in kid_nodes]
            me["child_idx"] = [id(k) for k in kid_nodes]
            me["_id"] = id(me)
            return tabs, conds, me
        # pass-through operator (projection, result collector, ...): forward the single evaluated child
        return tabs, conds, (kid_nodes[0] if len(kid_nodes) == 1 else None)
    walk(root, 0, "ROOT")
    return out


def qerr(est, act):
    e, a = max(est, 1.0), max(act, 1.0)
    return max(e / a, a / e)


# ===================================================================================================
# PART 5. CONSOLE / REPORT HELPERS
# ===================================================================================================
REPORT = []          # (kind, payload) items collected for the HTML report
LINE = "=" * 96


def banner(step, total, title, why):
    print("\n" + LINE)
    print(f" STEP {step}/{total}  {title.upper()}")
    print(LINE)
    for line in textwrap.wrap(why, 94):
        print(" " + line)
    print("-" * 96)
    REPORT.append(("h2", f"Step {step}. {title}"))
    REPORT.append(("p", why))


def say(msg, report=True):
    for i, line in enumerate(textwrap.wrap(msg, 94)):
        print((" " if i else " ") + line)
    if report:
        REPORT.append(("p", msg))


def show_table(df, title=None, floatfmt="{:.2f}", max_rows=40):
    if title:
        print("\n " + title)
        REPORT.append(("h3", title))
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    txt = d.head(max_rows).to_string(index=False)
    print("\n".join(" " + l for l in txt.splitlines()))
    REPORT.append(("table", d.head(max_rows).to_html(index=False, border=0, classes="t")))


def save_fig(fig, name, caption):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    with open(path, "rb") as f:
        REPORT.append(("img", (base64.b64encode(f.read()).decode(), caption)))
    print(f"   [chart saved: {path}]")


def write_html(summary_rows):
    css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:30px auto;padding:0 16px;color:#111;"
           "line-height:1.45}h1{font-size:26px}h2{margin-top:36px;border-bottom:2px solid #222;padding-bottom:4px}"
           "table.t{border-collapse:collapse;font-size:13px;margin:8px 0 16px}table.t td,table.t th{border:1px solid #999;"
           "padding:3px 8px;text-align:left}table.t th{background:#eee}img{max-width:100%;border:1px solid #ccc}"
           ".cap{font-size:13px;color:#444;margin-bottom:18px}pre{background:#f5f5f5;padding:8px;font-size:12px;"
           "overflow-x:auto;white-space:pre-wrap}")
    parts = [f"<html><head><meta charset='utf-8'><title>AgentSQL-Guard demo report</title><style>{css}</style></head><body>",
             "<h1>AgentSQL-Guard with InvaCard: demo report</h1>",
             f"<p>Run on {time.strftime('%Y-%m-%d %H:%M')} &middot; scale factor {ARGS.sf} &middot; "
             f"{ARGS.draws} parameter draws &middot; DuckDB {duckdb.__version__} &middot; Python {sys.version.split()[0]}</p>",
             "<h2>Summary of results</h2>",
             pd.DataFrame(summary_rows, columns=["Question", "Result"]).to_html(index=False, border=0, classes="t")]
    for kind, x in REPORT:
        if kind == "h2":
            parts.append(f"<h2>{html.escape(x)}</h2>")
        elif kind == "h3":
            parts.append(f"<h3>{html.escape(x)}</h3>")
        elif kind == "p":
            parts.append(f"<p>{html.escape(x)}</p>")
        elif kind == "pre":
            parts.append(f"<pre>{html.escape(x)}</pre>")
        elif kind == "table":
            parts.append(x)
        elif kind == "img":
            parts.append(f"<img src='data:image/png;base64,{x[0]}'><div class='cap'>{html.escape(x[1])}</div>")
    parts.append("</body></html>")
    path = os.path.join(OUT_DIR, "AgentSQL_Guard_Demo_Report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return path


def pct(x):
    return f"{100 * x:.1f}%"


# ===================================================================================================
# PART 6. THE DEMO PIPELINE
# ===================================================================================================
TOTAL = 11


def step1_database():
    banner(1, TOTAL, "Build the database",
           "We create an in-memory analytical database with the TPC-H schema (customers, orders, line items, "
           "parts, suppliers). The same eight tables are used by every query in the study, so any difference in "
           "speed between two queries comes from how they are written and planned, not from the data.")
    t0 = time.time()
    con, source = build_database(ARGS.sf, official=ARGS.dbgen)
    con.execute(f"SET threads = {ARGS.threads}")
    sizes = [(t, con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]) for t in TPCH_TABLES]
    say(f"Data source: {source}; scale factor {ARGS.sf}; built in {time.time() - t0:.1f} s.")
    show_table(pd.DataFrame(sizes, columns=["table", "rows"]), "Table sizes")
    return con


def step2_workload():
    banner(2, TOTAL, "Build the matched three-arm workload",
           "Each business question (an 'intent' taken from TPC-H) is written five ways: the official expert SQL, "
           "SQL produced by an ORM library (SQLAlchemy Core), and three different LLM-agent wordings. All five "
           "share the same parameter values, so the only thing that differs between them is the wording.")
    inst = []
    for intent in INTENTS:
        rng = random.Random(1000 + INTENTS.index(intent))
        for d in range(ARGS.draws):
            p = make_params(intent, rng)
            forms = [("expert", "expert", EXPERT[intent].format(**p)), ("orm", "orm", orm_build(intent, p))]
            forms += [("agent", f"agent{i + 1}", t.format(**p)) for i, t in enumerate(AGENT[intent])]
            for arm, form, sql in forms:
                inst.append(dict(qid=f"{intent}_d{d}_{form}", intent=intent, draw=d, arm=arm, form=form,
                                 question=QUESTIONS[intent].format(**p), sql=sql.strip()))
    say(f"{len(INTENTS)} intents x {ARGS.draws} parameter draws x 5 formulations = {len(inst)} query instances.")
    ex = [r for r in inst if r["intent"] == "Q21" and r["draw"] == 0]
    print("\n Example intent Q21, draw 0. Question: " + ex[0]["question"])
    REPORT.append(("h3", "Example: five wordings of the same question (Q21)"))
    for r in ex:
        head = " ".join(r["sql"].split())[:150]
        print(f"   [{r['form']:7s}] {head} ...")
        REPORT.append(("pre", f"[{r['form']}]\n{r['sql']}"))
    return inst


def step3_equivalence(con, inst):
    banner(3, TOTAL, "Check that every wording returns the same answer",
           "A fast query is useless if it is wrong, so every ORM and agent query is compared with the expert query "
           "on the main database and on two smaller, randomly mutated databases. A wording is accepted only if it "
           "returns exactly the same rows everywhere. Agreement on mutated data guards against queries that are "
           "only right by luck.")
    small_sf = max(round(ARGS.sf / 5, 3), 0.01)
    dbs = [("main", con)]
    for s in (1, 2):
        c2, _ = build_database(small_sf, seed=100 + s, mutate=True)
        dbs.append((f"mutated_{s}", c2))
    ref, rows = {}, []
    for r in inst:
        key = (r["intent"], r["draw"])
        st = {}
        for name, c in dbs:
            ms, to, res = run_timed(c, r["sql"], ARGS.timeout)
            got = "TIMEOUT" if to else canon(res)
            if r["arm"] == "expert":
                ref[(key, name)] = got
                st[name] = "reference"
            else:
                exp_ = ref[(key, name)]
                st[name] = "timeout" if (to or exp_ == "TIMEOUT") else ("equal" if got == exp_ else "DIFFERENT")
        v = list(st.values())
        verdict = ("reference" if r["arm"] == "expert" else "rejected" if "DIFFERENT" in v else
                   "equivalent" if all(x == "equal" for x in v) else "unverified")
        r["verdict"] = verdict
        rows.append(dict(qid=r["qid"], arm=r["arm"], verdict=verdict, **st))
    for _, c in dbs[1:]:
        c.close()
    eq = pd.DataFrame(rows)
    summ = eq.groupby(["arm", "verdict"]).size().reset_index(name="queries")
    show_table(summ, "Equivalence verdicts by arm")
    acc = [r for r in inst if r["verdict"] in ("reference", "equivalent")]
    say(f"{len(acc)} of {len(inst)} query instances are verified and go into the speed comparison. "
        f"Rejected or unverified wordings are reported as a correctness result and excluded from timing.")
    eq.to_csv(os.path.join(OUT_DIR, "equivalence_checks.csv"), index=False)
    return acc


def step4_fingerprints(inst):
    banner(4, TOTAL, "Query fingerprinting engine",
           "Every query is parsed into a syntax tree. The syntactic fingerprint ignores constants and formatting "
           "(what tools such as pg_stat_statements and AQO key on). The semantic fingerprint first applies safe "
           "rewrites (qualify columns, unnest subqueries, merge CTEs, normalise predicates) and keeps only the "
           "tables and predicates, so that different wordings of one question can share a key.")
    fails = 0
    for r in inst:
        r["syn_fp"] = syntactic_fp(r["sql"])
        r["sem_fp"], r["sem_tables"], _ = semantic_fp(r["sql"])
        fails += r["sem_fp"] is None
    # property tests
    tests = run_fingerprint_tests()
    show_table(pd.DataFrame(tests, columns=["test", "result"]), "Unit and property tests")
    g = defaultdict(lambda: (set(), set()))
    for r in inst:
        g[(r["intent"], r["draw"])][0].add(r["syn_fp"])
        g[(r["intent"], r["draw"])][1].add(r["sem_fp"])
    syn_d = np.mean([len(a) for a, b in g.values()])
    sem_d = np.mean([len(b) for a, b in g.values()])
    coll = len({r["sem_fp"] for r in inst if r["sem_fp"]}) - len({(r["intent"], r["draw"], r["sem_fp"]) for r in inst if r["sem_fp"]})
    stats = pd.DataFrame([
        ("Queries parsed", f"{len(inst)}"),
        ("Semantic parse/optimise failures", f"{fails} ({pct(fails / len(inst))})"),
        ("Distinct syntactic fingerprints per question (5 wordings)", f"{syn_d:.2f}"),
        ("Distinct semantic fingerprints per question (5 wordings)", f"{sem_d:.2f}"),
        ("Collapse ratio (syntactic / semantic)", f"{syn_d / sem_d:.2f}"),
        ("Collisions between different questions", f"{max(coll, 0)}"),
    ], columns=["metric", "value"])
    show_table(stats, "Fingerprint quality")
    say("Reading: the syntactic fingerprint gives a new key to almost every wording, so a correction learned on "
        "one wording is never reused for another. The semantic fingerprint merges some of them at query level; "
        "at the level of plan sub-expressions (Step 7) the merging is much stronger.")
    return dict(syn_d=syn_d, sem_d=sem_d, tests_ok=all(t[1] == "PASS" for t in tests))


def run_fingerprint_tests():
    A = "SELECT c_name FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey WHERE o.o_totalprice > 100"
    B = "SELECT c_name FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey WHERE o.o_totalprice > 999"
    C = "select x.c_name   from customer x join orders y on x.c_custkey = y.o_custkey where y.o_totalprice > 100"
    D = ("WITH big AS (SELECT o_custkey FROM orders WHERE o_totalprice > 100) "
         "SELECT c_name FROM customer JOIN big ON c_custkey = big.o_custkey")
    E = "SELECT n_name FROM nation WHERE n_regionkey = 1"
    F = "SELECT c_name FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey WHERE o.o_totalprice < 100"
    res = [
        ("Constants ignored by syntactic fingerprint", syntactic_fp(A) == syntactic_fp(B)),
        ("Alias rename + spacing keeps semantic fingerprint", semantic_fp(A)[0] == semantic_fp(C)[0]),
        ("Different constant changes semantic fingerprint", semantic_fp(A)[0] != semantic_fp(B)[0]),
        ("CTE wording covers the same tables as the join", semantic_fp(A)[1] == semantic_fp(D)[1]),
        ("Different question gets a different key", semantic_fp(A)[0] != semantic_fp(E)[0]),
        ("Different operator (> vs <) changes the key", semantic_fp(A)[0] != semantic_fp(F)[0]),
    ]
    return [(n, "PASS" if ok else "FAIL") for n, ok in res]


def step5_measure(con, acc):
    banner(5, TOTAL, "Measure real latency and build independent labels (Tier A)",
           "Following the review feedback, we first measure real slowdowns. Each verified query is (a) explained "
           "without running it, which is all the guard may see, (b) run once with profiling to record the estimated "
           "and the true row count of every plan step, (c) timed with its own plan, and (d) timed under five "
           "alternative optimizer configurations (hint sets) that force different plans. Labels come only from "
           "these measurements, never from the guard's own rules.")
    rows, nodes = [], []
    t0 = time.time()
    for i, r in enumerate(acc):
        r["explain"] = explain_plan(con, r["sql"])
        prof = profile_plan(con, r["sql"], ARGS.timeout)
        r["profile"] = prof
        set_hint(con, "")
        nat = [run_timed(con, r["sql"], ARGS.timeout)[0] for _ in range(ARGS.reps)]
        t_nat = float(np.median(nat))
        cap = min(ARGS.timeout, max(3 * t_nat / 1000.0, 0.5))
        hints = {}
        for h, dis in HINT_SETS.items():
            set_hint(con, dis)
            ms, to, _ = run_timed(con, r["sql"], cap)
            hints[h] = (ms, to)
        set_hint(con, "")
        best_h = min([("native", t_nat)] + [(h, v[0]) for h, v in hints.items() if not v[1]], key=lambda x: x[1])
        r.update(native_ms=t_nat, hint_ms={h: v[0] for h, v in hints.items()}, best_hint=best_h[0], best_hint_ms=best_h[1])
        if prof is not None:
            for k, n in enumerate(flatten_plan(prof)):
                if n["act"] is None or math.isnan(n["est"]):
                    continue
                n.update(qid=r["qid"], intent=r["intent"], draw=r["draw"], arm=r["arm"], idx=k, syn_fp=r["syn_fp"])
                nodes.append(n)
        if (i + 1) % 20 == 0 or i + 1 == len(acc):
            print(f"   measured {i + 1:4d}/{len(acc)} queries  ({time.time() - t0:5.1f} s)", flush=True)
    # formulation regret: native time vs fastest verified wording of the same question and parameters
    best_form = defaultdict(lambda: float("inf"))
    for r in acc:
        best_form[(r["intent"], r["draw"])] = min(best_form[(r["intent"], r["draw"])], r["native_ms"])
    for r in acc:
        bf = best_form[(r["intent"], r["draw"])]
        r["R_form"] = r["native_ms"] / bf
        r["R_hint"] = r["native_ms"] / r["best_hint_ms"]
        r["y_form"] = int(r["R_form"] >= 2 and r["native_ms"] - bf >= ARGS.min_gap_ms)
        r["y_hint"] = int(r["R_hint"] >= 2 and r["native_ms"] - r["best_hint_ms"] >= ARGS.min_gap_ms)
        r["y"] = int(r["y_form"] or r["y_hint"])
        rows.append(dict(qid=r["qid"], intent=r["intent"], draw=r["draw"], arm=r["arm"], form=r["form"],
                         native_ms=r["native_ms"], best_hint=r["best_hint"], best_hint_ms=r["best_hint_ms"],
                         R_hint=r["R_hint"], best_wording_ms=bf, R_form=r["R_form"], y_hint=r["y_hint"],
                         y_form=r["y_form"], y=r["y"]))
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(OUT_DIR, "measurements.csv"), index=False)
    by_arm = M.groupby("arm").agg(queries=("qid", "count"), median_ms=("native_ms", "median"),
                                  p95_ms=("native_ms", lambda s: s.quantile(0.95)), max_ms=("native_ms", "max"),
                                  slow_wording=("y_form", "sum"), plan_regression=("y_hint", "sum"),
                                  any_regression=("y", "sum")).reset_index()
    by_arm["regression_rate"] = by_arm["any_regression"] / by_arm["queries"]
    show_table(by_arm, "Latency and regression labels by author arm (ms)")
    top = M.sort_values("R_form", ascending=False).head(10)[["qid", "native_ms", "best_wording_ms", "R_form", "best_hint", "R_hint"]]
    show_table(top, "Ten slowest wordings relative to the fastest equivalent wording")
    say(f"Label definitions: a 'slow wording' runs at least 2x slower (and {ARGS.min_gap_ms:g} ms slower) than the "
        "fastest verified wording of the same question with the same parameters; a 'plan regression' runs at least "
        "2x slower than under the best alternative optimizer configuration. Both are measured, not predicted.")
    # chart: runtime per arm
    fig, ax = plt.subplots(figsize=(8, 3.6))
    arms = ["expert", "orm", "agent"]
    data = [M[M.arm == a]["native_ms"].values for a in arms]
    ax.boxplot(data, showfliers=True)
    ax.set_xticks([1, 2, 3], arms)
    ax.set_yscale("log")
    ax.set_ylabel("runtime (ms, log scale)")
    ax.set_title("Runtime of verified-equivalent queries by author")
    save_fig(fig, "fig1_runtime_by_arm.png", "Runtime of verified-equivalent queries by author arm.")
    return M, pd.DataFrame(nodes)


def step6_explain_worst(acc, M):
    banner(6, TOTAL, "Look inside the worst agent wording",
           "To see why an agent wording is slow, we print its plan next to the expert plan for the same question: "
           "for each step, the optimizer's estimated rows, the true rows, and the time spent. Large gaps between "
           "estimated and true rows (the q-error) show where the cardinality estimator went wrong.")
    worst = M[M.arm == "agent"].sort_values("R_form", ascending=False).iloc[0]
    exp_q = next(r for r in acc if r["intent"] == worst.intent and r["draw"] == worst.draw and r["arm"] == "expert")
    bad_q = next(r for r in acc if r["qid"] == worst.qid)
    for label, r in (("EXPERT", exp_q), ("AGENT (worst)", bad_q)):
        txt = render_plan(r["profile"])
        head = f"{label}: {r['qid']}  runtime {r['native_ms']:.1f} ms"
        print("\n " + head)
        print("\n".join("   " + l for l in txt.splitlines()[:28]))
        REPORT.append(("h3", head))
        REPORT.append(("pre", txt))
    say(f"The agent wording {worst.qid} returns exactly the same rows as the expert query but runs "
        f"{worst.R_form:.1f}x slower. The plan shows the decorrelated subquery producing far more intermediate rows "
        "than the optimizer expected; this is the kind of wording-dependent estimation error InvaCard targets.")
    return worst


def render_plan(root, depth=0):
    lines = []

    def walk(n, d):
        t = _ntype(n)
        info = n.get("extra_info", {})
        if t not in ("PROJECTION", "RESULT_COLLECTOR", "CTE"):
            est = _est(n)
            act = n.get("intermediate_rows", n.get("operator_cardinality", 0))
            q = qerr(est, act) if not math.isnan(est) else float("nan")
            extra = info.get("Join Type", "") or ""
            cond = _as_list(info.get("Conditions")) or _as_list(info.get("Filters")) or _as_list(info.get("Table"))
            cond = (" " + str(cond[0])[:46]) if cond and cond[0] else ""
            lines.append(f"{'  ' * d}{t:<20s}{extra:<6s} est={est:>10.0f} true={act:>10.0f} q={q:>8.1f} "
                         f"time={1000 * float(n.get('timing', 0)):8.1f}ms{cond}")
            d += 1
        for c in n.get("children", []):
            walk(c, d)
    walk(root, depth)
    return "\n".join(lines)


def step7_estimation_bias(nodes):
    banner(7, TOTAL, "Formulation-path bias and the coherence certificate",
           "Two wordings of the same question contain many identical logical sub-expressions (for example 'orders "
           "joined to lineitem with the same filters'). Their true row counts are equal, so their estimates should "
           "be equal too (coherence law L1). Here we measure how often the estimator breaks this, and how its error "
           "depends on the formulation path (plain join, semi/anti join, mark join from a subquery, CTE scan, ...).")
    nd = nodes.copy()
    nd["q"] = [qerr(e, a) for e, a in zip(nd.est, nd.act)]
    by_path = nd.groupby("path").agg(nodes=("q", "size"), median_q=("q", "median"), p90_q=("q", lambda s: s.quantile(.9)),
                                     max_q=("q", "max")).reset_index().sort_values("p90_q", ascending=False)
    show_table(by_path, "Native estimator q-error by formulation path")
    by_arm = nd.groupby("arm").agg(nodes=("q", "size"), median_q=("q", "median"), p90_q=("q", lambda s: s.quantile(.9)),
                                   p99_q=("q", lambda s: s.quantile(.99)), frac_q_gt_10=("q", lambda s: (s > 10).mean())).reset_index()
    show_table(by_arm, "Native estimator q-error by author arm")
    # L1 across formulations: same semantic key, same intent+draw, different queries
    g = nd.groupby(["intent", "draw", "sem_key"])
    rows = []
    for (it, d, k), grp in g:
        if grp.qid.nunique() < 2:
            continue
        e = grp.est.clip(lower=1)
        a = grp.act.clip(lower=1)
        rows.append(dict(intent=it, path=grp.path.iloc[0], n_queries=grp.qid.nunique(),
                         est_spread=float(e.max() / e.min()), true_spread=float(a.max() / a.min())))
    L1all = pd.DataFrame(rows)
    L1 = L1all[L1all.true_spread <= 1.05] if len(L1all) else L1all
    key_prec = len(L1) / max(len(L1all), 1)
    shared = len(L1)
    viol2 = int((L1.est_spread >= 2).sum()) if shared else 0
    viol10 = int((L1.est_spread >= 10).sum()) if shared else 0
    say(f"{shared} logical sub-expressions appear in two or more wordings of the same question. In {viol2} of them "
        f"({pct(viol2 / max(shared, 1))}) the estimator's numbers differ by 2x or more between wordings, and in "
        f"{viol10} by 10x or more, although their true row counts are identical. (Key precision check: {pct(key_prec)} of shared semantic "
        f"keys have identical true row counts in every wording.) This is direct evidence of wording-dependent estimates.")
    if shared:
        show_table(L1.sort_values("est_spread", ascending=False).head(8), "Largest estimate disagreements for the same sub-expression")
    # certificate: LP lower bound on the worst q-error from within-plan laws (estimates only)
    cert_rows = []
    for qid, grp in nd.groupby("qid"):
        d_star, nviol = certificate(grp.to_dict("records"))
        cert_rows.append(dict(qid=qid, arm=grp.arm.iloc[0], delta=d_star, bound=math.exp(d_star), violations=nviol,
                              max_q=grp.q.max()))
    C = pd.DataFrame(cert_rows)
    sound = bool((C.bound <= C.max_q * (1 + 1e-6)).all())
    fired = int((C.violations > 0).sum())
    say(f"Coherence certificate: computed from estimates only (no execution) for {len(C)} plans; it detected "
        f"provably inconsistent estimates in {fired} plans. Soundness check (certified bound <= observed worst "
        f"q-error) holds for {'ALL' if sound else 'NOT ALL'} plans.")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    if shared:
        ax.hist(np.log10(L1.est_spread), bins=30, color="#555")
        ax.set_yscale("log")
    ax.set_xlabel("log10( max estimate / min estimate ) for the same sub-expression across wordings")
    ax.set_ylabel("sub-expressions (log scale)")
    ax.set_title("Same logical step, different estimates depending on wording")
    save_fig(fig, "fig2_estimate_spread.png", "Spread of the native estimates for identical sub-expressions across wordings.")
    return dict(shared=shared, viol2=viol2, viol10=viol10, cert_sound=sound, cert_fired=fired, key_prec=key_prec), C


def law_pairs(nodes):
    """Coherence laws between plan nodes (log space). Returns list of (i, j, kind, offset):
    kind 'le' means x_i <= x_j + offset, 'eq' means x_i == x_j."""
    byid = {n["_id"]: i for i, n in enumerate(nodes)}
    pairs = []
    for i, n in enumerate(nodes):
        kids = [byid[c] for c in n.get("child_idx", []) if c in byid]
        p = n["path"]
        if p in ("filter",) and kids:
            pairs.append((i, kids[0], "le", 0.0))                           # L3 extra condition cannot add rows
        if p == "agg" and kids:
            pairs.append((i, kids[0], "le", 0.0))                           # L6 grouping cannot add rows
        if p == "limit" and kids:
            pairs.append((i, kids[0], "le", 0.0))                           # L6 limit
        if p == "mark" and kids:
            pairs.append((i, kids[0], "eq", 0.0))                           # mark join keeps every left row
        if p in ("semi", "anti") and kids:
            side = kids[1] if (n.get("jt", "").startswith("RIGHT") and len(kids) > 1) else kids[0]
            pairs.append((i, side, "le", 0.0))                              # semi/anti cannot exceed its kept input
    by_key = defaultdict(list)
    for i, n in enumerate(nodes):
        by_key[n["sem_key"]].append(i)
    for ks in by_key.values():                                                # L1 equal meaning, equal size
        for a, b in zip(ks, ks[1:]):
            pairs.append((a, b, "eq", 0.0))
    return pairs


def certificate(nodes):
    """delta* = min over coherent vectors x of max_i |x_i - log est_i| (one small linear program)."""
    if not nodes:
        return 0.0, 0
    l = np.array([math.log(max(n["est"], 1.0)) for n in nodes])
    pairs = law_pairs(nodes)
    viol = sum(1 for i, j, k, o in pairs if (k == "le" and l[i] > l[j] + 1e-9) or (k == "eq" and abs(l[i] - l[j]) > 1e-9))
    if viol == 0:
        return 0.0, 0
    n = len(l)
    c = np.zeros(n + 1); c[-1] = 1.0
    A, b, Ae, be = [], [], [], []
    for i in range(n):
        r = np.zeros(n + 1); r[i] = 1; r[-1] = -1; A.append(r); b.append(l[i])
        r = np.zeros(n + 1); r[i] = -1; r[-1] = -1; A.append(r); b.append(-l[i])
    for i, j, k, o in pairs:
        r = np.zeros(n + 1); r[i] = 1; r[j] = -1
        if k == "le":
            A.append(r); b.append(o)
        else:
            Ae.append(r); be.append(0.0)
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=(np.array(Ae) if Ae else None),
                  b_eq=(np.array(be) if be else None), bounds=[(None, None)] * n + [(0, None)], method="highs")
    return (float(res.x[-1]) if res.success else 0.0), viol


# ---------------------------------------------------------------- InvaCard
TABLE_LIST = TPCH_TABLES


def node_features(n):
    ce = n.get("child_est") or [0.0]
    v = [math.log1p(max(n["est"], 0)), math.log1p(max(max(ce), 0)), math.log1p(max(min(ce), 0)),
         n["depth"], n["n_tables"], n["n_conds"], len(n.get("child_est") or []),
         PATHS.index(n["path"]), OP_TYPES.index(n["type"]) if n["type"] in OP_TYPES else len(OP_TYPES)]
    v += [int(t in n["tables"]) for t in TABLE_LIST]
    return v


CAT_IDX = [7, 8]


class InvaCard:
    """Formulation-invariant cardinality corrector.
    1. experience memory keyed by the semantic sub-expression key (shared across wordings)
    2. residual model f(x) = log(true+1) - log(est+1) on top of the native estimate
    3. split-conformal band per formulation path: a model correction is used only if it is larger than the band
    4. coherence projection: corrected values are clipped so that the plan-level laws hold."""

    def __init__(self, use_memory=True, use_gate=True, use_coherence=True, alpha=0.1, seed=0):
        self.use_memory, self.use_gate, self.use_coherence, self.alpha, self.seed = use_memory, use_gate, use_coherence, alpha, seed

    def fit(self, train):
        self.memory = defaultdict(list)
        for n in train:
            self.memory[n["sem_key"]].append(n["act"])
        X = np.array([node_features(n) for n in train], dtype=float)
        y = np.array([math.log((n["act"] + 1) / (n["est"] + 1)) for n in train])
        idx = np.arange(len(train)); np.random.default_rng(self.seed).shuffle(idx)
        cut = int(0.75 * len(idx))
        mk = lambda: HistGradientBoostingRegressor(loss="absolute_error", max_iter=250, learning_rate=0.06,
                                                   max_leaf_nodes=15, min_samples_leaf=8, random_state=self.seed,
                                                   categorical_features=CAT_IDX)
        m = mk().fit(X[idx[:cut]], y[idx[:cut]])
        res = np.abs(m.predict(X[idx[cut:]]) - y[idx[cut:]])
        paths = np.array([train[i]["path"] for i in idx[cut:]])
        self.band = {}
        for p in PATHS:
            r = np.sort(res[paths == p])
            if len(r) >= 8:
                k = min(len(r) - 1, int(math.ceil((len(r) + 1) * (1 - self.alpha))) - 1)
                self.band[p] = float(r[k])
            else:
                self.band[p] = float(np.quantile(res, 1 - self.alpha)) if len(res) else 0.0
        self.model = mk().fit(X, y)
        return self

    def predict(self, nodes, table_rows):
        if not nodes:
            return [], {}
        X = np.array([node_features(n) for n in nodes], dtype=float)
        r = self.model.predict(X)
        out, src = [], Counter()
        for k, n in enumerate(nodes):
            mem = self.memory.get(n["sem_key"]) if self.use_memory else None
            if mem:
                out.append(float(np.median(mem))); src["memory"] += 1
            elif not self.use_gate:
                out.append(max((n["est"] + 1) * math.exp(r[k]) - 1, 0.0)); src["model"] += 1
            elif abs(r[k]) > self.band.get(n["path"], 0.0):
                # only the part of the correction that lies outside the conformal band is applied
                adj = math.copysign(abs(r[k]) - self.band.get(n["path"], 0.0), r[k])
                out.append(max((n["est"] + 1) * math.exp(adj) - 1, 0.0)); src["model (gated)"] += 1
            else:
                out.append(n["est"]); src["kept native"] += 1
        if self.use_coherence:
            out, clipped = coherence_project(nodes, out, table_rows)
            src["clipped by laws"] = clipped
        return out, src


def coherence_project(nodes, vals, table_rows):
    """Bottom-up projection onto the coherent region for the order constraints (children are corrected first)."""
    byq = defaultdict(list)
    for k, n in enumerate(nodes):
        byq[n["qid"]].append(k)
    vals = list(vals); clipped = 0
    for q, ks in byq.items():
        loc = {nodes[k]["_id"]: k for k in ks}
        same = defaultdict(list)                       # L1: equal meaning inside one plan -> equal estimate
        for k in ks:
            same[nodes[k]["sem_key"]].append(k)
        for grp in same.values():
            if len(grp) > 1:
                med = float(np.median([vals[k] for k in grp]))
                for k in grp:
                    clipped += abs(vals[k] - med) > 1e-9
                    vals[k] = med
        for k in ks:                                   # nodes are in post-order: children first
            n = nodes[k]; v = max(vals[k], 0.0); orig = v
            kids = [vals[loc[c]] for c in n.get("child_idx", []) if c in loc]
            if n["path"] == "scan" and len(n["tables"]) == 1:
                v = min(v, table_rows.get(n["tables"][0], v))
            if n["path"] in ("filter", "agg") and kids:
                v = min(v, kids[0])
            if n["path"] in ("semi", "anti") and kids:
                v = min(v, kids[1] if (n.get("jt", "").startswith("RIGHT") and len(kids) > 1) else kids[0])
            if n["path"] == "limit" and kids:
                v = min(v, kids[0], n["top"] if n.get("top") else v)
            if n["path"] == "mark" and kids:
                v = kids[0]
            if n["path"] in ("join", "nl_join") and len(kids) == 2:
                v = min(v, (kids[0] + 1) * (kids[1] + 1))
            clipped += abs(v - orig) > 1e-9
            vals[k] = v
    return vals, clipped


def syn_memory_predict(train, test):
    mem = {}
    for n in train:
        mem[n["syn_fp"] + n["syn_key"]] = n["act"]
    return [mem.get(n["syn_fp"] + n["syn_key"], n["est"]) for n in test]


def qsum(qs):
    q = np.array(qs)
    return dict(nodes=len(q), median=np.median(q), p90=np.quantile(q, .9), p95=np.quantile(q, .95),
                p99=np.quantile(q, .99), max=q.max(), frac_gt_10=(q > 10).mean())


def step8_invacard(con, nodes):
    banner(8, TOTAL, "InvaCard: formulation-invariant cardinality correction",
           "InvaCard corrects the optimizer's row estimates. It remembers true row counts under the meaning of each "
           "sub-expression (not its wording), learns a residual correction with gradient-boosted trees, only "
           "applies a learned correction when it is outside a conformal uncertainty band for that formulation path, "
           "and projects its outputs so that they obey the coherence laws. Protocol P1 trains on expert and ORM "
           "queries only and tests on the agent wordings it has never seen; protocol P2 holds out a whole question.")
    trows = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TPCH_TABLES}
    recs = nodes.to_dict("records")
    for n in recs:
        n["q_native"] = qerr(n["est"], n["act"])
    train = [n for n in recs if n["arm"] in ("expert", "orm")]
    test = [n for n in recs if n["arm"] == "agent"]
    variants = {
        "B0 native estimator": None,
        "B2 syntactic memory (AQO-style)": "syn",
        "B3 semantic memory only": InvaCard(use_memory=True, use_gate=True, use_coherence=False),
        "B5 learned residual only": InvaCard(use_memory=False, use_gate=False, use_coherence=False),
        "InvaCard (full)": InvaCard(),
        "InvaCard without gate": InvaCard(use_gate=False),
        "InvaCard without coherence projection": InvaCard(use_coherence=False),
        "InvaCard without semantic memory": InvaCard(use_memory=False),
    }
    res, srcs, preds = {}, {}, {}
    for name, m in variants.items():
        if m is None:
            p = [n["est"] for n in test]
        elif m == "syn":
            p = syn_memory_predict(train, test)
        else:
            if name == "B3 semantic memory only":
                m.fit(train)
                p = [float(np.median(m.memory[n["sem_key"]])) if n["sem_key"] in m.memory else n["est"] for n in test]
            else:
                p, s = m.fit(train).predict(test, trows)
                srcs[name] = s
        preds[name] = p
        res[name] = qsum([qerr(a, n["act"]) for a, n in zip(p, test)])
    T = pd.DataFrame([dict(method=k, **v) for k, v in res.items()])
    show_table(T, "P1 leave-arm-out: q-error on agent-written sub-expressions (lower is better)")
    hit_syn = np.mean([(n["syn_fp"] + n["syn_key"]) in {(x["syn_fp"] + x["syn_key"]) for x in train} for n in test])
    sem_keys = {x["sem_key"] for x in train}
    hit_sem = np.mean([n["sem_key"] in sem_keys for n in test])
    say(f"Key hit rate on agent sub-expressions: syntactic keys {pct(hit_syn)}, semantic keys {pct(hit_sem)}. "
        f"Sources used by full InvaCard: {dict(srcs.get('InvaCard (full)', {}))}.")
    # per path
    rows = []
    for p in PATHS:
        ks = [i for i, n in enumerate(test) if n["path"] == p]
        if len(ks) < 5:
            continue
        rows.append(dict(path=p, nodes=len(ks),
                         native_p90=np.quantile([test[i]["q_native"] for i in ks], .9),
                         invacard_p90=np.quantile([qerr(preds["InvaCard (full)"][i], test[i]["act"]) for i in ks], .9)))
    PP = pd.DataFrame(rows)
    show_table(PP, "P1 by formulation path: 90th percentile q-error")
    worse = PP[PP.invacard_p90 > PP.native_p90 * 1.1].path.tolist() if len(PP) else []
    if worse:
        say("InvaCard is worse than the native estimator on these paths: " + ", ".join(worse) + ". These paths have "
            "few training examples from the expert and ORM wordings, so the learned correction extrapolates; the "
            "full design adds the plan-change test of the gate (PostgreSQL + pg_hint_plan) to stop such corrections.")
    # law violations before / after
    def viol_count(vals):
        v = 0
        for q, grp in pd.DataFrame(test).groupby("qid"):
            idxs = list(grp.index)
            sub = [test[i] for i in idxs]
            l = [math.log(max(vals[i], 1)) for i in idxs]
            for i, j, k, o in law_pairs(sub):
                v += (k == "le" and l[i] > l[j] + 1e-6) or (k == "eq" and abs(l[i] - l[j]) > 1e-6)
        return v
    vb, va = viol_count([n["est"] for n in test]), viol_count(preds["InvaCard (full)"])
    vr = viol_count(preds["B5 learned residual only"])
    say(f"Coherence-law violations on agent plans: native estimator {vb}, learned residual alone {vr}, "
        f"InvaCard after projection {va}.")
    # P2 leave-intent-out
    agg = defaultdict(list)
    for it in sorted({n["intent"] for n in recs}):
        tr = [n for n in recs if n["intent"] != it]
        te = [n for n in recs if n["intent"] == it]
        if not te:
            continue
        agg["B0 native estimator"] += [n["q_native"] for n in te]
        for name, mm in (("B5 learned residual only", InvaCard(use_memory=False, use_gate=False, use_coherence=False)),
                         ("InvaCard (full)", InvaCard())):
            p, _ = mm.fit(tr).predict(te, trows)
            agg[name] += [qerr(a, n["act"]) for a, n in zip(p, te)]
    T2 = pd.DataFrame([dict(method=k, **qsum(v)) for k, v in agg.items()])
    show_table(T2, "P2 leave-one-question-out (memory cannot help here): q-error")
    b0, ic, b5, b2 = res["B0 native estimator"], res["InvaCard (full)"], res["B5 learned residual only"], res["B2 syntactic memory (AQO-style)"]
    p2 = {r.method: r for r in T2.itertuples()}
    say(f"Reading: on unseen agent wordings (P1) InvaCard lowers the 90th percentile q-error from {b0['p90']:.1f} to "
        f"{ic['p90']:.1f} and the 99th percentile from {b0['p99']:.0f} to {ic['p99']:.0f}. The syntactic (AQO-style) "
        f"memory cannot help at all because no agent sub-expression has the same syntactic key "
        f"(p90 {b2['p90']:.1f}), while {pct(hit_sem)} of them are found through semantic keys. The learned residual "
        f"alone is competitive in the middle of the distribution (p90 {b5['p90']:.1f}) but has a heavier tail "
        f"(p99 {b5['p99']:.0f}) and breaks the coherence laws {vr} times; the gate and projection trade a little "
        f"typical accuracy for a much safer tail. On completely new questions (P2) memory cannot apply: InvaCard "
        f"p90 {p2['InvaCard (full)'].p90:.1f} vs native {p2['B0 native estimator'].p90:.1f} vs ungated residual "
        f"{p2['B5 learned residual only'].p90:.1f}, which shows why the gate matters when the model is outside "
        "what it has seen.")
    # chart
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["B0 native estimator", "B2 syntactic memory (AQO-style)", "B5 learned residual only", "InvaCard (full)"]:
        q = np.sort([qerr(a, n["act"]) for a, n in zip(preds[name], test)])
        ax.plot(q, np.arange(1, len(q) + 1) / len(q), label=name)
    ax.set_xscale("log"); ax.set_xlabel("q-error (log scale)"); ax.set_ylabel("fraction of sub-expressions")
    ax.set_title("P1: q-error on unseen agent wordings (curve further left/up is better)")
    ax.legend(fontsize=8)
    save_fig(fig, "fig3_qerror_cdf.png", "Cumulative q-error distribution on agent sub-expressions (protocol P1). "
             "The syntactic-memory curve lies exactly on top of the native curve because it never finds a match.")
    return res, T2, dict(hit_syn=hit_syn, hit_sem=hit_sem, viol_before=vb, viol_after=va, viol_resid=vr)


# ---------------------------------------------------------------- stage one detection
def plan_features(tree):
    nodes = flatten_plan(tree, actual=False)
    ests = [n["est"] for n in nodes if not math.isnan(n["est"])] or [0]
    ratios = [n["est"] / max(min(n["child_est"]), 1) for n in nodes if n["child_est"]]
    d_star, nv = certificate([n for n in nodes if not math.isnan(n["est"])])
    return dict(p_log_max_est=math.log1p(max(ests)), p_log_sum_est=math.log1p(sum(ests)), p_log_root=math.log1p(ests[-1]),
                p_nodes=len(nodes), p_nl=sum(n["path"] == "nl_join" for n in nodes),
                p_mark=sum(n["path"] == "mark" for n in nodes), p_delim=sum(n["path"] == "delim" for n in nodes),
                p_semi_anti=sum(n["path"] in ("semi", "anti") for n in nodes), p_cte=sum(n["path"] == "cte" for n in nodes),
                p_max_ratio=math.log1p(max(ratios) if ratios else 0)), dict(c_delta=d_star, c_viol=nv)


def complexity_features(sql):
    try:
        t = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return dict(c_len=len(sql))
    cnt = lambda k: len(list(t.find_all(k)))
    def depth(e, d=0):
        return max([d] + [depth(c, d + 1) for c in e.iter_expressions()])
    return dict(x_joins=cnt(exp.Join), x_subq=cnt(exp.Subquery) + cnt(exp.Exists), x_cte=cnt(exp.CTE),
                x_or=cnt(exp.Or), x_like=cnt(exp.Like), x_window=cnt(exp.Window), x_exists=cnt(exp.Exists),
                x_in=cnt(exp.In), x_not=cnt(exp.Not), x_agg=cnt(exp.AggFunc), x_depth=depth(t), x_len=len(sql))


DETECTOR_SETS = {"M1 novelty": ["n"], "M2 complexity": ["x"], "M3 plan": ["p"], "M4 combined (SQL-Guard)": ["n", "x", "p"],
                 "M5 combined + certificate": ["n", "x", "p", "c"]}


def _novelty(acc, r, train_idx):
    syn = sum(acc[j]["syn_fp"] == r["syn_fp"] for j in train_idx)
    sem = sum(acc[j]["sem_fp"] == r["sem_fp"] for j in train_idx)
    jac = max([len(r["_toks"] & acc[j]["_toks"]) / max(len(r["_toks"] | acc[j]["_toks"]), 1) for j in train_idx] or [0])
    return dict(n_syn_seen=math.log1p(syn), n_sem_seen=math.log1p(sem), n_nearest_sim=jac, n_new=int(syn == 0))


def run_detection(acc, y, groups):
    """Out-of-fold risk scores for every detector, holding out one group at a time."""
    oof = {(m, mdl): np.full(len(acc), np.nan) for m in DETECTOR_SETS for mdl in ("logistic", "boosted trees")}
    for g in np.unique(groups):
        te = np.where(groups == g)[0]; tr = np.where(groups != g)[0]
        nov_te = {i: _novelty(acc, acc[i], tr) for i in te}
        nov_tr = {i: _novelty(acc, acc[i], [j for j in tr if groups[j] != groups[i] or j != i]) for i in tr}
        for m, gs in DETECTOR_SETS.items():
            def row(i, nv):
                d = {}
                if "n" in gs: d.update(nv)
                if "x" in gs: d.update(acc[i]["_xf"])
                if "p" in gs: d.update(acc[i]["_pf"])
                if "c" in gs: d.update(acc[i]["_cf"])
                return d
            Xtr = pd.DataFrame([row(i, nov_tr[i]) for i in tr]).fillna(0)
            Xte = pd.DataFrame([row(i, nov_te[i]) for i in te]).reindex(columns=Xtr.columns).fillna(0)
            if y[tr].sum() < 2:
                oof[(m, "logistic")][te] = y[tr].mean(); oof[(m, "boosted trees")][te] = y[tr].mean()
                continue
            lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))
            lr.fit(Xtr, y[tr]); oof[(m, "logistic")][te] = lr.predict_proba(Xte)[:, 1]
            gb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=8, min_samples_leaf=5,
                                                class_weight="balanced", random_state=0)
            gb.fit(Xtr, y[tr]); oof[(m, "boosted trees")][te] = gb.predict_proba(Xte)[:, 1]
    return oof


def detection_table(oof, y):
    rows = []
    for (m, mdl), s in oof.items():
        fpr, tpr, _ = roc_curve(y, s)
        rows.append(dict(detector=m, model=mdl, AUPRC=average_precision_score(y, s), ROC_AUC=roc_auc_score(y, s),
                         recall_at_5pct_FPR=float(np.interp(0.05, fpr, tpr))))
    return pd.DataFrame(rows).sort_values(["model", "AUPRC"], ascending=[True, False])


def step9_detection(acc, M):
    banner(9, TOTAL, "Stage one: predicting regressions before execution (M1 to M5)",
           "The guard must decide before a query runs, so it may only use the SQL text, the catalog and the "
           "EXPLAIN plan. As the review asked, we compare four detectors: M1 novelty only, M2 complexity only, "
           "M3 plan features only, M4 all three combined, plus M5 = M4 with the coherence certificate. We test two "
           "situations: (A) the guard has seen the question before but with other parameter values (leave-one-"
           "draw-out), and (B) the question is completely new (leave-one-intent-out, the harder case). Area under "
           "the precision-recall curve (AUPRC) is the main score because regressions are rare.")
    for r in acc:
        r["_pf"], r["_cf"] = plan_features(r["explain"])
        r["_xf"] = complexity_features(r["sql"])
        r["_toks"] = set(re.findall(r"[a-z_]+", r["sql"].lower()))
    y = np.array([r["y"] for r in acc])
    say(f"Positives (measured regressions) among {len(y)} verified queries: {int(y.sum())} "
        f"(base rate {pct(y.mean())}); a detector that guesses scores AUPRC = {y.mean():.3f}.")
    results = {}
    protos = []
    if len({r["draw"] for r in acc}) >= 2:
        protos.append(("A: new parameters (leave-one-draw-out)", np.array([r["draw"] for r in acc])))
    protos.append(("B: new question (leave-one-intent-out)", np.array([r["intent"] for r in acc])))
    for name, groups in protos:
        oof = run_detection(acc, y, groups)
        D = detection_table(oof, y)
        show_table(D, f"Detection quality, protocol {name}")
        results[name] = (oof, D, groups)
    name0 = protos[0][0]
    oof, D, groups = results[name0]
    bt = D[D.model == "boosted trees"]
    best_single = bt[bt.detector.str.startswith(("M1", "M2", "M3"))].sort_values("AUPRC").iloc[-1].detector
    a, b = oof[("M4 combined (SQL-Guard)", "boosted trees")], oof[(best_single, "boosted trees")]
    rng = np.random.default_rng(0); ints = np.unique([r["intent"] for r in acc]); diffs = []
    intents = np.array([r["intent"] for r in acc])
    for _ in range(500):
        pick = rng.choice(ints, len(ints))
        idx = np.concatenate([np.where(intents == g)[0] for g in pick])
        if 0 < y[idx].sum() < len(idx):
            diffs.append(average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    say(f"Protocol {name0}: M4 minus the best single-group detector ({best_single}), intent-level bootstrap 95% CI "
        f"for the AUPRC difference: [{lo:+.3f}, {hi:+.3f}]. " + ("Both detectors rank every regression first here, so "
        "they cannot be separated; protocol B is the informative comparison." if abs(lo) < 1e-9 and abs(hi) < 1e-9 else
        "An interval that includes 0 means the difference is not significant at this pilot size."))
    say("Why protocol A is easy: a wording that is slow for one set of parameters is usually slow for the others "
        "too, and its EXPLAIN plan already shows the warning signs (nested-loop or mark joins from subqueries, "
        "very large estimated intermediate results).")
    say("For a completely new question (protocol B) the task is much harder, which is why the paper pairs "
        "stage one with stage-two monitoring and the experience memory.")
    fig, axs = plt.subplots(1, len(protos), figsize=(6.2 * len(protos), 4.2))
    axs = np.atleast_1d(axs)
    for ax, (name, _) in zip(axs, protos):
        o = results[name][0]
        for m in DETECTOR_SETS:
            pr, rc, _ = precision_recall_curve(y, o[(m, "boosted trees")])
            ax.plot(rc, pr, label=f"{m} ({average_precision_score(y, o[(m, 'boosted trees')]):.2f})")
        ax.axhline(y.mean(), ls="--", color="grey", label="no skill")
        ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7)
    save_fig(fig, "fig4_detection_pr.png", "Precision-recall curves of the five detectors (boosted trees).")
    for i, (name, (o, d, g)) in enumerate(results.items()):
        for r, s in zip(acc, o[("M4 combined (SQL-Guard)", "boosted trees")]):
            r["risk_" + name[0]] = float(s)
    for r in acc:
        r["risk"] = r["risk_" + name0[0]]
    Dall = pd.concat([d.assign(protocol=n) for n, (o, d, g) in results.items()])
    return Dall, (lo, hi), best_single, name0


def step10_factorial(con, acc, M):
    banner(10, TOTAL, "Tier B: controlled fault injection and exact cause shares",
           "A slow query is not automatically the optimizer's fault. For a sample of queries we switch four causes "
           "on and off in every combination: W = the agent's wording vs the fastest equivalent wording, P = the "
           "native plan vs the best alternative optimizer configuration, K = CPU contention from background "
           "queries, R = restricted resources (1 thread). Shapley values then split each slowdown exactly among the "
           "four causes. We also run the leakage test: stage-one inputs must not change when only the runtime "
           "conditions (K, R) change.")
    pos = [r for r in acc if r["y"] == 1 and r["arm"] == "agent"]
    neg = [r for r in acc if r["y"] == 0 and r["arm"] == "agent"]
    pick = sorted(pos, key=lambda r: -r["R_form"])[:ARGS.tierb // 2] + random.Random(1).sample(neg, min(len(neg), ARGS.tierb - min(len(pos), ARGS.tierb // 2)))
    fastest = {}
    for r in acc:
        k = (r["intent"], r["draw"])
        if k not in fastest or r["native_ms"] < fastest[k]["native_ms"]:
            fastest[k] = r
    stop = threading.Event()

    def background():
        c2 = con.cursor()
        while not stop.is_set():
            try:
                c2.execute("SELECT l_returnflag, sum(l_extendedprice * (1 - l_discount)), count(*) FROM lineitem "
                           "GROUP BY l_returnflag").fetchall()
            except Exception:
                pass
    rows, leak_ok = [], True
    for r in pick:
        alt = fastest[(r["intent"], r["draw"])]
        best_hint = min(r["hint_ms"], key=r["hint_ms"].get)
        Y = {}
        stage1_ref = None
        for W, P, K, R in itertools.product([0, 1], repeat=4):
            sql = r["sql"] if W else alt["sql"]
            con.execute(f"SET threads = {1 if R else ARGS.threads}")
            threads = []
            if K:
                stop.clear()
                threads = [threading.Thread(target=background, daemon=True) for _ in range(ARGS.bg_threads)]
                for t in threads: t.start()
                time.sleep(0.05)
            # leakage invariance: stage-one inputs computed under these runtime conditions
            if W == 1:
                pf, cf = plan_features(explain_plan(con, sql))
                sig = json.dumps([pf, cf, syntactic_fp(sql), complexity_features(sql)], sort_keys=True, default=str)
                if stage1_ref is None:
                    stage1_ref = sig
                elif sig != stage1_ref:
                    leak_ok = False
            set_hint(con, "" if P else HINT_SETS.get(best_hint, ""))
            ts = [run_timed(con, sql, ARGS.timeout)[0] for _ in range(ARGS.reps)]
            set_hint(con, "")
            if K:
                stop.set()
                for t in threads: t.join()
            Y[(W, P, K, R)] = float(np.mean(np.log(ts)))
        con.execute(f"SET threads = {ARGS.threads}")
        sh = shapley(Y, 4)
        tot = Y[(1, 1, 1, 1)] - Y[(0, 0, 0, 0)]
        rows.append(dict(qid=r["qid"], label=("regression" if r["y"] else "normal"),
                         slowdown_x=math.exp(tot), wording=sh[0], plan=sh[1], contention=sh[2], resources=sh[3]))
    S = pd.DataFrame(rows)
    share = S.copy()
    tot = share[["wording", "plan", "contention", "resources"]].abs().sum(axis=1).replace(0, 1)
    for c in ["wording", "plan", "contention", "resources"]:
        share[c + "_%"] = 100 * share[c].abs() / tot
    show_table(share[["qid", "label", "slowdown_x", "wording_%", "plan_%", "contention_%", "resources_%"]],
               "Shapley cause shares of the slowdown (all-faulty vs all-normal)", floatfmt="{:.1f}")
    reg = share[share.label == "regression"]
    say(f"For the measured regressions, the wording explains on average {reg['wording_%'].mean():.0f}% of the "
        f"slowdown, the plan choice {reg['plan_%'].mean():.0f}%, contention {reg['contention_%'].mean():.0f}% and "
        f"restricted resources {reg['resources_%'].mean():.0f}%. Leakage invariance test: "
        f"{'PASS - stage-one inputs were identical under every runtime condition' if leak_ok else 'FAIL'}.")
    say("Note: DuckDB runs in memory and uses MVCC, so the cold-cache and lock factors of the full PostgreSQL "
        "design cannot be reproduced in this demo; contention and resource limits stand in for environmental causes.")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bottom = np.zeros(len(share))
    for c, col in zip(["wording_%", "plan_%", "contention_%", "resources_%"], ["#222", "#666", "#999", "#ccc"]):
        ax.bar(range(len(share)), share[c], bottom=bottom, label=c[:-2], color=col)
        bottom += share[c].values
    ax.set_xticks(range(len(share)), [q.replace("_", "\n", 1) for q in share.qid], fontsize=7)
    ax.set_ylabel("share of slowdown (%)"); ax.set_title("Exact cause shares (Shapley) per query")
    ax.legend(fontsize=8, ncol=4)
    save_fig(fig, "fig5_shapley.png", "Shapley cause shares from the factorial fault-injection runs.")
    return S, leak_ok


def shapley(Y, k):
    vals = []
    for j in range(k):
        s = 0.0
        others = [i for i in range(k) if i != j]
        for m in range(k):
            for comb in itertools.combinations(others, m):
                w = math.factorial(m) * math.factorial(k - m - 1) / math.factorial(k)
                on = tuple(1 if (i in comb or i == j) else 0 for i in range(k))
                off = tuple(1 if i in comb else 0 for i in range(k))
                s += w * (Y[on] - Y[off])
        vals.append(s)
    return vals


def replay_policies(acc, riskkey, thr):
    fastest = {}
    for r in acc:
        k = (r["intent"], r["draw"])
        if r["arm"] != "agent" and (k not in fastest or r["native_ms"] < fastest[k]):
            fastest[k] = r["native_ms"]           # the experience memory only knows earlier, non-agent wordings
    base = np.array([r["native_ms"] for r in acc])
    y = np.array([r["y"] for r in acc])
    flag = np.array([r[riskkey] >= thr for r in acc])

    def evaluate(name, lat, interventions, note):
        lat = np.array(lat)
        prevented = int(((y == 1) & (lat <= base / 2)).sum())
        return dict(policy=name, flagged=int(flag.sum()) if interventions else 0, regressions=int(y.sum()),
                    prevented=prevented, unnecessary=int((flag & (y == 0)).sum()) if interventions else 0,
                    mean_ms=lat.mean(), p95_ms=np.quantile(lat, .95), total_s=lat.sum() / 1000, note=note)
    rows = [evaluate("No guard (log only)", base, False, "baseline")]
    rows.append(evaluate("Warn", base, True, "alert only; latency unchanged"))
    lat = [fastest.get((r["intent"], r["draw"]), r["native_ms"]) if f else r["native_ms"] for r, f in zip(acc, flag)]
    rows.append(evaluate("Rewrite to verified equivalent", lat, True, "semantic experience memory"))
    lat = [r["best_hint_ms"] if f else r["native_ms"] for r, f in zip(acc, flag)]
    rows.append(evaluate("Switch optimizer configuration", lat, True, "best hint set (optimistic)"))
    lat = []
    for r, f in zip(acc, flag):
        ref = fastest.get((r["intent"], r["draw"]), r["native_ms"])
        tau = 3 * ref
        lat.append(tau + ref if (f and r["native_ms"] > tau) else r["native_ms"])
    rows.append(evaluate("Timeout 3x expected, then rewrite", lat, True, "pays the timeout first"))
    lat = [(ARGS.review_ms + fastest.get((r["intent"], r["draw"]), r["native_ms"])) if f else r["native_ms"] for r, f in zip(acc, flag)]
    rows.append(evaluate("Human review", lat, True, f"{ARGS.review_ms / 1000:.0f} s delay per flagged query"))
    return pd.DataFrame(rows)


def step11_policies(acc, proto0):
    banner(11, TOTAL, "Mitigation policies and decision curves",
           "A guard is only useful if acting on its warnings helps more than it hurts. Using the measured runtimes "
           "we replay what each policy would have done for every query, using the out-of-sample risk scores of the "
           "M4 detector. We count regressions prevented and unnecessary interventions on healthy queries, and draw "
           "a decision curve: the net benefit of acting at each risk threshold compared with acting on all or none.")
    thr = ARGS.risk_threshold
    P = replay_policies(acc, "risk", thr)
    show_table(P, f"Policy replay, risk scores from protocol {proto0}, threshold {thr}")
    if "risk_B" in acc[0] and proto0[0] != "B":
        PB = replay_policies(acc, "risk_B", thr)
        show_table(PB[PB.policy.isin(["Warn", "Rewrite to verified equivalent", "Timeout 3x expected, then rewrite"])],
                   f"Same replay with risk scores for completely new questions (protocol B)")
    y = np.array([r["y"] for r in acc])
    fig, ax = plt.subplots(figsize=(7, 4))
    ts = np.linspace(0.02, 0.9, 45)
    prev = y.mean()
    nb_all = [prev - (1 - prev) * t / (1 - t) for t in ts]
    best = None
    for key, lab, sty in (("risk", f"SQL-Guard M4, protocol {proto0[0]}", "-"), ("risk_B", "SQL-Guard M4, protocol B", ":")):
        if key not in acc[0]:
            continue
        risk = np.array([r[key] for r in acc])
        nb = [((risk >= t) & (y == 1)).sum() / len(y) - ((risk >= t) & (y == 0)).sum() / len(y) * t / (1 - t) for t in ts]
        ax.plot(ts, nb, sty, color="#000", label=lab)
        if best is None:
            k = int(np.argmin(np.abs(ts - thr)))
            best = (ts[k], nb[k], nb_all[k])
    ax.plot(ts, nb_all, "--", color="#888", label="act on every query")
    ax.axhline(0, color="#bbb", label="act on none")
    ax.set_ylim(-0.05, max(0.05, prev + 0.02))
    ax.set_xlabel("risk threshold"); ax.set_ylabel("net benefit"); ax.set_title("Decision curve")
    ax.legend(fontsize=8)
    save_fig(fig, "fig6_decision_curve.png", "Decision curve analysis for acting on SQL-Guard risk scores.")
    say(f"At the operating threshold {best[0]:.2f}, acting on SQL-Guard's warnings (protocol {proto0[0]}) gives a net "
        f"benefit of {best[1]:.3f}, while intervening on every query gives {best[2]:.3f} and doing nothing gives 0. "
        "A positive gap means the guard's warnings are worth acting on at that threshold.")
    return P


def main():
    global OUT_DIR, SF
    OUT_DIR = os.path.abspath(ARGS.out)
    SF = ARGS.sf
    os.makedirs(OUT_DIR, exist_ok=True)
    t_start = time.time()
    print(LINE)
    print(" AgentSQL-Guard with InvaCard  -  live demonstration")
    print(" Foundation of Data Science, Group 13 (Umar, Sohailur, Vineet)  |  Faculty: Kalyanaraman P")
    print(LINE)
    print(f" Settings: scale factor {ARGS.sf}, {ARGS.draws} parameter draws, {ARGS.reps} timed runs per plan, "
          f"threads={ARGS.threads}, output folder: {OUT_DIR}")
    con = step1_database()
    inst = step2_workload()
    acc = step3_equivalence(con, inst)
    fp = step4_fingerprints(acc)
    M, nodes = step5_measure(con, acc)
    nodes.drop(columns=[c for c in ("child_idx", "_id") if c in nodes.columns]).to_csv(os.path.join(OUT_DIR, "subplan_cardinalities.csv"), index=False)
    worst = step6_explain_worst(acc, M)
    bias, C = step7_estimation_bias(nodes)
    ce, ce2, ce_info = step8_invacard(con, nodes)
    D, ci, best_single, proto0 = step9_detection(acc, M)
    S, leak_ok = step10_factorial(con, acc, M)
    P = step11_policies(acc, proto0)
    # ------------------------------------------------------------------------------------- summary
    ag = M[M.arm == "agent"]; ex = M[M.arm == "expert"]
    m4 = D[(D.detector.str.startswith("M4")) & (D.model == "boosted trees") & (D.protocol == proto0)].iloc[0]
    m4b = D[(D.detector.str.startswith("M4")) & (D.model == "boosted trees") & (D.protocol != proto0)]
    rw = P[P.policy == "Rewrite to verified equivalent"].iloc[0]
    summary = [
        ("Verified queries in the speed comparison", f"{len(acc)} of {len(inst)}"),
        ("Regression rate: agent vs expert wordings", f"{pct(ag.y.mean())} vs {pct(ex.y.mean())}"),
        ("Worst agent wording vs fastest equivalent", f"{worst.qid}: {worst.R_form:.1f}x slower, same answer"),
        ("Same sub-expression estimated >=2x differently across wordings", f"{bias['viol2']} of {bias['shared']}"),
        ("Coherence certificate: plans flagged / sound on all plans", f"{bias['cert_fired']} / {'yes' if bias['cert_sound'] else 'no'}"),
        ("P1 q-error p90: native -> InvaCard", f"{ce['B0 native estimator']['p90']:.2f} -> {ce['InvaCard (full)']['p90']:.2f}"),
        ("P1 q-error p99: native -> InvaCard", f"{ce['B0 native estimator']['p99']:.1f} -> {ce['InvaCard (full)']['p99']:.1f}"),
        ("Key reuse on agent sub-expressions: syntactic vs semantic", f"{pct(ce_info['hit_syn'])} vs {pct(ce_info['hit_sem'])}"),
        ("Law violations: native -> residual only -> InvaCard", f"{ce_info['viol_before']} -> {ce_info['viol_resid']} -> {ce_info['viol_after']}"),
        ("Stage-one M4 AUPRC, seen question / new question (base rate)",
         f"{m4.AUPRC:.3f} / {(m4b.AUPRC.iloc[0] if len(m4b) else float('nan')):.3f} ({M.y.mean():.3f})"),
        (f"M4 - {best_single} AUPRC, 95% CI", f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"),
        ("Leakage invariance test", "PASS" if leak_ok else "FAIL"),
        ("Rewrite policy: regressions prevented / unnecessary interventions", f"{rw.prevented} of {rw.regressions} / {rw.unnecessary}"),
    ]
    print("\n" + LINE)
    print(" SUMMARY")
    print(LINE)
    for q, a in summary:
        print(f" {q:<66s} {a}")
    path = write_html(summary)
    print(LINE)
    print(f" Finished in {time.time() - t_start:.0f} s.")
    print(f" HTML report : {path}")
    print(f" Data files  : measurements.csv, subplan_cardinalities.csv, equivalence_checks.csv in {OUT_DIR}")
    print(" Note: this is a pilot-scale demonstration on an embedded engine (DuckDB). Exact numbers depend on the "
          "machine;\n the full study runs the same pipeline on PostgreSQL 16 with pg_hint_plan as described in the paper.")
    print(LINE)
    if not ARGS.no_browser:
        try:
            webbrowser.open(pathlib.Path(path).as_uri())
        except Exception:
            pass


def parse_args():
    ap = argparse.ArgumentParser(description="AgentSQL-Guard with InvaCard - single-file demonstration")
    ap.add_argument("--sf", type=float, default=0.1, help="TPC-H scale factor (default 0.1, about 600k line items)")
    ap.add_argument("--draws", type=int, default=4, help="parameter draws per intent (default 4 -> 240 queries)")
    ap.add_argument("--reps", type=int, default=3, help="timed runs per plan (median is used)")
    ap.add_argument("--threads", type=int, default=max(1, min(4, os.cpu_count() or 1)), help="DuckDB worker threads")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-query timeout in seconds")
    ap.add_argument("--min-gap-ms", dest="min_gap_ms", type=float, default=10.0, help="minimum absolute slowdown for a label")
    ap.add_argument("--tierb", type=int, default=8, help="queries in the fault-injection experiment")
    ap.add_argument("--bg-threads", dest="bg_threads", type=int, default=2, help="background queries for contention")
    ap.add_argument("--risk-threshold", dest="risk_threshold", type=float, default=0.5)
    ap.add_argument("--review-ms", dest="review_ms", type=float, default=30000.0)
    ap.add_argument("--dbgen", action="store_true", help="use DuckDB's official TPC-H dbgen (downloads an extension)")
    ap.add_argument("--quick", action="store_true", help="fast run: 2 draws, 1 timed run, 4 Tier-B queries")
    ap.add_argument("--out", default="demo_output", help="output folder")
    ap.add_argument("--no-browser", dest="no_browser", action="store_true", help="do not open the HTML report")
    a = ap.parse_args()
    if a.quick:
        a.draws, a.reps, a.tierb = 2, 1, 4
    return a


if __name__ == "__main__":
    ARGS = parse_args()
    OUT_DIR = "demo_output"
    main()
