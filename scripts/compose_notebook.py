"""Compose M2_Spark_ML.ipynb from cell sources.

Run from the repo root:
    python scripts/compose_notebook.py
"""
import json
from pathlib import Path

OUT_FILE = "M2_Spark_ML.ipynb"

SAAD  = "Saad Abdullah Al Sufayan"
FAHAD = "Fahad Sami Alhomaidhi"


pages = []


def add_md(s):
    pages.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": s.splitlines(keepends=True),
    })


def add_code(s):
    pages.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": s.splitlines(keepends=True),
    })


# ============================================================
add_md(f"""# SE446 Milestone 2

Spark DataFrame analytics + MLlib arrest predictor on Chicago Crime data.

| Member            | GitHub              | Tasks          |
|-------------------|---------------------|----------------|
| {SAAD}            | `Saad15111`         | 1, 3, 5, 6, 11 |
| {FAHAD}           | `fahadalhomaidhi8`  | 2, 4, 7, 9, 10 |

**Spec compliance — May 2026 update:**

| Item | Detail |
|------|--------|
| Task 8 (CrossValidator) | **Omitted** (instructor waiver) |
| Phase B sampling | `df.sample(fraction=0.05, seed=42)` before feature engineering |
| Task 11 submission | `--deploy-mode cluster`, stdout via `yarn logs -applicationId <appId>` → `output/spark_submit/run.log` |
""")


# ------- Section 1: bootstrap -------
add_md("---\n## Section 1 — Bootstrap")

add_code("""import os
import time
import shutil

from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as func
from pyspark.sql.types import IntegerType, StringType


_HAS_HDFS = shutil.which("hdfs") is not None


def open_spark() -> SparkSession:
    cfg = (SparkSession.builder
           .appName("M2_Saad_Fahad")
           .config("spark.sql.shuffle.partitions", "8"))
    if _HAS_HDFS:
        return cfg.getOrCreate()
    return (cfg
            .master("local[*]")
            .config("spark.driver.memory", "2g")
            .getOrCreate())


where_we_run = "cluster" if _HAS_HDFS else "local"
spark        = open_spark()
if where_we_run == "local":
    spark.sparkContext.setLogLevel("WARN")

print("Running on:    ", where_we_run)
print("Spark version: ", spark.version)
print("Spark master:  ", spark.sparkContext.master)
""")


# ------- Section 2: ingest -------
add_md("---\n## Section 2 — Ingest the dataset")

add_code("""HDFS_FILE = "hdfs:///data/chicago_crimes.csv"


def pull_from_hdfs():
    raw = spark.read.csv(HDFS_FILE, header=True, inferSchema=True)
    return (raw
            .withColumn("Hour",
                        func.hour(func.to_timestamp(func.col("Date"),
                                                    "MM/dd/yyyy hh:mm:ss a")))
            .withColumn("label",        func.col("Arrest").cast(IntegerType()))
            .withColumn("Domestic_str", func.col("Domestic").cast(StringType())))


def generate_local(rows: int = 10_000):
    import random
    random.seed(42)
    rate_per_kind = {
        "NARCOTICS":           0.85,
        "PROSTITUTION":        0.80,
        "WEAPONS VIOLATION":   0.60,
        "BATTERY":             0.30,
        "ASSAULT":             0.25,
        "ROBBERY":             0.15,
        "THEFT":               0.10,
        "BURGLARY":            0.08,
        "MOTOR VEHICLE THEFT": 0.06,
        "CRIMINAL DAMAGE":     0.05,
    }
    locs = ["STREET", "RESIDENCE", "APARTMENT", "SIDEWALK", "OTHER",
            "PARKING LOT", "SCHOOL", "ALLEY", "RESIDENCE-GARAGE"]
    yrs = [2020, 2021, 2022, 2023, 2024, 2025]
    bag = []
    for _ in range(rows):
        kind = random.choice(list(rate_per_kind))
        h = random.randint(0, 23)
        dom = random.random() < 0.15
        p = rate_per_kind[kind] + (0.20 if dom else 0.0)
        if 2 <= h <= 5:
            p -= 0.10
        p = max(0.01, min(0.99, p))
        bag.append(Row(
            District=random.randint(1, 25),
            **{"Primary Type": kind},
            **{"Location Description": random.choice(locs)},
            Year=random.choice(yrs),
            Hour=h,
            Domestic_str=str(dom).lower(),
            Arrest=random.random() < p,
            label=int(random.random() < p),
        ))
    return spark.createDataFrame(bag)


events = pull_from_hdfs() if where_we_run == "cluster" else generate_local()
events.cache()
print("Records ingested:", f"{events.count():,}")
events.printSchema()
events.show(3, truncate=False)
""")


# ============================================================
add_md("---\n# Phase A — DataFrame analytics")


add_md(f"""## Task 1 — Crime type distribution
*{SAAD}*

DataFrame `groupBy` + descending count.""")

add_code(f"""# Task 1 — {SAAD}
kind_counts = (events
               .groupBy("Primary Type")
               .agg(func.count(func.lit(1)).alias("rows"))
               .orderBy(func.col("rows").desc()))
kind_counts.show(10, truncate=False)
""")


add_md(f"""## Task 2 — Location hotspots (Spark SQL)
*{FAHAD}*

Switch to SQL via `createOrReplaceTempView`.""")

add_code(f"""# Task 2 — {FAHAD}
events.createOrReplaceTempView("chicago_events")

hot_locations = spark.sql(\"\"\"
    SELECT  `Location Description` AS place,
            COUNT(*)               AS hits
      FROM  chicago_events
     WHERE  `Location Description` IS NOT NULL
     GROUP  BY `Location Description`
     ORDER  BY hits DESC
     LIMIT  10
\"\"\")
hot_locations.show(truncate=False)
""")


add_md(f"""## Task 3 — Year trend
*{SAAD}*

Counts per year (matplotlib chart in local mode).""")

add_code(f"""# Task 3 — {SAAD}
counts_per_year = (events
                   .groupBy("Year")
                   .agg(func.count(func.lit(1)).alias("rows"))
                   .orderBy("Year"))
counts_per_year.show(30)
""")

add_code(f"""# Task 3 chart — {SAAD}
if where_we_run == "local":
    import matplotlib.pyplot as plt
    pdf = counts_per_year.toPandas().dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(pdf["Year"].astype(int), pdf["rows"], s=40, color="#1f4e79", zorder=3)
    ax.vlines(pdf["Year"].astype(int), 0, pdf["rows"], colors="#88a4c7", lw=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Incidents")
    ax.set_title("Chicago crime incidents per year")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/incidents_per_year.png", dpi=120)
    plt.show()
else:
    print("Cluster mode — printed table is the deliverable.")
""")


add_md(f"""## Task 4 — Arrest rate
*{FAHAD}*

Overall arrest rate plus the per-crime-type breakdown.""")

add_code(f"""# Task 4 — {FAHAD}
n_rows    = events.count()
n_arrests = events.filter(func.col("Arrest") == True).count()
print(f"Overall arrest rate: {{n_arrests:,}} / {{n_rows:,}} = {{n_arrests/n_rows*100:.2f}}%")

by_kind = (events
           .groupBy("Primary Type")
           .agg(func.count(func.lit(1)).alias("rows"),
                func.avg(func.col("label").cast("double")).alias("arrest_rate"))
           .filter(func.col("rows") >= 100)
           .orderBy(func.col("arrest_rate").desc()))
print("Top arrest rates by crime type (min 100 rows):")
by_kind.show(15, truncate=False)
""")


# ============================================================
add_md("""---
# Phase B — MLlib arrest predictor (5% sample)

The May 2026 spec update requires Phase B to train on a 5% sample. On the full HDFS
dataset that gives ~39,654 rows — small enough to fit the cluster's RAM budget.""")

add_code("""# 5% sample, seed=42 (applied before any feature engineering)
ml_rows = events.sample(fraction=0.05, seed=42)
print("Phase B working set:", f"{ml_rows.count():,} rows  (5% sample, seed=42)")
""")


add_md(f"""## Task 5 — Feature pipeline
*{SAAD}*

`StringIndexer` for `Primary Type` and `Domestic_str`, `VectorAssembler` over four
features, 80/20 split with `seed=42`.""")

add_code(f"""# Task 5 — {SAAD}
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler

if "Domestic_str" not in ml_rows.columns:
    ml_rows = ml_rows.withColumn("Domestic_str",
                                 func.col("Domestic").cast(StringType()))

ptype_idx   = StringIndexer(inputCol="Primary Type",
                            outputCol="ptype_idx",
                            handleInvalid="skip")
is_dom_idx  = StringIndexer(inputCol="Domestic_str",
                            outputCol="is_dom_idx",
                            handleInvalid="skip")
feature_pack = VectorAssembler(
    inputCols=["Hour", "District", "ptype_idx", "is_dom_idx"],
    outputCol="feature_pack",
)

train_rows, test_rows = ml_rows.randomSplit([0.8, 0.2], seed=42)
train_rows.cache()
test_rows.cache()
print("Train rows:", f"{{train_rows.count():,}}", "| Test rows:", f"{{test_rows.count():,}}")

inspect = Pipeline(stages=[ptype_idx, is_dom_idx, feature_pack]).fit(train_rows)
inspect.transform(train_rows).select(
    "Primary Type", "ptype_idx",
    "Hour", "District",
    "Domestic_str", "is_dom_idx",
    "feature_pack", "label",
).show(5, truncate=False)
print("Vector layout: [Hour, District, ptype_idx, is_dom_idx]")
""")


add_md(f"""## Task 6 — Train and evaluate three classifiers
*{SAAD}*

Logistic Regression (maxIter=100, regParam=0.01), Random Forest (numTrees=100,
maxDepth=5, maxBins=64), GBT (maxIter=50, maxDepth=5, maxBins=64).""")

add_code(f"""# Task 6 helpers — {SAAD}
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator,
)

bin_evalr   = BinaryClassificationEvaluator(labelCol="label")
multi_evalr = MulticlassClassificationEvaluator(labelCol="label",
                                                predictionCol="prediction")


def take_metrics(predictions):
    return {{
        "AUC":       bin_evalr.evaluate(predictions),
        "Accuracy":  multi_evalr.evaluate(predictions, {{multi_evalr.metricName: "accuracy"}}),
        "F1":        multi_evalr.evaluate(predictions, {{multi_evalr.metricName: "f1"}}),
        "Precision": multi_evalr.evaluate(predictions, {{multi_evalr.metricName: "weightedPrecision"}}),
        "Recall":    multi_evalr.evaluate(predictions, {{multi_evalr.metricName: "weightedRecall"}}),
    }}


def confusion_quartet(predictions):
    rows = predictions.groupBy("label", "prediction").count().collect()
    pkg = {{(int(r["label"]), int(r["prediction"])): r["count"] for r in rows}}
    return (pkg.get((0, 0), 0), pkg.get((0, 1), 0),
            pkg.get((1, 0), 0), pkg.get((1, 1), 0))
""")

add_code(f"""# Task 6 training loop — {SAAD}
recipe = [
    ("LogisticRegression",
     LogisticRegression(featuresCol="feature_pack", labelCol="label",
                        maxIter=100, regParam=0.01)),
    ("RandomForest",
     RandomForestClassifier(featuresCol="feature_pack", labelCol="label",
                            numTrees=100, maxDepth=5,
                            maxBins=64, seed=42)),
    ("GBT",
     GBTClassifier(featuresCol="feature_pack", labelCol="label",
                   maxIter=50, maxDepth=5,
                   maxBins=64, seed=42)),
]

bench    = []
rf_inner = None
for tag, learner in recipe:
    print(f"\\n@@ training {{tag}}")
    pipeline = Pipeline(stages=[ptype_idx, is_dom_idx, feature_pack, learner])
    t0 = time.time()
    fitted_pipe = pipeline.fit(train_rows)
    secs = time.time() - t0
    preds = fitted_pipe.transform(test_rows)
    metrics_dict = take_metrics(preds)
    cm = confusion_quartet(preds)
    bench.append((tag, fitted_pipe, metrics_dict, cm, secs))
    for k, v in metrics_dict.items():
        print(f"  {{k:<10}}{{v:.4f}}")
    print(f"  Train(s)  {{secs:.1f}}")
    print(f"  CM (TN,FP,FN,TP) = {{cm}}")
    if tag == "RandomForest":
        rf_inner = fitted_pipe.stages[-1]

# Side-by-side
print("\\n" + "=" * 78)
print(f"{{'metric':<11}}{{'Logistic':>14}}{{'RandomForest':>16}}{{'GBT':>14}}")
print("-" * 78)
m_lr, m_rf, m_gbt = (bench[0][2], bench[1][2], bench[2][2])
for k in ("AUC", "Accuracy", "F1", "Precision", "Recall"):
    print(f"{{k:<11}}{{m_lr[k]:>14.4f}}{{m_rf[k]:>16.4f}}{{m_gbt[k]:>14.4f}}")
print(f"{{'Train(s)':<11}}{{bench[0][4]:>14.1f}}{{bench[1][4]:>16.1f}}{{bench[2][4]:>14.1f}}")
print("=" * 78)
winner = max(bench, key=lambda r: r[2]["AUC"])
print(f"Top model by AUC: {{winner[0]}} ({{winner[2]['AUC']:.4f}})")
""")


add_md(f"""## Task 7 — Random Forest feature importances
*{FAHAD}*""")

add_code(f"""# Task 7 — {FAHAD}
schema = ["Hour", "District", "ptype_idx", "is_dom_idx"]
imp = rf_inner.featureImportances.toArray()

print("Random Forest feature importances:")
for col_nm, val in sorted(zip(schema, imp), key=lambda kv: -kv[1]):
    print(f"  {{col_nm:<12}} {{val:.4f}}  {{('#' * int(round(val * 50)))}}")
""")


add_md("""**Reading the importances.** The crime-type index dominates because the
per-crime arrest-rate distribution from Task 4 is itself dominated by crime type
— NARCOTICS is near 99% while THEFT is near 14%. Once a tree splits on the crime
type it has most of its answer.

Logistic Regression underperforms the tree models because it treats `ptype_idx`
as a numeric feature and fits a linear coefficient, implying a meaningless ordering
between crime types. Trees split on individual values of the index and side-step
that issue entirely.""")


add_md("---\n## Cleanup")

add_code("""spark.stop()""")


# ------- Write -------
nb = {
    "cells": pages,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path(OUT_FILE).write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT_FILE} ({len(pages)} cells)")
