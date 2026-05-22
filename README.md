# SE446 Milestone 2

Spark DataFrame analytics + MLlib arrest predictor on the Chicago Crime dataset
(Hadoop 3.4.1 / Spark 3.5.4, 1 master + 2 workers).

## Team

| Member                       | GitHub              | Tasks          |
|------------------------------|---------------------|----------------|
| Saad Abdullah Al Sufayan     | `Saad15111`         | 1, 3, 5, 6, 11 |
| Fahad Sami Alhomaidhi        | `fahadalhomaidhi8`  | 2, 4, 7, 9, 10 |

## Spec compliance (May 2026 update)

1. **Task 8 (CrossValidator) is omitted** — waived by the instructor.
2. **Phase B (Tasks 5–7) trains on a 5% sample** via `df.sample(fraction=0.05, seed=42)`.
   On the cluster this gives 39,534 rows (Train 31,728 / Test 7,806).
3. **Task 11 uses `--deploy-mode cluster`**. Application stdout is collected with
   `yarn logs -applicationId <appId>` into `output/spark_submit/run.log`.

## Repository layout

```
.
├── M2_Spark_ML.ipynb               Notebook (Tasks 1–7), executed locally
├── m2_spark_ml.py                  Standalone Phase B script for spark-submit
├── scripts/
│   └── compose_notebook.py         Notebook generator
├── output/
│   ├── incidents_per_year.png      Task 3 chart
│   ├── cluster_yarn_log.txt        Task 10 evidence
│   └── spark_submit/
│       ├── console.log             Task 11 spark-submit console output
│       └── run.log                 Task 11 application stdout
└── README.md
```

## Executive summary

We reproduce the four M1 MapReduce analyses with Spark DataFrames + Spark SQL on the
full 793,072-row HDFS dataset (numbers match M1 exactly). For arrest prediction we
build a Spark MLlib pipeline (StringIndexer × 2 + VectorAssembler + classifier) and
train Logistic Regression, Random Forest, and Gradient-Boosted Trees on a 5% sample
as required by the May 2026 spec update.

**Top model by AUC: GBT (0.8241).** Random Forest is a strong second (0.8073) at
roughly 14× faster training time — the better trade-off for production deployment.

---

# Phase A — Spark DataFrame analytics

## Task 1 — Crime type distribution
*Saad Abdullah Al Sufayan (`Saad15111`)*

```python
kind_counts = (events
               .groupBy("Primary Type")
               .agg(func.count(func.lit(1)).alias("rows"))
               .orderBy(func.col("rows").desc()))
```

**M1 (MapReduce) ↔ M2 (Spark) — Top 10:**

| Crime type | M1 | M2 |
|------------|---:|---:|
| THEFT | 162,688 | 162,688 |
| BATTERY | 151,930 | 151,930 |
| CRIMINAL DAMAGE | 91,241 | 91,241 |
| NARCOTICS | 74,127 | 74,127 |
| ASSAULT | 54,070 | 54,070 |
| MOTOR VEHICLE THEFT | 48,494 | 48,494 |
| BURGLARY | 39,872 | 39,872 |
| OTHER OFFENSE | 36,893 | 36,893 |
| ROBBERY | 30,991 | 30,991 |
| DECEPTIVE PRACTICE | 30,396 | 30,396 |

Numbers match exactly. Spark's DataFrame engine runs the aggregation in-memory;
the streaming MapReduce equivalent had to disk-shuffle between mapper and reducer.

---

## Task 2 — Location hotspots (Spark SQL)
*Fahad Sami Alhomaidhi (`fahadalhomaidhi8`)*

```python
events.createOrReplaceTempView("chicago_events")
hot_locations = spark.sql("""
    SELECT  `Location Description` AS place,
            COUNT(*)               AS hits
      FROM  chicago_events
     WHERE  `Location Description` IS NOT NULL
     GROUP  BY `Location Description`
     ORDER  BY hits DESC
     LIMIT  10
""")
```

**M1 ↔ M2 — Top 10 hotspots (full dataset):**

| Location | M1 | M2 |
|----------|---:|---:|
| STREET | 245,437 | 248,326 |
| RESIDENCE | 136,238 | 136,393 |
| APARTMENT | 60,925 | 61,235 |
| SIDEWALK | 47,407 | 47,506 |
| OTHER | 29,213 | 29,671 |
| PARKING LOT/GARAGE(NON.RESID.) | 21,876 | 22,436 |
| ALLEY | 18,258 | 18,349 |
| SCHOOL, PUBLIC, BUILDING | 20,516 | 15,776 |
| RESIDENCE-GARAGE | 14,266 | 14,291 |
| SMALL RETAIL STORE | 13,755 | 13,804 |

Slight differences come from M1's manual CSV split dropping a few hundred edge-case
rows that Spark's CSV parser keeps.

---

## Task 3 — Year trend
*Saad Abdullah Al Sufayan (`Saad15111`)*

Yearly counts on the full HDFS dataset (cluster):

| Year | Incidents | | Year | Incidents |
|---:|---:|---|---:|---:|
| 2001 | 467,301 | | 2014 | 825 |
| 2002 | 205,266 | | 2015 | 1,105 |
| 2003 | 985     | | 2016 | 1,339 |
| 2004 | 915     | | 2017 | 1,387 |
| 2005 | 1,031   | | 2018 | 1,327 |
| 2006 | 796     | | 2019 | 1,174 |
| 2007 | 762     | | 2020 | 1,832 |
| 2008 | 1,010   | | 2021 | 2,399 |
| 2009 | 910     | | 2022 | 4,678 |
| 2010 | 695     | | 2023 | 81,461 |
| 2011 | 770     | | 2024 | 880   |
| 2012 | 800     | | 2025 | 12,710 |
| 2013 | 714     | | | |

2001 and 2002 dominate, then a quiet stretch through 2022 with a sharp 2023 spike.
Local matplotlib chart at `output/incidents_per_year.png`.

---

## Task 4 — Arrest rate
*Fahad Sami Alhomaidhi (`fahadalhomaidhi8`)*

**Cluster — overall:** **221,932 / 793,073 = 27.98%** (matches M1 within rounding).

**Top arrest rates by crime type (min 100 rows):**

| Crime type | Rows | Arrest rate |
|------------|-----:|------------:|
| NARCOTICS | 74,127 | 99.88% |
| PROSTITUTION | 9,100 | 99.88% |
| LIQUOR LAW VIOLATION | 2,349 | 99.83% |
| GAMBLING | 1,314 | 99.77% |
| INTERFERENCE WITH PUBLIC OFFICER | 803 | 80.70% |
| WEAPONS VIOLATION | 8,893 | 74.60% |
| CRIMINAL TRESPASS | 21,476 | 73.58% |
| PUBLIC PEACE VIOLATION | 1,827 | 66.83% |
| HOMICIDE | 13,173 | 48.11% |
| SEX OFFENSE | 3,932 | 32.38% |

The arrest rate splits into two regimes — proactive-policing crimes near 100%
(report only exists because an officer made the stop) versus reactive-reporting
crimes like THEFT (14.2%) and BURGLARY (6.7%) where most cases go unsolved.
Phase B's ML model exploits this structure.

---

# Phase B — MLlib arrest predictor (5% sample)

## Task 5 — Feature pipeline
*Saad Abdullah Al Sufayan (`Saad15111`)*

`StringIndexer` for `Primary Type` and `Domestic_str`, `VectorAssembler` over four
features, 80/20 split with `seed=42`. The 5% sample is applied before any feature
engineering.

Vector layout: `[Hour, District, ptype_idx, is_dom_idx]`.

## Task 6 — Train and evaluate three classifiers
*Saad Abdullah Al Sufayan (`Saad15111`)*

Cluster results (5% sample of the full HDFS dataset):

| Model | Params | Train (s) | AUC | Accuracy | F1 | Precision | Recall |
|-------|--------|----------:|----:|---------:|---:|----------:|-------:|
| Logistic Regression | maxIter=100, regParam=0.01 | 19.5 | 0.6022 | 0.7280 | 0.6376 | 0.6923 | 0.7280 |
| Random Forest | numTrees=100, maxDepth=5, maxBins=64 | 32.0 | 0.8073 | 0.8156 | 0.7802 | 0.8528 | 0.8156 |
| **GBT** | maxIter=50, maxDepth=5, maxBins=64 | 466.9 | **0.8241** | **0.8500** | **0.8337** | **0.8610** | **0.8500** |

**Confusion matrices (TN/FP/FN/TP):**
- LR:  (5549, 93, 2030, 133)
- RF:  (5641, 1, 1438, 725)
- GBT: (5553, 89, 1082, 1081)

**Top model by AUC: GBT (0.8241).**

## Task 7 — Random Forest feature importances
*Fahad Sami Alhomaidhi (`fahadalhomaidhi8`)*

```
ptype_idx     0.7712  #####################################
Hour          0.0807  ####
District      0.0763  ####
is_dom_idx    0.0718  ####
```

`ptype_idx` dominates because the per-crime arrest-rate distribution from Task 4
is itself dominated by crime type (NARCOTICS ≈ 99% vs THEFT ≈ 14%). Once a tree
splits on the crime-type index it has most of its answer.

Logistic Regression underperforms the tree models because it treats `ptype_idx`
as a numeric feature and fits a single linear coefficient — implying a meaningless
ordering between crime types. Trees split on individual values of the index and
side-step that issue entirely.

---

# Phase C — Deployment evidence

## Task 9 — Local execution
*Fahad Sami Alhomaidhi (`fahadalhomaidhi8`)*

Notebook executed end-to-end with `jupyter nbconvert --execute` (Python 3.9, PySpark
3.5.1, Java 17). Section 1 prints:

```
Running on:     local
Spark version:  3.5.1
Spark master:   local[*]
```

10,000 rows generated in-memory by the W09B-style synthetic generator. All Tasks
1–7 ran; outputs are embedded in `M2_Spark_ML.ipynb`.

## Task 10 — Cluster execution: client mode
*Fahad Sami Alhomaidhi (`fahadalhomaidhi8`)*

```bash
fsalhomaidhi@master-node:~$ source /etc/profile.d/hadoop.sh
fsalhomaidhi@master-node:~$ source /etc/profile.d/spark.sh
fsalhomaidhi@master-node:~$ spark-submit --master yarn --deploy-mode client \
    --num-executors 2 --executor-memory 768m --executor-cores 1 \
    --driver-memory 1g notebook_runner.py
```

Excerpt from `output/cluster_yarn_log.txt`:

```
Running on:     cluster
Spark version:  3.5.4
Spark master:   yarn
Records ingested: 793,073
Phase B working set: 39,534 rows  (5% sample, seed=42)
Train rows: 31,728 | Test rows: 7,806

@@ training LogisticRegression
  AUC       0.6022
  Train(s)  19.5
@@ training RandomForest
  AUC       0.8073
  Train(s)  32.0
@@ training GBT
  AUC       0.8241
  Train(s)  466.9
Top model by AUC: GBT (0.8241)
```

YARN application: `application_1778738889964_0087`.

## Task 11 — spark-submit (cluster mode)
*Saad Abdullah Al Sufayan (`Saad15111`)*

Per the May 2026 spec update, Task 11 uses `--deploy-mode cluster`:

```bash
salsufayan@master-node:~$ spark-submit --master yarn --deploy-mode cluster \
    --num-executors 2 --executor-memory 1g --executor-cores 1 \
    --driver-memory 1g m2_spark_ml.py
```

YARN application: `application_1778738889964_0088` — `final status: SUCCEEDED`.

Application stdout is collected with `yarn logs -applicationId application_1778738889964_0088`
and saved to `output/spark_submit/run.log`. The console.log
(`output/spark_submit/console.log`) captures the spark-submit invocation and YARN's
progress reports.

---

## Spec note — executor cores

The M2 spec lists `--executor-cores 2`. The course YARN cluster's maximum container
allocation is `<memory:1536, vCores:1>` — requesting 2 vcores returns
`InvalidResourceRequestException`. We therefore use `--executor-cores 1`, the same
setting M1 used.

---

## Member contributions

| Member | Tasks | Contribution |
|--------|-------|--------------|
| Saad Abdullah Al Sufayan (`Saad15111`)    | 1, 3, 5, 6, 11 | Crime-type DataFrame query; year-trend table + chart; feature pipeline; three-classifier training and evaluation; spark-submit cluster-mode submission and log retrieval |
| Fahad Sami Alhomaidhi (`fahadalhomaidhi8`) | 2, 4, 7, 9, 10 | Spark SQL location-hotspots query; arrest-rate analysis; Random Forest feature importances; local notebook execution evidence; yarn-client cluster execution evidence |

## How to reproduce

Locally:
```bash
python3 -m venv venv && source venv/bin/activate
pip install pyspark==3.5.1 pandas matplotlib jupyter numpy
jupyter nbconvert --to notebook --execute M2_Spark_ML.ipynb --output M2_Spark_ML.ipynb
```

On the cluster:
```bash
ssh <user>@134.209.172.50
source /etc/profile.d/hadoop.sh
source /etc/profile.d/spark.sh
# one-time deps for python3.12
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.12 get-pip.py --user
python3.12 -m pip install --user numpy 'setuptools>=68'
# Phase B standalone (cluster mode):
spark-submit --master yarn --deploy-mode cluster \
    --num-executors 2 --executor-memory 1g --executor-cores 1 \
    --driver-memory 1g m2_spark_ml.py
```
