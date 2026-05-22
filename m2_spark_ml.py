"""SE446 Milestone 2

Phase B (Tasks 5-7): standalone Spark MLlib pipeline for the spark-submit deliverable.

Authors:
    Task 5 — Saad Abdullah Al Sufayan
    Task 6 — Saad Abdullah Al Sufayan
    Task 7 — Fahad Sami Alhomaidhi

Per the May 2026 spec update:
    * Task 8 (CrossValidator) is waived.
    * Phase B trains on a 5% sample (df.sample(fraction=0.05, seed=42)).

Submit via:
    spark-submit \\
        --master yarn --deploy-mode cluster \\
        --num-executors 2 --executor-memory 1g --executor-cores 1 \\
        m2_spark_ml.py
"""
import time

from pyspark.sql import SparkSession
import pyspark.sql.functions as func
from pyspark.sql.types import IntegerType, StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator,
)


HDFS_FILE = "hdfs:///data/chicago_crimes.csv"


def open_spark() -> SparkSession:
    return (SparkSession.builder
            .appName("M2_Saad_Fahad_spark_submit")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())


def pull_events(session: SparkSession):
    raw = session.read.csv(HDFS_FILE, header=True, inferSchema=True)
    rows = (raw
            .withColumn("Hour",
                        func.hour(func.to_timestamp(func.col("Date"),
                                                    "MM/dd/yyyy hh:mm:ss a")))
            .withColumn("label",        func.col("Arrest").cast(IntegerType()))
            .withColumn("Domestic_str", func.col("Domestic").cast(StringType())))
    return rows.dropna(subset=["District", "Primary Type",
                               "Hour", "Domestic_str", "label"])


def take_metrics(predictions, bin_evalr, multi_evalr):
    return {
        "AUC":       bin_evalr.evaluate(predictions),
        "Accuracy":  multi_evalr.evaluate(predictions, {multi_evalr.metricName: "accuracy"}),
        "F1":        multi_evalr.evaluate(predictions, {multi_evalr.metricName: "f1"}),
        "Precision": multi_evalr.evaluate(predictions, {multi_evalr.metricName: "weightedPrecision"}),
        "Recall":    multi_evalr.evaluate(predictions, {multi_evalr.metricName: "weightedRecall"}),
    }


def confusion_quartet(predictions):
    rows = predictions.groupBy("label", "prediction").count().collect()
    pkg = {(int(r["label"]), int(r["prediction"])): r["count"] for r in rows}
    return (pkg.get((0, 0), 0), pkg.get((0, 1), 0),
            pkg.get((1, 0), 0), pkg.get((1, 1), 0))


def main():
    spark = open_spark()
    print("Spark version:", spark.version)
    print("Master:       ", spark.sparkContext.master)

    events = pull_events(spark)
    print("Records ingested:", f"{events.count():,}")

    # ----- Task 5 (Saad): pipeline + 5% sample -----
    sub = events.sample(fraction=0.05, seed=42)
    print("Phase B sample:", f"{sub.count():,} rows  (5%, seed=42)")

    ptype_idx    = StringIndexer(inputCol="Primary Type",
                                 outputCol="ptype_idx",
                                 handleInvalid="skip")
    is_dom_idx   = StringIndexer(inputCol="Domestic_str",
                                 outputCol="is_dom_idx",
                                 handleInvalid="skip")
    feature_pack = VectorAssembler(
        inputCols=["Hour", "District", "ptype_idx", "is_dom_idx"],
        outputCol="feature_pack",
    )

    train_rows, test_rows = sub.randomSplit([0.8, 0.2], seed=42)
    train_rows.cache()
    test_rows.cache()
    print("Train rows:", f"{train_rows.count():,}", "| Test rows:", f"{test_rows.count():,}")

    bin_evalr   = BinaryClassificationEvaluator(labelCol="label")
    multi_evalr = MulticlassClassificationEvaluator(labelCol="label",
                                                    predictionCol="prediction")

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

    rf_inner = None
    bench    = []
    for tag, learner in recipe:
        pipeline = Pipeline(stages=[ptype_idx, is_dom_idx, feature_pack, learner])
        t0 = time.time()
        fitted = pipeline.fit(train_rows)
        secs = time.time() - t0
        preds = fitted.transform(test_rows)
        metrics_dict = take_metrics(preds, bin_evalr, multi_evalr)
        cm = confusion_quartet(preds)
        bench.append((tag, secs, metrics_dict, cm))
        print(f"\n@@ {tag}")
        for k, v in metrics_dict.items():
            print(f"  {k:<10}{v:.4f}")
        print(f"  Train(s)  {secs:.1f}")
        print(f"  CM (TN,FP,FN,TP) = {cm}")
        if tag == "RandomForest":
            rf_inner = fitted.stages[-1]

    print("\n" + "=" * 78)
    print(f"{'metric':<11}{'Logistic':>14}{'RandomForest':>16}{'GBT':>14}")
    print("-" * 78)
    for k in ("AUC", "Accuracy", "F1", "Precision", "Recall"):
        print(f"{k:<11}{bench[0][2][k]:>14.4f}{bench[1][2][k]:>16.4f}{bench[2][2][k]:>14.4f}")
    print(f"{'Train(s)':<11}{bench[0][1]:>14.1f}{bench[1][1]:>16.1f}{bench[2][1]:>14.1f}")
    print("=" * 78)
    winner = max(bench, key=lambda r: r[2]["AUC"])
    print("Top model by AUC:", winner[0], f"({winner[2]['AUC']:.4f})")

    # ----- Task 7 (Fahad): RF feature importances -----
    print("\n--- Random Forest feature importances ---")
    schema = ["Hour", "District", "ptype_idx", "is_dom_idx"]
    for col_nm, val in sorted(zip(schema, rf_inner.featureImportances.toArray()),
                              key=lambda kv: -kv[1]):
        print(f"  {col_nm:<12} {val:.4f}  {'#' * int(round(val * 50))}")

    spark.stop()


if __name__ == "__main__":
    main()
