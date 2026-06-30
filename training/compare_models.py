"""
training/compare_models.py
Pulls the latest runs for Model V1 and Model V2 from MLflow and prints/saves
a side-by-side comparison report (used for the "Comparison Report" deliverable).

Run AFTER both train_v1.py and train_v2.py have been run at least once:
    python training/compare_models.py
"""

import mlflow
import pandas as pd

EXPERIMENT_NAME = "imdb-sentiment-classification"


def main():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"Experiment '{EXPERIMENT_NAME}' not found. Run train_v1.py and train_v2.py first.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=20,
    )

    rows = []
    for run in runs:
        name = run.data.tags.get("mlflow.runName", run.info.run_id)
        rows.append({
            "run_name": name,
            "run_id": run.info.run_id,
            "final_accuracy": run.data.metrics.get("final_accuracy"),
            "final_precision": run.data.metrics.get("final_precision"),
            "final_recall": run.data.metrics.get("final_recall"),
            "final_f1": run.data.metrics.get("final_f1"),
        })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["final_accuracy"])
    print("\n=== Model Comparison Report ===\n")
    print(df.to_string(index=False))

    df.to_csv("artifacts/model_comparison_report.csv", index=False)
    print("\nSaved report to artifacts/model_comparison_report.csv")


if __name__ == "__main__":
    main()
