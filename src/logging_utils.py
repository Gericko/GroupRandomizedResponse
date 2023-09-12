import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DIR_LOGS = BASE_DIR / "logs"
COMPILED_FILE = "compiled_experiment_results.csv"
SUMMARY_FILE = "summarized_experiment_results.csv"


if __name__ == "__main__":
    list_of_df = [
        pd.read_csv(file)
        for file in DIR_LOGS.glob("*.csv")
        if file != DIR_LOGS / COMPILED_FILE and file != DIR_LOGS / SUMMARY_FILE
    ]
    df = pd.concat(list_of_df, ignore_index=True)
    df.drop("nb_iter", axis=1, inplace=True)
    df["estimation"] = df["count"] + df["bias"] + df["noise"]
    df["error"] = df["estimation"] - df["true_count"]
    df["l2_error"] = (df["estimation"] - df["true_count"]) ** 2
    df["relative_error"] = abs(df["estimation"] - df["true_count"]) / df["true_count"]
    df.to_csv(DIR_LOGS / COMPILED_FILE)

    df_summary = df.groupby(
        ["exp_name", "graph", "graph_size", "algorithm", "privacy_budget", "sample"]
    ).agg("mean")
    df_summary.to_csv(DIR_LOGS / SUMMARY_FILE)
