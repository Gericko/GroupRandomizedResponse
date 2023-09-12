import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DIR_LOGS = BASE_DIR / "logs"
COMPILED_FILE = "compiled_experiment_results.csv"
SUMMARY_FILE = "summarized_experiment_results.csv"


def plot_size(df):
    sns.set_theme()
    figure = sns.relplot(
        data=df,
        kind="line",
        x="graph_size",
        y="l2_error",
        col="graph",
        hue="algorithm",
        style="algorithm",
    )
    ax = figure.ax
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.show()


def plot_sample(df):
    sns.set_theme()
    figure = sns.relplot(
        data=df,
        kind="line",
        x="sample",
        y="l2_error",
        col="graph",
        hue="algorithm",
        style="algorithm",
    )
    ax = figure.ax
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.show()


def plot_download(df):
    sns.set_theme()
    figure = sns.relplot(
        data=df,
        kind="line",
        x="download_cost",
        y="l2_error",
        col="graph",
        hue="algorithm",
        style="algorithm",
    )
    ax = figure.ax
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.show()


def plot_bias(df):
    sns.set_theme()
    figure = sns.relplot(
        data=df,
        kind="line",
        x="download_cost",
        y="error",
        col="graph",
        hue="algorithm",
        style="algorithm",
    )
    ax = figure.ax
    ax.set_xscale("log")
    # ax.set_yscale("log")
    plt.show()


def plot_exp1(df):
    sns.set_theme()
    figure = sns.relplot(
        data=df,
        kind="line",
        x="download_cost",
        y="l2_error",
        col="graph",
        hue="algorithm",
        style="algorithm",
    )
    ax = figure.ax
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(DIR_LOGS / COMPILED_FILE)
    df_sum = pd.read_csv(DIR_LOGS / SUMMARY_FILE)
    plot_exp1(df[(df["exp_name"] == "exp1") & (df["graph"] == "wiki")])
    plot_sample(df[df["exp_name"] == "exp2"])
    plot_sample(df[df["exp_name"] == "exp0"])
