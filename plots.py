import os
from glob import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_results():
    """
    Collect the results of the experiments from the Results folder.
    :return: A pandas DataFrame with the results.
    """
    ex_results = []
    for ex in glob(f"Results/*/*/"):
        with open(ex + "config.json") as f:
            ex_config = json.load(f)
        if ex_config["data_type"] == "experiment":
            ex_config["training_size"] = round(ex_config["n_train_samples"] / 6 * 100, 1)
        if ex_config["data_type"] == "synthetic":
            ex_config["training_size"] = ex_config["n_train_samples"]
        with open(ex + "result.json") as f:
            ex_result = json.load(f)
        ex_config.update(ex_result)
        ex_results.append(ex_config)
    df_ex_results = pd.DataFrame.from_records(ex_results)
    return df_ex_results


def plot_training_size(results):
    """
    Plot and save the results of the training size experiment.
    :param results: A pandas DataFrame with the results of the training sizes experiment.
    """
    fig, axes = plt.subplots(nrows=2, figsize=(10, 10), dpi=140)
    fig.suptitle(f"Effect of Varying Training Size")
    for i, data_type in enumerate(set(results["data_type"])):
        df_plot = results[results["data_type"] == data_type].copy()
        axes[i].set_title(f"{data_type.capitalize()} Data")
        sns.pointplot(
            ax=axes[i],
            data=df_plot,
            x="training_size",
            y="test_rmse_mean",
            hue="approach",
            ci=68,
            marker="o",
            capsize=0.1,
            alpha=0.7,
        ).set(xlabel="Training Size [%]", ylabel="RMSE")
    fig.tight_layout()
    fig.savefig("Results/Plots/varying_training_size.png", dpi=fig.dpi)
    fig.savefig("Results/Plots/varying_training_size.svg", dpi=fig.dpi)
    plt.close("all")
    plt.clf()


def plot_lambda_value(results):
    """
    Plot and save the results of the lambda value experiment.
    :param results: A pandas DataFrame with the results of the lambda values experiment.
    """
    fig, axes = plt.subplots(nrows=2, figsize=(10, 10), dpi=140)
    fig.suptitle(f"Sensitivity to Hyperparameter lambda")
    for i, data_type in enumerate(set(results["data_type"])):
        df_plot = results[results["data_type"] == data_type].copy()
        axes[i].set_title(f"{data_type.capitalize()} Data")
        sns.pointplot(
            ax=axes[i],
            data=df_plot,
            x="lambda",
            y="test_rmse_mean",
            hue="approach",
            ci=68,
            marker="o",
            capsize=0.1,
            alpha=0.7,
        ).set(xlabel="Lambda", ylabel="RMSE")
    fig.tight_layout()
    fig.savefig("Results/Plots/varying_lambda_value.png", dpi=fig.dpi)
    fig.savefig("Results/Plots/varying_lambda_value.svg", dpi=fig.dpi)
    plt.close("all")
    plt.clf()


def plot_steepness_value(results):
    """
    Plot and save the results of the steepness values experiment.
    :param results: A pandas DataFrame with the results of the steepness values experiment.
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    df_plot = results.copy()
    fig.suptitle(f"Sensitivity to Degree of Steepness for Synthetic Data")
    sns.pointplot(
        ax=ax,
        data=df_plot,
        x="steepness",
        y="test_rmse_mean",
        hue="approach",
        ci=68,
        marker="o",
        capsize=0.1,
        alpha=0.7,
    ).set(xlabel="Steepness", ylabel="RMSE")
    fig.tight_layout()
    fig.savefig("Results/Plots/varying_steepness_value.png", dpi=fig.dpi)
    fig.savefig("Results/Plots/varying_steepness_value.svg", dpi=fig.dpi)
    plt.close("all")
    plt.clf()


def plot_wrong_domain(results):
    """
    Plot and save the results of the wrong domain experiment.
    :param results: A pandas DataFrame with the results of the wrong domain experiment.
    """
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(20, 10), dpi=140)
    fig.suptitle(f"Sensitivity to Provided Domain Knowledge")
    df_plot = results[results["data_type"] == "synthetic"].copy()
    for i, steepness in enumerate(set(df_plot["steepness"])):
        df_plot_steepness = df_plot[df_plot["steepness"] == steepness].copy()
        axes[0, i].set_title(f"Correct Steepness: {steepness}")
        sns.pointplot(
            ax=axes[0, i],
            data=df_plot_steepness,
            x="steepness",
            y="test_rmse_mean",
            hue="approach",
            ci=68,
            marker="o",
            capsize=0.1,
            alpha=0.7,
        ).set(xlabel="Steepness", ylabel="RMSE")
    df_plot = results[results["data_type"] == "experiment"].copy()
    axes[1, 1].set_title(f"Correct Steepness: {df_plot['steepness'].iloc[0]}")
    sns.pointplot(
        ax=axes[1, 1],
        data=df_plot,
        x="steepness",
        y="test_rmse_mean",
        hue="approach",
        ci=68,
        marker="o",
        capsize=0.1,
        alpha=0.7,
    ).set(xlabel="Steepness", ylabel="RMSE")
    axes[1, 0].axis("off")
    axes[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig("Results/Plots/varying_lambda_value.png", dpi=fig.dpi)
    fig.savefig("Results/Plots/varying_lambda_value.svg", dpi=fig.dpi)
    plt.close("all")
    plt.clf()


def main():
    os.makedirs("Results/Plots")
    results = get_results()
    plot_training_size(results[results["ex_name"] == "training_size"])
    plot_lambda_value(results[results["ex_name"] == "lambda_value"])
    plot_steepness_value(results[results["ex_name"] == "steepness_value"])
    plot_wrong_domain(results[results["ex_name"] == "wrong_domain"])


if __name__ == "__main__":
    main()
