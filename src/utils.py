from datetime import datetime
import os
import seaborn as sns
import matplotlib.pyplot as plt


def create_results_folder(data_type, approach):
    """
    Creates a folder for the results of the experiment. The folder is named after the current date and time.
    :param data_type: The type of data used for the experiment (e.g. "experiment" or "synthetic").
    :param approach: The approach used for the experiment (e.g. "loss_function" or "data_baseline").
    :return: The path to the folder.
    """
    time = datetime.now()
    folder_name = f"type-{data_type}_date-" + time.strftime("%Y-%m-%d_%H-%M-%S/")
    results_folder = f"../Results/{approach}/" + folder_name
    os.makedirs(results_folder)
    results_folder_images = results_folder + "Images/"
    os.makedirs(results_folder_images)
    results_folder_model = results_folder + "Model/"
    os.makedirs(results_folder_model)
    results_folder_history = results_folder + "History/"
    os.makedirs(results_folder_history)
    return results_folder


def create_prediction_plot(
    times, true_values, preds_time, predictions, image_folder, show_plot, sample_name, dataset_type, title
):
    """
    Creates a plot of the predictions of the model. The plot is saved in the specified result folder. The plot can be shown.
    :param times: The timestamps of the true values.
    :param true_values: The true values.
    :param preds_time: The timestamps of the predictions.
    :param predictions: The predictions.
    :param image_folder: The folder where the plot is saved.
    :param show_plot: If True, the plot is shown.
    :param sample_name: The name of the predictions sample.
    :param dataset_type: The type of the dataset (e.g. "test" or "train").
    :param title: The title of the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    sns.lineplot(ax=ax, x=times, y=true_values, color="#666666", label="True", linewidth=2)
    sns.lineplot(ax=ax, x=preds_time, y=predictions, color="#009682", label="Predictions", linewidth=3)
    ax.set_ylabel("Electric power [W]")
    ax.set_xlabel("Time [sec]")
    ax.set_ylim([0, 4050])
    ax.set_title(title)
    fig.tight_layout()
    if show_plot:
        plt.show()
    fig.savefig(f"{image_folder}{dataset_type}_{sample_name}.png", dpi=fig.dpi)
    fig.savefig(f"{image_folder}{dataset_type}_{sample_name}.svg", dpi=fig.dpi)
    plt.close("all")
    plt.clf()


def create_sample_plot(exp, ylim, folder, title, show_plot=False):
    """
    Creates a plot of the original sample. The plot is saved in the specified data folder. The plot can be shown.
    :param exp: The sample dataframe.
    :param ylim: The y-axis limits.
    :param folder: The data folder where the plot is saved.
    :param title: The title of the plot.
    :param show_plot: If True, the plot is shown.
    """
    plt.rcParams["figure.max_open_warning"] = 0
    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    (line1,) = ax.plot(exp["time"], exp["el_power"], color="#666666", label="el_power")
    ax.set_ylabel("Electric power [W]", color="#666666")
    ax.tick_params(axis="y", labelcolor="#666666")
    ax.set_ylim([-10, ylim])
    ax.set_xlabel("Time [sec]")
    ax2 = ax.twinx()
    (line2,) = ax2.plot(exp["time"], exp["input_voltage"], color="#A22223", label="input_voltage")
    ax2.set_ylabel("Voltage [V]", color="#A22223")
    ax2.tick_params(axis="y", labelcolor="#A22223")
    ax2.set_ylim([-0.5, 10.5])
    ax2.legend([line1, line2], ["el. power", "voltage"], loc="upper right")
    fig.tight_layout()
    fig.savefig(folder + f"{title}.png", dpi=fig.dpi)
    fig.savefig(folder + f"{title}.svg", dpi=fig.dpi)
    if not show_plot:
        plt.close("all")
        plt.clf()
