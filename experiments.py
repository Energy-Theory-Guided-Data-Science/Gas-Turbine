import os

os.chdir("src/")
from src.dataset import Dataset
from src.model import Model

NUMBER_OF_EXPERIMENTS = 5
LOSS_FUNCTIONS = ["mean_squared_error", "custom"]
SEED = 42


def run_ex(data_type, n_train_samples, seed, loss_function, ex_name, steepness=None, steepness_loss=None, theta=100):
    """
    Run an experiment with the given parameters.
    :param data_type: The type of data to use. Can be "synthetic" or "experiment".
    :param n_train_samples: The number of samples to use for training.
    :param seed: The seed to use for the random number generator.
    :param loss_function: The loss function used for the model (e.g. "mean_squared_error" or "custom").
    The custom loss function consists of the mean squared error and the weighted physical part.
    :param steepness: The steepness of the transition part of the synthetic data. Only used if data_type is "synthetic".
    :param steepness_loss: The steepness domain knowledge of the loss function. Only used if loss_function is "custom".
    :param theta: The weight of the physical part of the loss function. Only used if loss_function is "custom".
    """
    if steepness is not None and steepness_loss is None:
        steepness_loss = steepness
    dataset = Dataset(data_type=data_type, steepness=steepness, n_train_samples=n_train_samples, seed=seed)
    if data_type == "experiment" and steepness_loss is None:
        steepness_loss = dataset.steepness
    dataset.create_train_data()
    model = Model(
        data_type=dataset.data_type,
        loss_function=loss_function,
        scaler=dataset.scaler,
        steepness=steepness_loss,
        input_shape=(dataset.X_train.shape[1], dataset.X_train.shape[2]),
        theta=theta,
        seed=seed,
        ex_name=ex_name,
    )
    model.train(dataset.X_train, dataset.y_train, epochs=100, early_stopping=False)
    model.predict(dataset=dataset, show_plot=False)
    model.get_result()


def ex_training_size(seed):
    """
    Run experiments with different training sizes. The training size is the number of samples used for training.
    :param seed: The seed to use for the random number generator.
    """
    synthetic_size = [2, 20, 40, 60, 100]  # equals to percentage of training data
    for n in synthetic_size:
        for loss in LOSS_FUNCTIONS:
            run_ex(
                data_type="synthetic",
                steepness=10,
                n_train_samples=n,
                seed=seed,
                loss_function=loss,
                ex_name="training_size",
            )

    experiment_size = range(1, 7)  # number of training samples (equal to n/6 * 100 %)
    for n in experiment_size:
        for loss in LOSS_FUNCTIONS:
            run_ex(data_type="experiment", n_train_samples=n, seed=seed, loss_function=loss, ex_name="training_size")


def ex_lambda_value(seed):
    """
    Run experiments with different values for lambda. Lambda is the weight of the physical part of the loss function.
    :param seed: The seed to use for the random number generator.
    """
    lambdas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    for l in lambdas:
        for loss in LOSS_FUNCTIONS:
            run_ex(
                data_type="synthetic",
                steepness=10,
                n_train_samples=20,
                seed=seed,
                loss_function=loss,
                theta=l,
                ex_name="lambda_value",
            )
            run_ex(
                data_type="experiment",
                n_train_samples=6,
                seed=seed,
                loss_function=loss,
                theta=l,
                ex_name="lambda_value",
            )


def ex_steepness_value(seed):
    """
    Run experiments with different steepness values.
    The steepness value is the steepness of the transition part of the synthetic data.
    :param seed: The seed to use for the random number generator.
    """
    steepness_values = [5, 10, 20, 40]
    for s in steepness_values:
        for loss in LOSS_FUNCTIONS:
            run_ex(
                data_type="synthetic",
                steepness=s,
                n_train_samples=20,
                seed=seed,
                loss_function=loss,
                ex_name="steepness_value",
            )


def ex_wrong_domain(seed):
    """
    Run experiments with different steepness values for the loss function to simulate a wrong domain knowledge.
    :param seed: The seed to use for the random number generator.
    """
    synthetic_steepness_values = [5, 10, 20]
    for s_domain in synthetic_steepness_values:
        for s_loss in synthetic_steepness_values:
            run_ex(
                data_type="synthetic",
                steepness=s_domain,
                steepness_loss=s_loss,
                n_train_samples=20,
                seed=seed,
                loss_function="custom",
                ex_name="wrong_domain",
            )

    experiment_steepness_values = [3.194, 6.388, 12.776]
    for s_loss in experiment_steepness_values:
        run_ex(
            data_type="experiment",
            steepness=6.388,
            steepness_loss=s_loss,
            n_train_samples=6,
            seed=seed,
            loss_function="custom",
            ex_name="wrong_domain",
        )


def main():
    for i in range(NUMBER_OF_EXPERIMENTS):
        ex_training_size(seed=SEED + i)
        ex_lambda_value(seed=SEED + i)
        ex_steepness_value(seed=SEED + i)
        ex_wrong_domain(seed=SEED + i)


if __name__ == "__main__":
    main()
