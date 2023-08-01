import optuna
import multiprocessing
from src.dataset import Dataset
from src.model import Model


def create_model(trial_params, ds, input_shape):
    # n_layers = trial_params['n_layers']
    # n_units = trial_params['n_units']
    # dropout_rate = trial_params['dropout_rate']
    # batch_size = trial_params['batch_size']

    theta = trial_params['theta']

    model = Model(
        data_type=ds.data_type,
        loss_function="two_state",  # "mean_squared_error",  "two_state"
        scaler=ds.scaler,
        steepness=ds.steepness,
        input_shape=input_shape,
        neurons=2 ** 5,
        learning_rate=0.001,
        theta=theta,
        batch_size=2 ** 7,
        verbose=False,
        dropout_rate=0.25,
        n_lstm_layers=3,
        gpu=0,
        loss_normalized=True,
    )

    return model


def train_model(trial_params, return_dict):
    # lag = trial_params['lag']
    lag = 450
    ds = Dataset(data_type="experiment", n_train_samples=3, test_samples=["4", "22"])
    x_train, y_train = ds.prepare_data(ds.train_indices, lag=lag)
    x_val, y_val = ds.prepare_data(ds.test_indices, lag=lag)

    try:
        model = create_model(trial_params, ds, (x_train.shape[1], x_train.shape[2]))
        history = model.train(x_train, y_train, x_val=x_val, y_val=y_val, epochs=100, get_history=True)
        val_loss = min(history.history['val_mean_squared_error'])
        return_dict['loss'] = val_loss
    except Exception as e:
        print(f"An exception occurred during training: {str(e)}")
        return_dict['loss'] = None


def objective(trial):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    # trial_params = {'n_layers': trial.suggest_int('n_layers', 1, 4),
    #                'n_units': trial.suggest_int('n_units', 4, 10),
    #                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.05),
    #                'batch_size': trial.suggest_int('batch_size', 4, 10)}
    # trial_params = {'lag': trial.suggest_int('lag', 50, 1000, step=50)}
    trial_params = {'theta': trial.suggest_float('theta', 0, 1)}
    p = multiprocessing.Process(target=train_model, args=(trial_params, return_dict))
    p.start()
    p.join()
    if return_dict['loss'] is None:
        raise optuna.TrialPruned()
    else:
        return return_dict['loss']


def main():
    study = optuna.create_study(direction='minimize')
    # more trials for real use case
    study.enqueue_trial({'theta': 1 / 10001})
    study.enqueue_trial({'theta': 1 / 1001})
    study.enqueue_trial({'theta': 1 / 101})
    study.enqueue_trial({'theta': 1 / 11})
    study.enqueue_trial({'theta': 1 / 2})
    study.enqueue_trial({'theta': 10 / 11})
    study.enqueue_trial({'theta': 100 / 101})
    study.enqueue_trial({'theta': 1000 / 1001})
    study.enqueue_trial({'theta': 10000 / 10001})

    study.optimize(objective, n_trials=1000, timeout=36 * 60 * 60)

    results = study.trials_dataframe()
    # results.to_csv("results_layers_units_dropout_batch.csv")
    # results.to_csv("results_lag.csv")
    results.to_csv("results_theta.csv")

    trial = study.best_trial
    print("Best trial: ", trial.number)
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
