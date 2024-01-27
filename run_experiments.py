import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import concurrent.futures
from tqdm import tqdm
from src.dataset import Dataset
from src.model import Model


def worker(config, gpu_queue):
    gpu_id = gpu_queue.get()

    try:
        ds = Dataset(data_type="experiment", n_train_samples=config['train_size'], seed=config['run'])
        x_train, y_train = ds.prepare_data(ds.train_indices)
        x_val, y_val = ds.prepare_data(ds.test_indices)
        input_shape = (x_train.shape[1], x_train.shape[2])

        if 'theta' not in config:
            config['theta'] = 1000

        if 'tgds_ratio' not in config:
            config['tgds_ratio'] = 0.6

        if 'loss_tolerance' not in config:
            config['loss_tolerance'] = 0

        if 'steepness' not in config:
            config['steepness'] = ds.steepness

        model = Model(
            data_type=ds.data_type,
            n_train_samples=config['train_size'],
            loss_function=config['loss'],
            scaler=ds.scaler,
            steepness=config['steepness'],
            input_shape=input_shape,
            theta=config['theta'],
            tgds_ratio=config['tgds_ratio'],
            verbose=False,
            seed=config['run'],
            gpu=int(gpu_id),
            loss_tolerance=config['loss_tolerance'],
            ex_name=config['ex_name'],
        )
        model.train(x_train, y_train, x_val=x_val, y_val=y_val, epochs=300)
        model.predict(dataset=ds, show_plot=False)
    except Exception as e:
        print(str(e))
        traceback.print_exc()

    gpu_queue.put(gpu_id)


thetas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
tgds_ratios = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
train_sizes = [1, 2, 3, 4, 5, 6]
losses = ['mean_squared_error', 'soft_weighted_two_state']
loss_tolerances = [0, 0.01, 0.1]
incorrect_steepnesses = [6.388 / 100, 6.388 / 2, 6.388, 6.388 * 2, 6.388 * 100]

configs = []
for run in range(5):

    # Incorrect prior knowledge
    for st in incorrect_steepnesses:
        configs.append({
            'run': run,
            'train_size': 4,
            'loss': 'soft_weighted_two_state',
            'steepness': st,
            'ex_name': 'incorrect_prior_knowledge'
        })

    # Different train sizes with different loss tolerances
    for train_size in train_sizes:
        for tolerance in loss_tolerances:
            configs.append({
                'run': run,
                'train_size': train_size,
                'loss': 'soft_weighted_two_state',
                'loss_tolerance': tolerance,
                'ex_name': 'diff_sizes'
            })
        configs.append({
            'run': run,
            'train_size': train_size,
            'loss': 'mean_squared_error',
            'ex_name': 'diff_sizes'
        })

    # Sensitivity analysis for lambda
    for theta in thetas:
        configs.append({
            'run': run,
            'train_size': 4,
            'loss': 'soft_weighted_two_state',
            'theta': theta,
            'tgds_ratio': 1,
            'ex_name': 'sensitivity_analysis_lambda'
        })

    # Sensitivity analysis for tgds_ratio
    for tgds_ratio in tgds_ratios:
        configs.append({
            'run': run,
            'train_size': 4,
            'loss': 'soft_weighted_two_state',
            'theta': 1000,
            'tgds_ratio': tgds_ratio,
            'ex_name': 'sensitivity_analysis_tgds_ratio'
        })


def main():
    gpus = ["0", "1", "2", "3"]

    with Manager() as manager:
        gpu_queue = manager.Queue()
        for gpu_id in gpus:
            gpu_queue.put(gpu_id)

        with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [executor.submit(worker, config, gpu_queue) for config in configs]

            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(configs)):
                pass


if __name__ == "__main__":
    main()
