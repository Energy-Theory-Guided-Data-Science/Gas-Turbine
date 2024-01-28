import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import concurrent.futures
from tqdm import tqdm
from src.dataset import Dataset
from src.model import Model


# Function to execute the model training and evaluation on a given GPU.
def worker(config, gpu_queue):
    # Retrieve a GPU ID from the queue.
    gpu_id = gpu_queue.get()

    try:
        # Initialize dataset with specified configuration.
        ds = Dataset(data_type="experiment", n_train_samples=config['train_size'], seed=config['run'])
        x_train, y_train = ds.prepare_data(ds.train_indices)
        x_val, y_val = ds.prepare_data(ds.test_indices)
        input_shape = (x_train.shape[1], x_train.shape[2])

        # Set default values for parameters if not specified in the config.
        config.setdefault('theta', 1000)
        config.setdefault('tgds_ratio', 0.6)
        config.setdefault('loss_tolerance', 0)
        config.setdefault('steepness', ds.steepness)

        # Initialize and train the model with the given configuration.
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

    # Return the GPU ID to the queue for reuse.
    gpu_queue.put(gpu_id)


# Define various configurations for the experiments.
thetas = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
tgds_ratios = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
train_sizes = [1, 2, 3, 4, 5, 6]
losses = ['mean_squared_error', 'soft_weighted_two_state']
loss_tolerances = [0, 0.01, 0.1]
incorrect_steepnesses = [6.388 / 100, 6.388 / 2, 6.388, 6.388 * 2, 6.388 * 100]

configs = []
for run in range(5):

    # Experiment configurations for incorrect prior knowledge.
    for st in incorrect_steepnesses:
        configs.append({
            'run': run,
            'train_size': 4,
            'loss': 'soft_weighted_two_state',
            'steepness': st,
            'ex_name': 'incorrect_prior_knowledge'
        })

    # Experiment configurations for different training sizes and loss tolerances.
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

    # Experiment configurations for sensitivity analysis of lambda parameter.
    for theta in thetas:
        configs.append({
            'run': run,
            'train_size': 4,
            'loss': 'soft_weighted_two_state',
            'theta': theta,
            'tgds_ratio': 1,
            'ex_name': 'sensitivity_analysis_lambda'
        })

    # Experiment configurations for sensitivity analysis of tgds_ratio parameter.
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
    # Define available GPUs.
    gpus = ["0", "1", "2", "3"]

    with Manager() as manager:
        # Create a queue for managing GPU allocation.
        gpu_queue = manager.Queue()
        for gpu_id in gpus:
            gpu_queue.put(gpu_id)

        # Use a process pool to run experiments in parallel.
        with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
            # Submit all configurations to the process pool.
            futures = [executor.submit(worker, config, gpu_queue) for config in configs]

            # Track the progress of experiments.
            for _ in tqdm(concurrent.futures.as_completed(futures), total=len(configs)):
                pass


if __name__ == "__main__":
    main()
