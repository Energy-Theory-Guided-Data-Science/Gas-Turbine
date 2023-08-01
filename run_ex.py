from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import concurrent.futures
from tqdm import tqdm
import gc
import random
from src.dataset import Dataset
from src.model import Model


def worker(config, gpu_queue):
    gc.collect()

    gpu_id = gpu_queue.get()

    ds = Dataset(data_type="experiment", n_train_samples=config['train_size'], test_samples=["4", "22"],
                 seed=config['run'])
    x_train, y_train = ds.prepare_data(ds.train_indices, lag=config['lag'])
    x_val, y_val = ds.prepare_data(ds.test_indices, lag=config['lag'])
    input_shape = (x_train.shape[1], x_train.shape[2])

    model = Model(
        data_type=ds.data_type,
        loss_function=config['loss'],
        scaler=ds.scaler,
        steepness=ds.steepness,
        input_shape=input_shape,
        neurons=32,
        learning_rate=0.001,
        theta=0.5,
        batch_size=128,
        verbose=False,
        dropout_rate=0.25,
        n_lstm_layers=3,
        seed=config['run'],
        gpu=gpu_id,
        loss_normalized=True
    )
    model.train(x_train, y_train, x_val=x_val, y_val=y_val, epochs=200)
    model.predict(dataset=ds, show_plot=False)
    model.get_result()

    gpu_queue.put(gpu_id)


configs = []
for run in range(3, 5):
    for loss in ['mean_squared_error']: #['two_state', 'mean_squared_error']:
        for train_size in [1, 2, 3, 4, 5, 6]:
            configs.append({'run': run, 'train_size': train_size, 'lag': 450, 'loss': loss})
for train_size in [3, 4, 5, 6]:
    configs.append({'run': 2, 'train_size': train_size, 'lag': 450, 'loss': 'mean_squared_error'})
random.shuffle(configs)
configs = sorted(configs, key=lambda x: x['run'])


def main():
    gpus = ["1", "1"]

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
