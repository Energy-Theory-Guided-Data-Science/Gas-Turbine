import traceback
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import concurrent.futures
from tqdm import tqdm
import gc
import random
import time
from src.dataset import Dataset
from src.model import Model


def worker(config, gpu_queue):
    gc.collect()

    gpu_id = gpu_queue.get()

    try:
        while True:
            ds = Dataset(data_type="experiment", n_train_samples=config['train_size'], test_samples=["4", "22"],
                         seed=config['run'])
            x_train, y_train = ds.prepare_data(ds.train_indices)
            x_val, y_val = ds.prepare_data(ds.test_indices)
            input_shape = (x_train.shape[1], x_train.shape[2])

            if 'theta' not in config:
                config['theta'] = 0.04 # 0.03866414715431152

            if 'dropout_rate' not in config:
                config['dropout_rate'] = 0.25

            if 'tgds_ratio' not in config:
                config['tgds_ratio'] = 0.5

            model = Model(
                data_type=ds.data_type,
                loss_function=config['loss'],
                scaler=ds.scaler,
                steepness=ds.steepness,
                input_shape=input_shape,
                neurons=32,
                learning_rate=0.001,
                theta=config['theta'],
                tgds_ratio=config['tgds_ratio'],
                batch_size=128,
                verbose=False,
                dropout_rate=config['dropout_rate'],
                n_lstm_layers=3,
                seed=config['run'],
                gpu=int(gpu_id),
                loss_normalized=True,
                ex_name="dropout_check"
            )
            model.train(x_train, y_train, x_val=x_val, y_val=y_val, epochs=200, check=False)
            model.predict(dataset=ds, show_plot=False)
            is_deleted = model.get_result(check=False)
            if not is_deleted:
                break
    except Exception as e:
        print(str(e))
        traceback.print_exc()

    gpu_queue.put(gpu_id)


configs = []
for run in range(5):
    for loss in ['two_state', 'mean_squared_error']:  # ['two_state', 'mean_squared_error']:
        for train_size in [2, 6]:
            for dropout_rate in [0, 0.25]:
                configs.append({'run': run, 'train_size': train_size, 'loss': loss, 'dropout_rate': dropout_rate})
random.shuffle(configs)
configs = sorted(configs, key=lambda x: x['run'])


def main():
    gpus = ["0", "1", "0", "1", "0", "1"]

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
