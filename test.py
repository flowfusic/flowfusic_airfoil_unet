import os
import time
import torch

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm

from src.networks import UNet
from src.dataset import SimulationDataset

BASE_DIR = os.path.join("data", "flow", "test")
SAVE_DIR = os.path.join("results")
test_dataset = SimulationDataset(base_dir=BASE_DIR)


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

print(BASE_DIR)

for i, data_sample in tqdm(enumerate(test_dataset)):

    if i > 20:
        break

    geometry_array = data_sample["geometry"]
    flow_array = data_sample["flow"]

    # Load model and make prediction based on geometry
    path_to_model = os.path.join("models", "model_checkpoint.pt")
    model = UNet()
    model.load_state_dict(torch.load(path_to_model))
    model.eval()

    geometry = torch.from_numpy(geometry_array)
    geometry = geometry.unsqueeze(0)
    with Timer('Feed forward'):
        prediction = model(geometry)

    # Postprocessing
    geometry = np.transpose(geometry_array, [1, 2, 0])

    prediction = prediction.squeeze(0)
    prediction = prediction.permute(1, 2, 0)
    prediction = prediction.detach().numpy()
    flow_array = np.transpose(flow_array, (1, 2, 0))

    difference = prediction - flow_array
    magnitude = np.linalg.norm(difference, axis=2)

    fig = plt.figure(figsize=(12.0, 9.0))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(4, 2),
        axes_pad=0.5,
    )

    grid[0].imshow(flow_array[:, :, 2])
    grid[0].set_title("pressure: OpenFAOM")
    grid[1].imshow(prediction[:, :, 2])
    grid[1].set_title("pressure: Neural network")
    grid[2].imshow(flow_array[:, :, 0])
    grid[2].set_title("Velocity x-component: OpenFAOM")
    grid[3].imshow(prediction[:, :, 0])
    grid[3].set_title("Velocity x-component: Neural network")
    grid[4].imshow(flow_array[:, :, 1])
    grid[4].set_title("Velocity y-component: OpenFAOM")
    grid[5].imshow(prediction[:, :, 1])
    grid[5].set_title("Velocity y-component: Neural network")
    grid[6].imshow(geometry.squeeze())
    grid[6].set_title("Computational domain")
    grid[7].imshow(magnitude)
    grid[7].set_title("Error distribution")

    subdir_name = str(i).zfill(6)
    dir_to_save = os.path.join(SAVE_DIR, subdir_name)
    path_to_save = os.path.join(dir_to_save, "simulation_comparison.png")

    try:
        os.makedirs(dir_to_save)
    except OSError:
        pass

    plt.savefig(fname=path_to_save)
