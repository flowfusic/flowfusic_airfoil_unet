# A case for machine learning in CFD

This repository contains the code used in the article [**A case for machine learning in CFD**](https://medium.com/@mskarysz_35929/a-case-for-machine-learning-in-cfd-3aca27aaca76) published by [**FlowFusic**](https://www.flowfusic.com/).

<a href="https://www.flowfusic.com/">
<p align="center">
  <img src="https://www.flowfusic.com/brandmark-design_transparent_for_web.png" width="350">
</p>
</a>

## Installation

We recommend using a separate virtual environmet. To create one, run

```bash
python3 -m venv <ENV_NAME>
```

Enter it by running

```bash
source <ENV_NAME>/bin/activate
```

To install the requirements run

```bash
cd flowfusic_airfoil_unet
pip install -r requirements.txt
```

## Using the codebase
There are three main steps in creatin a machine learning model, which could replace parts of the CFD pipeline. Before going further, make sure that you are in the repository's main directory.

1. Generating geometries

```
python make_geometry_dataset.py
```

2. Generating sample flows

3. Training the Convolutional Neural Network model

```
python train.py
```
