# A case for machine learning in CFD

This repository contains the code used in the article [**A case for machine learning in CFD**](https://medium.com/@mskarysz_35929/a-case-for-machine-learning-in-cfd-3aca27aaca76) published by [**FlowFusic**](https://www.flowfusic.com/).

<a href="https://www.flowfusic.com/">
<p align="center">
  <img src="https://www.flowfusic.com/brandmark-design_transparent_for_web.png" width="350">
</p>
</a>

## Installation

Clone the repository:
  ```bash
  git clone https://github.com/flowfusic/flowfusic_airfoil_unet.git
  ```
  ```bash
  cd flowfusic_airfoil_unet
  ```

We recommend using a separate virtual environmet. To create one, run

  ```bash
  python3 -m venv VENV
  ```

Enter it by running

  ```bash
  source VENV/bin/activate
  ```

To install the requirements run

  ```bash
  pip install -r requirements.txt
  ```

## Using the codebase
There are three main steps in creatin a machine learning model, which could replace parts of the CFD pipeline. Before going further, make sure that you are in the repository's main directory.

1. Generating geometries

    ```bash
    python make_geometry_dataset.py
    ```
  
    This will create a geometry set in ```./data```. By default, it creates 20 train examples, 20 validation and 20 test. To generate dataset of different size       (e.g., 1000 train, 200 validate, 200 test) run
    
    ```bash
    python make_geometry_dataset.py --train_samples=1000 --validation_samples=200 --test_samples=200
    ```


2. Generating sample flows

    *comming soon*

3. Training the Convolutional Neural Network model

    We recommend using GPU for training. On flowfusic run

    ```bash
    ff alloc gpu
    ff connect
    ```
    Note that this option is not available on Free account. Please contact support@flowfusic.com for more details.

    Navigate to the project and run
    ```bash
    python3 train.py
    ```

4. Test the trained model

    ```bash
    python3 test.py
    ```

