# [AAAI 2025] Neural Conformal Control for Time Series Forecasting
This is the official implementation of "[Neural Conformal Control for Time Series Forecasting](https://arxiv.org/pdf/2412.18144)" (NCC) appearing in AAAI 2025 (main track). Authors are Ruipu Li and Alexander Rodríguez from the University of Michigan.

In this repository, we provide the data and code used in our paper's experiments. Code for model, NCC, can be found in `end2endmodel.py`. To visualize results, use `show_results.ipynb`.

## How to run
To run the code, clone this repo and follow the instructions below.

### Setup environment
Create an venv with python 3.7. Then run `bin/setup.sh` to create the necessary directories and install dependencies.

### Step 1: Data preparation
The five datasets we used in our paper (***flu hospitalization***, ***Covid-19 hospitalization***, ***SMD***, ***Electricity***, ***Weather***) are already pre-processed. The raw data of these datasets are put in `data/` folder in csv format. The `<dataset>_clip_idx.pkl` files in the folder are the indexes we use for evaluation. For example, for the ***covid hospitalization*** dataset, we only evaluate during the respiratory season. The indexes are used to select the respiratory season.

### Step 2: Get predictions from base forecaster
First use `src/forecaster/online_training.py` to get the predictions from the base forecaster. Configure the parameters in `setup/exp_params/`. For example, to run experiment 1, use `setup/exp_params/1.yaml`:
` python online_training.py -i=1 `
Upon finish, a file under `results/base_pred/` is generated. The base prediction files for our datasets can be downloaded from this [link](https://drive.google.com/drive/folders/1RE0Wwfxvz0_mv9Q2gbNZIzGkL23_dHKG?usp=sharing).

:exclamation: If you want to use our method on a new dataset <new_data>, you can load the data using `src/forecaster/load_std_data.py`, following the steps below:
+ **Add the data file to the `data/` folder**: The file should be in **csv** format. The columns should contain the data features. And one of the data features will be used as the target. Both the data features and target will need to be specified in the setup files. For an example of how this csv file looks like, check `data/smd.csv`.
+ **Add corresponding setup files**: Two files need to be added to the `setup/` folder. The first file is `setup/<new_data>.yaml`, which defines the name of the dataset, the target, the path of the data file and some data parameters. See `setup/smd.yaml` as an example. Note that if the 'index_column' is 'NA', then the code will generate the index as integers automatically. If the 'data_features' is an empty list, all the columns in the csv file will be used as data features. The second file is `setup/exp_params/<exp_id>.yaml`. This file defines parameters that are different in different experiments. Note that you can use this file to overwrite the parameters defined in `setup/<new_data>.yaml` and `setup/seq2seq.yaml`. The 'dataset' should match that in `setup/<new_data>.yaml`. See `setup/exp_params/35.yaml` as an example for the smd dataset.
+ **Modify the code**: In `src/forecaster/online_training.py`, make modifications to the code in the section **add new dataset**.
+ :tada: Done! You can now use our method on your own dataset!


:exclamation: To use a new base forecaster:
+ **Add code**: Add the code for the base forecaster to `src/forecaster/` folder. 
+ **Modify the code**: Modify model initialization part and forward pass in `online_training.py` and `utils.py` so that it is compatible with the NCC model.

### Step 3: Calibrate the predictions with NCC
Now with the predictions, you can use e2ecp.py to construct the uncertainty bounds. First configure the parameters in `setup/cp_exp_params/<exp_id>.yaml`. Then to train a model for calibration, run:
` python e2ecp.py -i=<exp_id> -t`
This will save a trained model's parameters as `models/<exp_id>.pt`. Then to calibrate for the test dataset, run:
` python e2ecp.py -i=<exp_id>`
This generates a `<exp_id>_<seed>.pkl` file under the `results/` folder. The results can be downloaded directly from this [link]([https://](https://drive.google.com/drive/folders/1KfYcmDPUVc0vJgN6QZ2eB7Z_DwDNIJE6?usp=sharing))

#### How to tune hyperparameters for your own dataset or base forecasters?
The `bayesian_tuning.py` file implements bayesian optimization for hyperparameter tuning. Modify the parameter space and black box function to set the hyperparameters you would like to tune.

### Step 4: Visualize the results
Visualize and analyze the results using `show_results.ipynb` under `src/notebooks/` folder.

## Contact
If you have any questions, please contact Ruipu Li at liruipu@umich.edu.

## Cite our work
If you find this work helpful, cite our work:
<!-- Li, Ruipu, and Alexander Rodríguez. "Neural Conformal Control for Time Series Forecasting." arXiv preprint arXiv:2412.18144 (2024).
```bibtex
@article{li2024neural,
  title={Neural Conformal Control for Time Series Forecasting},
  author={Li, Ruipu and Rodr{\'\i}guez, Alexander},
  journal={arXiv preprint arXiv:2412.18144},
  year={2024}
}
``` -->
Li, Ruipu, and Alexander Rodríguez. "Neural Conformal Control for Time Series Forecasting". In Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39.
```bibtex
@article{li2024neural,
  title={Neural Conformal Control for Time Series Forecasting},
  author={Li, Ruipu and Rodr'\iguez, Alexander},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  year={2025}
}
```

## Credits and Attributions

This project includes code from the following sources. We are grateful for their contributions that made this work possible.

1. [conformal-time-series](https://github.com/aangelopoulos/conformal-time-series)  
   Description: We modify the code from this repo to reimplement ACI and conformal-PID in our experiments. 

2. [SPCI code](https://github.com/hamrel-cxu/SPCI-code)  
   Description: We implement NexCP based on the code provided in this repo.
