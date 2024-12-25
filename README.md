## Hybrid Attention-based Multi-task Vehicle Motion Prediction Using Non-Autoregressive Transformer and Mixture of Experts

![Python](https://img.shields.io/badge/python-3.10-blue.svg) ![PyTorch](https://img.shields.io/badge/pytorch-1.9.0-red.svg) ![License](https://img.shields.io/badge/license-apache-blue.svg) ![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg) ![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg) ![GitHub issues](https://img.shields.io/github/issues/sunstroperao/TAME)

![image](./picture/framework.png)

This is offical repository for the paper "Hybrid Attention-based Multi-task Vehicle Motion Prediction Using Non-Autoregressive Transformer and Mixture of Experts". The code is implemented in PyTorch.

### Set up the environment
#### method 1: using conda
```bash
conda env create -f environment.yaml
conda activate tame
```
#### method 2: using pip
```bash
conda create -n tame python=3.10
conda activate tame
pip install -r requirements.txt
```
### Prepare the dataset
Download the NGSIM US-101 and I-80 dataset from [here](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj) and highD dataset from [here](https://levelxdata.com/highd-dataset/). Then use [preprocess_*.m](./data/) in the data folder to preprocess the dataset.
#### data structure
```
data
├── ngsimdata
│   ├── TrainSet.mat
│   ├── TestSet.mat
│   ├── ValSet.mat
├── highDdata
│   ├── TrainSet.mat
│   ├── TestSet.mat
│   ├── ValSet.mat
```

### Train the model
```bash
cd method
bash train.sh # train the model in DDP mode
# or
python train.py # train the model in single GPU mode
```
### Evaluate the model
```bash
cd method
python evaluate.py 
```

### Citation
If you find this work useful, please consider citing (coming soon):
```

```

### License
This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.

