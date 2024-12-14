## Hybrid Attention-based Multi-task Vehicle Motion Prediction Using Non-Autoregressive Transformer and Mixture of Experts

This is offical repository for the paper "Hybrid Attention-based Multi-task Vehicle Motion Prediction Using Non-Autoregressive Transformer and Mixture of Experts" by [Hao Jiang](https://sunstroperao.github.io/)

### Set up the environment
#### method 1: using conda
```bash
conda env create -f environment.yml
conda activate motion_prediction
```
#### method 2: using pip
```bash
conda create -n env_name python=3.10
conda activate env_name
pip install -r requirements.txt
```
### Prepare the dataset
Download the ngsim dataset from [here](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj) and highD dataset from [here](https://www.highd-dataset.com/). Then use [preprocess_*.m](./data/) in the data folder to preprocess the dataset.
### data structure
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
bash train.sh 
```
### Evaluate the model
```bash
python evaluate.py 
```

### Citation
If you find this work useful, please consider citing:
```
@article{jiang2021hybrid,
  title={Hybrid Attention-based Multi-task Vehicle Motion Prediction Using Non-Autoregressive Transformer and Mixture of Experts},
  author={Jiang, Hao and Li, Zhi and Zhang, Yuxuan and Wang, Yizhou and Wang, Jia and Li, Bo},
  journal={arXiv preprint arXiv:2109.06794},
  year={2021}
}
```

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
```
