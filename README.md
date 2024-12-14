## Hybrid Attention-based Multi-task Vehicle Motion Prediction Using Non-Autoregressive Transformer and Mixture of Experts

This is offical repository for the paper "Hybrid Attention-based Multi-task Vehicle Motion Prediction Using Non-Autoregressive Transformer and Mixture of Experts" by [Hao Jiang](https://sunstroperao.github.io/)

### Set up the environment
```bash
conda env create -f environment.yml
conda activate motion_prediction
```
### Download the dataset
```bash 
cd data
bash download_data.sh
```
### Train the model
```bash
python train.py --config config/transformer.yaml
```
### Evaluate the model
```bash
python evaluate.py --config config/transformer.yaml
```
### Visualize the prediction
```bash
python visualize.py --config config/transformer.yaml
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
### Acknowledgement
This code is based on the [Trajectron++](https://sunstroperao.github.io/)

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
```
