<div align="center">
  <h1>Relational test trest Embedding for Few-Shot Classification <br> (ICCV 2021)</h1>
</div>

## :bqcaihong,1=&#x1F308; ：高级人工智能作业
* 尝试针对这个项目进行一些测试和修改
* 尝试使用Transformer进行修改这个项目

## :heavy_check_mark: Requirements
Ubuntu 16.04
* Python 3.7
* [CUDA 11.0](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.7.1](https://pytorch.org)


## :gear: Conda environmnet installation
```bash
conda env create --name renet_iccv21 --file environment.yml
conda activate renet_iccv21
```

## :books: Datasets
```bash
cd datasets
bash download_miniimagenet.sh
bash download_cub.sh
bash download_cifar_fs.sh
bash download_tieredimagenet.sh
```

## :deciduous_tree: Authors' checkpoints

```bash
cd checkpoints
bash download_checkpoints_renet.sh
```
The file structure should be as follows:


    
    renet/
    ├── datasets/
    ├── model/
    ├── scripts/
    ├── checkpoints/
    │   ├── cifar_fs/
    │   ├── cub/
    │   ├── miniimagenet/
    │   └── tieredimagenet/
    train.py
    test.py
    README.md
    environment.yml
    
    
    
   
## :pushpin: Quick start: testing scripts
To test in the 5-way K-shot setting:
```bash
bash scripts/test/{dataset_name}_5wKs.sh
```
For example, to test ReNet on the miniImagenet dataset in the 5-way 1-shot setting:
```bash
bash scripts/test/miniimagenet_5w1s.sh
```

## :fire: Training scripts
To train in the 5-way K-shot setting:
```bash
bash scripts/train/{dataset_name}_5wKs.sh
```
For example, to train ReNet on the CUB dataset in the 5-way 1-shot setting:
```bash
bash scripts/train/cub_5w1s.sh
```


Training & testing a 5-way 1-shot model on the CUB dataset using a TitanRTX 3090 GPU takes **41m 30s**.

## :art: Few-shot classification results
Experimental results on few-shot classification datasets with ResNet-12 backbone. We report average results with 2,000 randomly sampled episodes.

<table>
  <tr>
    <td>datasets</td>
    <td colspan="2" align="center">miniImageNet</td>
    <td colspan="2" align="center">tieredImageNet</td>
  </tr>
  <tr>
    <td>setups</td>
    <td>5-way 1-shot</td>
    <td>5-way 5-shot</td>
    <td>5-way 1-shot</td>
    <td>5-way 5-shot</td>
  </tr>
  <tr>
    <td>accuracy</td>
    <td align="center">67.60</td>
    <td align="center">82.58</td>
    <td align="center">71.61</td>
    <td align="center">85.28</td>
  </tr>
</table>

<table>
  <tr>
    <td>datasets</td>
    <td colspan="2" align="center">CUB-200-2011</td>
    <td colspan="2" align="center">CIFAR-FS</td>
  </tr>
  <tr>
    <td>setups</td>
    <td>5-way 1-shot</td>
    <td>5-way 5-shot</td>
    <td>5-way 1-shot</td>
    <td>5-way 5-shot</td>
  </tr>
  <tr>
    <td>accuracy</td>
    <td align="center">79.49</td>
    <td align="center">91.11</td>
    <td align="center">74.51</td>
    <td align="center">86.60</td>
  </tr>
</table>


## :mag: Related repos
Our project references the codes in the following repos:

* Zhang _et al_., [DeepEMD](https://github.com/icoz69/DeepEMD).
* Ye _et al_., [FEAT](https://github.com/Sha-Lab/FEAT)
* Wang _et al_., [Non-local neural networks](https://github.com/AlexHex7/Non-local_pytorch)
* Ramachandran _et al_., [Stand-alone self-attention](https://github.com/leaderj1001/Stand-Alone-Self-Attention)
* Huang _et al_., [DCCNet](https://github.com/ShuaiyiHuang/DCCNet)
* Yang _et al_., [VCN](https://github.com/gengshan-y/VCN)

## :love_letter: Acknowledgement
* We adopted the main code bases from [DeepEMD](https://github.com/icoz69/DeepEMD)
* CSDN解释的网址：https://blog.csdn.net/qq_42578970/article/details/120875948（2021，10，21）
