<div align="center">
  <h1>Relational test trest Embedding for Few-Shot Classification <br> (ICCV 2021)</h1>
</div>

## :scroll: 高级人工智能作业  &#x1F308; 
* 尝试针对这个项目进行一些测试和修改
* 尝试使用Visual Transformer的进行修改这个项目
* 结合SSFormer对support和query进行修改


## :heavy_check_mark: Requirements
Ubuntu 16.04
* Python 3.7
* [CUDA 11.0](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.7.1](https://pytorch.org) 1.0.1版本也可以跑
* enipos 0.3.2 
* torch 1.0.1
* torchvision 0.2.2
* wandb 0.12.6
* tqdm 4.62.3

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
训练的记录参考：https://wandb.ai/tjudyk

![](https://github.com/TJUdyk/renet/blob/main/%E8%AE%AD%E7%BB%83%E8%AE%B0%E5%BD%95.png)

Training & testing a 5-way 1-shot model on the CIFAR dataset using 4 NVIDIA 2080Ti GPU takes **7.29h left**.

Training & testing a 5-way 5-shot model on the CIFAR dataset using 4 NVIDIA 2080Ti GPU takes **4.32h left**.

Training & testing a 5-way 1-shot model on the CUB dataset using 4 NVIDIA 2080Ti GPU takes **1.7h**.

Training & testing a 5-way 5-shot model on the CUB dataset using 4 NVIDIA 2080Ti GPU takes **1.16h**.

Training & testing a 5-way 1-shot model on the ImageNet dataset using 4 NVIDIA 2080Ti GPU takes **5.45h**.

Training & testing a 5-way 5-shot model on the ImageNet dataset using 4 NVIDIA 2080Ti GPU takes **2.61h**.

## :art: Few-shot classification results
Experimental results on few-shot classification datasets with ResNet-12 backbone. We report average results with 2,000 randomly sampled episodes.


<table>
  <tr>
    <td>datasets</td>
    <td colspan="2" align="center">miniImageNet</td>
  </tr>
  <tr>
    <td>setups</td>
    <td>5-way 1-shot #24</td>
    <td>5-way 5-shot #19</td>
  </tr>
   <tr>
    <td>accuracy</td>
    <td align="center">85.54</td>
    <td align="center">85.54</td>
  </tr>
  <tr>
    <td>accuracy</td>
    <td align="center">67.60</td>
    <td align="center">90.42666/81.554</td>
  </tr>
  <tr>
    <td>accuracy_3channel</td>
    <td align="center">67.96</td>
    <td align="center">82.13</td>
  </tr>
  <tr>
    <td>accuracy_3*3kernel</td>
    <td align="center">84.9489/67.8/66.39668</td>
    <td align="center">92.16002/82.60001/82.52333</td>
  </tr>
  <tr>
    <td>accuracy_3channel_SCR</td>
    <td align="center">77.76444/66.23333</td>
    <td align="center">89.46223/81.43334</td>
  </tr>
  <tr>
    <td>SENet</td>
    <td align="center">85.6/64.16534/66.01334/64.165 +- 0.433</td>
    <td align="center">93.20445/79.84868/81.36001/79.849 +- 0.322</td>
  </tr>
   <tr>
    <td>CBAM</td>
    <td align="center">待测试</td>
    <td align="center">待测试</td>
  </tr>
    <tr>
    <td>ResNet_CBAM_SCR_2test</td>
    <td align="center">86.99556/ 63.54933/66.29333</td>
    <td align="center">66.293333</td>
  </tr>
  </tr>
    <tr>
    <td>SCR_NOSC_CCA</td>
    <td align="center">83.73777/65.88066/67.76001/ 65.881 +- 0.437</td>
    <td align="center">66.293333</td>
  </tr>
</table>


<table>
  <tr>
    <td>datasets</td>
    <td colspan="2" align="center">CUB-200-2011</td>
  </tr>
  <tr>
    <td>setups</td>
    <td>5-way 1-shot #15</td>
    <td>5-way 5-shot #14</td>
  </tr>
    <tr>
    <td>SOTA</td>
    <td align="center">95.48</td>
    <td align="center">96.43</td>
  </tr>
  <tr>
    <td>basic</td>
    <td align="center">79.49</td>
    <td align="center">91.11</td>
  </tr>
  <tr>
    <td>accuracy_3channel</td>
    <td align="center">82.29112/65.75334/74.87333</td>
    <td align="center">95.26086/92.066667/91.037</td>
  </tr>
  <tr>
    <td>accuracy_2*3kernel</td>
    <td align="center">88.8695/82.70666/79.6839</td>
    <td align="center">95.26086/92.06667/91.037</td>
  </tr>
  <tr>
    <td>accuracy_3channel_SCR</td>
    <td align="center">待测试</td>
    <td align="center">91.66</td>
  </tr>
  <tr>
    <td>accuracy_2branch_SCR</td>
    <td align="center">84.02/81.56</td>
    <td align="center">92.17</td>
  </tr>
  <tr>
    <td>SCE</td>
    <td align="center">80.85508/78.46399/78.464 +- 0.446</td>
    <td align="center">92.04346/89.636/91.10667/ 89.636 +- 0.268</td>
  </tr>
 <tr>
    <td>SENet</td>
    <td align="center">84.28985/77.97466/81.88/77.975 +- 0.439</td>
    <td align="center">93.94204/ 90.32533/91.09332/90.325 +- 0.256</td>
  </tr>
  <tr>
    <td>SENet_CCA</td>
    <td align="center">84.57972/78.53667/81.70667/78.537 +- 0.444</td>
    <td align="center">test</td>
  </tr>
  <tr>
    <td>SENet_CCA</td>
    <td align="center">84.59422/79.13735/82.14667/79.137 +- 0.437</td>
    <td align="center">93.94204/ 90.32533/91.09332/90.325 +- 0.256</td>
  </tr>
  <tr>
    <td>ResNet_CBAM_SCR_1test</td>
    <td align="center">87.40581/79.89802/82.08667/79.898 +- 0.430</td>
    <td align="center">94.18842/90.61333/92.19333/90.613 +- 0.254</td>
  </tr>
  <tr>
    <td>ResNet_CBAM_SCR_2test</td>
    <td align="center">待测试</td>
    <td align="center">94.02898/91.51933/92.97333/91.519 +- 0.243</td>
  </tr>

  <tr>
    <td>LSA_CCA</td>
    <td align="center">79.191 +- 0.438</td>
    <td align="center">5-way 5-shot </td>
  </tr>
  <tr>
    <td>SCR_NOSC_CCA</td>
    <td align="center">79.131 +- 0.430</td>
    <td align="center">5-way 5-shot </td>
  </tr>
  <tr>
    <td>SCR_NOSC_CBAM4_CCA</td>
    <td align="center">79.581 +- 0.425</td>
    <td align="center">5-way 5-shot </td>
  </tr>
  <tr>
    <td>SCR_NOSC_1x1_CBAM2_2x2_CCA</td>
    <td align="center"> 86.43478/89.55668/82.11333/79.557 +- 0.439</td>
    <td align="center">95.13046/92.38667/91.50067/| 91.501 +- 0.234 </td>
  </tr>
</table>



<table>
  <tr>
    <td>datasets</td>
    <td colspan="2" align="center">CIFAR-FS</td>
  </tr>
  <tr>
    <td>setups</td>
    <td>5-way 1-shot #12</td>
    <td>5-way 5-shot #7</td>
  </tr>
    <tr>
    <td>SOTA</td>
    <td align="center">84.44</td>
    <td align="center">91.86</td>
  </tr>
  <tr>
    <td>Basic</td>
    <td align="center">74.51</td>
    <td align="center">86.60</td>
  </tr>
  <tr>
    <td>accuracy_3channel</td>
    <td align="center">74.38532/65.527</td>
    <td align="center">87.397</td>
  </tr>
  <tr>
    <td>accuracy_2*3kernel</td>
    <td align="center">83.29112/65.75334/74.87333</td>
    <td align="center">92.59113/79.4668/87.056</td>
  </tr>
  <tr>
    <td>accuracy_3channel_SCR</td>
    <td align="center">待测试</td>
    <td align="center">待测试</td>
  </tr>
  <tr>
    <td>accuracy_2branch_SCR</td>
    <td align="center">80.8822/74.47/65.96667/74.470 +- 0.461</td>
    <td align="center">90.31111/86.38734/79.95335/86.387 +- 0.333</td>
  </tr>
  <tr>
    <td>SCE</td>
    <td align="center">待测试</td>
    <td align="center"> 92.12667/85.11066/78.24/85.111 +- 0.346</td>
  </tr>
 <tr>
    <td>SENet</td>
    <td align="center">待测试</td>
    <td align="center">待测试</td>
  </tr>
  <tr>
    <td>SENet</td>
    <td align="center">83.52222/73.06734/63.82666/73.067 +- 0.468</td>
    <td align="center">92.55112/85.64867/78.88/85.649 +- 0.339</td>
  </tr>
  <tr>
    <td>ResNet_CBAM_SCR_1test</td>
    <td align="center">待测试</td>
    <td align="center">待测试</td>
  </tr>
  <tr>
    <td>ResNet_CBAM_SCR_2test</td>
    <td align="center">待测试</td>
    <td align="center">待测试</td>
  </tr>
</table>

<table>
  <tr>
    <td>CUB数据集测试</td>
    <td>5-way 1-shot </td>
    <td>5-way 5-shot </td>
  </tr>
  <tr>
    <td>SCR_CCA</td>
    <td>79.49</td>
    <td>91.11</td>
  </tr>
  <tr>
    <td>SCR_CCA_test</td>
    <td>79.647</td>
    <td>91.11</td>
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
* chenhaoxing_.,[SSFormer](https://github.com/chenhaoxing/SSFormers)
* yhu01/PT-MAP.,[PT-MAP](https://github.com/TJUdyk/PT-MAP)
* alihassanijr [CCT](https://github.com/SHI-Labs/Compact-Transformers)

## :love_letter: Acknowledgement
* We adopted the main code bases from [DeepEMD](https://github.com/icoz69/DeepEMD)
* CSDN解释的网址：https://blog.csdn.net/qq_42578970/article/details/120875948（2021，10，21）
* 小样本图片分类的榜单：https://paperswithcode.com/task/few-shot-image-classification
