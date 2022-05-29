# Satellite-Based-Counting

Code for Drone-Based Car Counting via Density Map Learning (VCIP 2020) and Satellite-Based Object Counting via Adaptive Density Map Assisted Learning.

Pre-trained models
---
[Google Drive](https://drive.google.com/drive/folders/1UAuLVEDF4HUk8WtASATrGz_VjTZUcAqI?usp=sharing)

[Baidu Cloud](https://pan.baidu.com/s/1pbfkCU6ROBTVPsBpmuLtCA) : 6svf

Environment
---
We are good in the environment:

python 3.6

CUDA 9.2

Pytorch 1.2.0

numpy 1.19.2

matplotlib 3.3.4

Usage
---
We provide the test code for our model. 
The `adml_small_vehicle.pth` model is adapted on the RSOC_small-vehicle dataset. 
We randomly select an image from the RSOC_small-vehicle dataset and place it in the image folder.
And you can either choose the other images for a test.

We are good to run:

```
python test.py --model ADML --mode DME --model_state ./model/adml_small_vehicle.pth --out ./out/out.png
```

We will release more trained models soon.
The core code will be released after the journal paper is accepted.
Please see the paper for more details.

Data set
---
We propose a Tree data set, The download link is:

[Baidu Cloud](https://pan.baidu.com/s/1pjnrCqKeaucwhuoDXaxzZg?pwd=Tree) : Tree

We have only shared the training and validation set images and annotations.
If you are interested in this data set, please contact us (Email address at the bottom) for a test set.


Citation
---

```
@INPROCEEDINGS{9301785,
  author={Huang, Jingxian and Ding, Guanchen and Guo, Yujia and Yang, Daiqin and Wang, Sihan and Wang, Tao and Zhang, Yunfei},
  booktitle={2020 IEEE International Conference on Visual Communications and Image Processing (VCIP)}, 
  title={Drone-Based Car Counting via Density Map Learning}, 
  year={2020},
  volume={},
  number={},
  pages={239-242},
  doi={10.1109/VCIP49819.2020.9301785}}
```

Acknowledgement
---

Thanks to these repositories
- [C-3 Framework](https://github.com/gjy3035/C-3-Framework)
- [EDSC-pytorch](https://github.com/Xianhang/EDSC-pytorch)
- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

If you have any question, please feel free to contact us. (gcding@whu.edu.cn and ceoilmp@whu.edu.cn)
