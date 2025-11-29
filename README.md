# FAST 3D SAM: CRAFTING LARGE VISION MODELS FOR 3D MEDICAL IMAGE SEGMENTATION VIA TOKEN CLUSTERING AND FINE-TUNING

[paper](https://ieeexplore.ieee.org/document/10635248)

## Overview
> **<p align="justify"> Abstract:** *The Segment Anything Model (SAM) revolutionizes image segmentation, introducing a groundbreaking approach. Despite the promising techniques demonstrated in previous studies for developing 3D variants of SAM with additional modules, the high training parameters and computational burden impede real-time demanding applications such as interactive segmentation. To this end, we present Fast 3D SAM, a novel and lightweight 3D SAM network that integrates parameter-efficient fine-tuning and token clustering, effectively addressing the challenges of heavy computational requirements. Specifically, we propose Efficient Hyper-adapter Module and 3D Token Clustering techniques, which significantly reduce parameter count and alleviate calculation overload. Remarkably, our method outperforms previous state-of-the-art fine-tuning approaches, while requiring only 60% of the parameters and 50% of the Multiply-Accumulate (MACs) operations compared with them.* </p>

## Method

![main figure](image/model.png)

## Segmentation Results
![main figure](image/image.png)

## Data preparation
* Download the ckpt [here](https://huggingface.co/datasets/fish111111/FAST_3D_SAM).

* Download the dataset [here](https://huggingface.co/datasets/fish111111/ProtoSAM/tree/main).

* Place dataset under `datadile` like the following:
```
datadile/
|–– lung/
|   |–– imagesTr/
|   |–– labelsTr/
|   |–– split.pkl/
|–– lung2/
|   |–– imagesTr/
|   |–– labelsTr/
|   |–– split.pkl/
...
```

## Training and Evaluation
* Run the training and evaluation script

train
```bash
# train our model
bash train_iseg_tri_attn_loraAdapter_pEncodeS_miniD.sh
```

test
```bash
# random click test
bash test_iseg_simpleClick_ours.sh
# interactive simulated click test
bash _similative_click.sh
```

You can modify the parameters in the file to train and test with different parameters or models.

## Citation
If you use our work, please consider citing:
```
@INPROCEEDINGS{10635248,
  author={Qing, Zhongfei and Liu, Yan and Xu, Qian and Li, Haiyan},
  booktitle={2024 IEEE International Symposium on Biomedical Imaging (ISBI)}, 
  title={Fast 3D SAM: Crafting Large Vision Models for 3D Medical Image Segmentation via Token Clustering and Fine-Tuning}, 
  year={2024},
  volume={},
  number={},
  pages={1-4},
  keywords={Training;Image segmentation;Solid modeling;Three-dimensional displays;Computational modeling;Biological system modeling;Transformers;medical image segmentation;interactive segmentation;vision transformer;fine-tuning},
  doi={10.1109/ISBI56570.2024.10635248}
}
```

## Acknowledgements

Our code builds upon the [sam](https://github.com/facebookresearch/segment-anything) repositories. We are grateful to the authors for making their code publicly available. If you use our model or code, we kindly request that you also consider citing these foundational works.
