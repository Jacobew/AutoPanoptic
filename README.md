# Auto-Panoptic: Cooperative Multi-Component Architecture Search for Panoptic Segmentation

This repository provides the implementation of NeurIPS2020 Paper: 

[Auto-Panoptic: Cooperative Multi-Component Architecture Search for Panoptic Segmentation](https://drive.google.com/file/d/16AAx-Rdi4tO22ta0CrFAyMsaaRbgQfCF/view?usp=sharing),
and supplementary materials can be downloaded [here](https://drive.google.com/file/d/1osijWS1HcdZmW0P9Tels3nnAZbXugMRl/view?usp=sharing).

This repository is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [DetNAS](https://github.com/megvii-model/DetNAS).

## Installation
Check [INSTALL.md](https://github.com/Jacobew/AutoPanoptic/blob/master/INSTALL.md) for installation instructions.


## Data Preparation
Download and extract COCO 2017 train and val images with annotations from http://cocodataset.org. 
Following Panoptic-FPN, we predict 53 stuff classes plus a single ‘other’ class for all 80 thing classes for semantic segmentation. 
Thus we squeeze all thing classes label to id 0 and create the folder `PanopticAnnotation`.
For architecture search, we randomly split coco train set into `nas_train` and `nas_val` set (5k images). 
We provide the [download link](https://drive.google.com/file/d/16JwyqmHHTit6g5BI37DxVeOW5H3IBvS9/view?usp=sharing) here.

We expect the directory structure to be the following:
```
maskrcnn-benchmark/
    - datasets/
        - coco/
            - train2017/
            - val2017/
            - nas/
                - instances_nas_train2017.json
                - instances_nas_val2017.json
            - annotations/  
                - ...
                - PanopticAnnotation/ 
```

## Supernet Pretrain on ImageNet
We pretrain our model on ImageNet using the same search space and training schedule as [DetNAS](https://github.com/megvii-model/DetNAS).
Please follow the [instructions](https://github.com/megvii-model/DetNAS#step-2-supernet-training) here or you can download our pretrain model [autopanoptic_imagenet_pretrain.pkl](https://drive.google.com/file/d/1d6dx5M_vT0dIq33B2roya_C_df9zeP_L/view?usp=sharing) directly.

## Supernet Finetuning & Architecture Search on COCO
```bash
export CURRENT_DIR={your_root_dir}
cd $CURRENT_DIR
sh scripts/architecture_search.sh
```
We provide our search log [here](https://drive.google.com/file/d/1ZSkXjBBjYYO3gLIR0reGpKD0YkpMtu1D/view?usp=sharing) and searched architecture in `maskrcnn-benchmark/test_models/`.
Note that you should change `MODEL.WEIGHT` to the correct pretrain model path before architecture search.

## Pretrain Searched Model on ImageNet
```bash
export CURRENT_DIR={your_root_dir}
cd $CURRENT_DIR
sh scripts/pretrain_searched_model.sh
```
We provide our imagenet pretrain model [here](https://drive.google.com/file/d/1amkCFrHj7JfnJ6YmjGNO_YatmTZl3Lyn/view?usp=sharing).

## Retrain Searched Model on COCO
```bash
export CURRENT_DIR={your_root_dir}
cd $CURRENT_DIR
sh scripts/train_searched_model.sh
```
We provide our [training log](https://drive.google.com/file/d/1uTZ5GZj0YQxl6uirEqagKv7tI29M22zO/view?usp=sharing) and [panoptic model](https://drive.google.com/file/d/15qiYYNUYqpR0UBGDzbX3jz_UQe8V3sxw/view?usp=sharing). 
Note that you should change `MODEL.WEIGHT` to the correct imagenet pretrain model path before retraining.

## Evaluation on COCO
```
bash
export CURRENT_DIR={your_root_dir}
cd $CURRENT_DIR
sh scripts/eval.sh
```
We provide our [evaluation log](https://drive.google.com/file/d/1eTybVn4PErK-b4Csystgjn6ljIYCUvSk/view?usp=sharing).

## Results
| Method  | PQ | PQ_thing | PQ_stuff  |
|----------|--------|-----------|-----------|
| UPSNet |  42.5 | 48.5 | 33.4 |
| BGRNet | 43.2 | 49.8 | 33.4 |
| SOGNet | 43.7 | 50.6 | 33.2 |
| **Auto-Panoptic(Ours)** | **44.8** | **51.4** | **35.0** |

## Citations
Will be available once published.