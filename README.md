# Pix2seq: A Language Modeling Framework for Object Detection
This is an unofficial re-implementation for [Pix2Seq](https://arxiv.org/abs/2109.10852v1). It is mainly developped based on [Pretrained-Pix2Seq](https://github.com/gaopengcuhk/Pretrained-Pix2Seq) and [Pix2Seq](https://github.com/google-research/pix2seq).

If you have any ideas, please feel free to let us know.


## Installation

Install PyTorch 1.5+ and torchvision 0.6+ (recommend torch1.8.1 torchvision 0.8.0)

Install pycocotools (for evaluation on COCO):

```
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

That's it, should be good to train and evaluate detection models.

## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

Please link coco dataset to the project folder
```
ln -s /path/to/coco ./coco 
```

## Training

*Not Ready.*

## Evaluation
`top_k` and `top_p` are tunable parameters for evaluation.

```
bash scripts/resnet50_pretrained.sh 8 --eval --resume /path/to/checkpoint/file
```

### COCO 

We provide AP

| Backbone       | Input Size | Epoch | Batch Size | AP   | Weights | Comments  |
| :------------: | :---------:| :----:| :---------:| :---:| :-----: | :-------: |
| R50            | 640        | -     | -          | 39.3 | [Weight](https://drive.google.com/file/d/1ykR5QMVrW0yGSmrs9cpWI5LrxXECX0Ox/view?usp=sharing) | [Official](https://console.cloud.google.com/storage/browser/pix2seq/coco_det_finetune/resnet_640x640) |

### Official Model

Convert the official model with `scripts/convert_official.py`.

# Acknowledegement

This repo borrows a lot from [Pix2Seq](https://github.com/google-research/pix2seq), [Pretrained-Pix2Seq](https://github.com/gaopengcuhk/Pretrained-Pix2Seq) and [DETR](https://github.com/facebookresearch/detr). Thanks a lot!
