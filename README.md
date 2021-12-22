# Pix2seq: A Language Modeling Framework for Object Detection
We provide an unofficial re-implementation for [Pix2Seq](https://arxiv.org/abs/2109.10852v1). It is mainly developped based on [Pretrained-Pix2Seq](https://github.com/gaopengcuhk/Pretrained-Pix2Seq). Our target is to fully reproduce the accuracy provided in paper. With limited resource, we only try training with resolution $640\times640$, $200$ epochs as in ablation study.

If you have any ideas, please feel free to let us know.

There are two branches in this repo:
- **Generate**: Version with only sequence generation.
- [SeqAugment](): Version with sequence augmentation introduced in paper.


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

## Training

First, link coco dataset to the project folder
```
ln -s /path/to/coco ./coco 
```

Training
```
bash ./resnet50_pretrained.sh 8
```

## Evaluation
Nucleus search and bias for EOS are optional for evaluation.

Vanilla evaluation
```
bash ./resnet50_pretrained.sh 8 --eval --resume /path/to/checkpoint/file
```

Evaluation with nucleus search
```
bash ./resnet50_pretrained.sh 8 --eval --resume /path/to/checkpoint/file --eval_p 0.4
```

### COCO 

We provide AP

| Branch     | backbone       | Input Size | Epoch | Batch Size | AP   | +nucleus  | Weights |
| :-----:    | :------------: | :---------:| :----:| :---------:| :---:| :-------: | :-----: |
| Generate   | R50-pretrained | 640        | 100   | 128        | 29.6 | 31.1      | [not ready]() | 
| Generate   | R101-scratch   | 640        | 200   | 128        | 29.1 | 30.4      | [not ready]() | 
| SeqAugment | R50-pretrained | 640        | 100   | 128        | ---- | ----      | [not ready]() | 
| SeqAugment | R101-scratch   | 640        | 200   | 128        | ---- | ----      | [not ready]() | 

# Acknowledegement

This repo borrows a lot from [Pretrained-Pix2Seq](https://github.com/gaopengcuhk/Pretrained-Pix2Seq) and [DETR](https://github.com/facebookresearch/detr). Thanks a lot!