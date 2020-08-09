### Introduction
There are some codes for implementing the paper "CVPR19-DMPHN（Deep Stacked Hierarchical Multi-Patch Network for Image Deblurring）"

We implement that work by using PyTorch.

Before using those codes, you have to check the enviornment.

|Package| Version|
|---|---|
|Python|3.6.7|
|PyTorch|>1.0.0|
|torchvision|0.2.1|
|tensorflow|1.8.0|
|tensorboard|1.8.0|
|tensorboardx|2.1|


### Usage

You can change the config.yml for different requirements.

Here are some tips about the config file:

#### training parameter
- epoch: 3000
- lr: 1e-4  
- decay_rate: 0.1
- batch_size: 6
- solver: 
    - 'adam'
    - 'sgd'
- device: 
    - 'cuda:0'
    - 'cpu'

#### other settings
- mode: 
    - 'train'
    - 'test'
- model_path: 
    - '.../checkpoints/002_lr_1e-4_deblur.pth'
    - 'none' **for the first training**
- data_path: '/home/yourname/data/gopro/'
- save_path: '/home/yourname/workspace/deblur-projects/cvpr19-dmphn/checkpoints/'
- save_name: '002_lr_1e-4_deblur.pth'

#### GoPro Dataset

You could download the GoPro dataset at [link](https://github.com/SeungjunNah/DeepDeblur_release)

You have to change some paths which are related to the dataset or checkpoints.

> I will release the pre-trained model in the future.

-------

After preparing all the setting, you can simply call:
```shell
nohup python -u train.py >log &
```

And, all the history data will be saved into ./tensorboard-history, you can use tensorboard to watch the training loss and some intermediate results.
```
tensorboard --logdir ./tensorboard-history
```

### Deblur performance

