# A Pytorch Toy Template

[TOC]

## Motivation
Deep Learning (DL) also involves learning of the framework where datasets are prepared, models are built, training are implemented, etc. 
I myself had a hard time when it came to the framework choice, and then even a harder time when I needed to learn how to handle with it.  
A good template is supposed to touch on all the basic components for the DL, and if possible, have some features that are beneficial such that even without the template, the users integrate that feature(s) into their own project, or even their own templates.  
There are already great templates in the community to learn from, see repos of [victoresque](https://github.com/victoresque/pytorch-template), or [ahangchen](https://github.com/ahangchen/torch_base), to name a few.  
The first intention of building this template is for self-learning as an entry-level learner in DL, and then evolves for a wider use for my labmates. Now I think it is even better to be pushed to the Github as a standalone repo.  
Hope this can help users have a glimpse to what is under the hood such that they are better prepared for more specific and sophisticated projects.

## Features

### General

* [x] Provides general base classes like `TrainerBase()` and `EvalatorBase()` for inheritance for other more specific tasks
* [x] Supports basic train, validation, and test workflow
* [x] Supports automatic selection of GPU (if available), otherwise CPU
* [x] Supports dataset sampling for fast implementation
* [x] Supports different dataset, model, optimizer, scheduler, and loss based on the configuration
* [x] Supports data parallelism
* [x] Supports random seed for reproduction
* [x] Supports up-to-date logger info to the terminal and log file
* [x] Supports decent checkpoint saving when an exception is thrown
* [x] Supports timestamp naming for each training
* [x] Supports Tensorboard for visualization
* [x] Supports saving configurations in a `.json` file
* [x] Supports gradient clipper
* [x] Supports early stopper
* [x] Supports AvgMeter
* [x] Supports different metrics
* [x] Provides a set of helper functions
  * Configuration Comparator
  * Model parameter counter

### Specific

* [x] Implement a vanilla classification task
* [x] Implement a segmentation task on Carvana  

Note that this template is highly bound to the configuration part, which can be retrieved at the very end of `trainer.py` and `eval.py` as basic config options, and in their more specific counterparts like `trainer_vanilla.py` and `trainer_seg_on_carvana.py`, whose config options override those of the base class.

## Structure of The Repo
```txt
.
├── base # base dir where base classes reside
│   ├── eval.py # base class for the evaluator with attributes and methods about to be overridden
│   └── trainer.py # base class for the trainer with attributes and methods about to be overridden
├── data
│   ├── Carvana.py # dataloader for Carvana
│   └── data_entry.py # dataset selector based on the `data_type`
├── dataset # where the FashionMNIST is supoosed to be downloaded
├── dev.ipynb # a Jupyter notebook for dev (Empty)
├── environment.yml # for the conda env dependencies
├── eval_seg_on_carvana.py # evaluator for the test on segmentation on carvana
├── eval_vanilla.py # evaluator for the vanilla classification
├── img # for visualization demo
├── loss
│   └── dice_loss.py # dice loss for segmentation on Carvana
├── model
│   ├── model_entry.py # model selector
│   ├── UNetpp.py # UNet++ the model
│   ├── UNet.py # UNet the model
│   └── vanilla_model.py # the vanilla model
├── README.md # this file
├── Results 
│   ├── Carvana # the result dir of segmentation on Carvana training
│   │   └── 1664021850 # an example for demo
│   │       ├── config_SegOnCarvana_1664021850.json
│   │       ├── SegOnCarvana_1664021850_log.txt
│   │       └── summary
│   │           └── events.out.tfevents.1664021867.ustc115.31421.0
│   └── Vanilla # the result dir of vanilla training
│       └── 1663932921
│           ├── config_Vanilla_1663932921.json
│           ├── summary
│           │   └── events.out.tfevents.1663932930.ustc115.13751.0
│           └── Vanilla_1663932921_log.txt
├── test_viz # where the visualization is saved, a triplet is transferred into the `./img/`
├── trainer_seg_on_carvana.py # trainer for segmentation on Carvana, this overrides the base trainer class partly
├── trainer_vanilla.py # trainer for vanilla, this also overrides the base class partly
└── utils
    ├── clipper.py # gradient clipper
    ├── configcmp.py # config comparator for two config json files
    ├── earlystopper.py # early stopper if there is no improvement under a threshold
    ├── func.py # some functions
    ├── logger.py # logger to track the training
    ├── meter.py # meter to track the metrics
    ├── metrics.py # calculator of metrics
    ├── optimizer.py # optimizer selector
    ├── scheduler.py # scheduler selector
    ├── seed.py # seed generator for reproductivity
    └── timestamp.py # timestamp generator for every training 
```

## Usage
### A vanilla classification demo
#### Introduction
A vanilla classification demo is provided to show how to use this template.  
For simplicity:
  * Dataset: [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist), which is automatically downloaded once training starts to the `opt.data_dir` if there is not one.
  * Model: The model is just as plain as (and actually the performance is not very well)
    ```txt
    Vanilla(
    (flatten): Flatten(start_dim=1, end_dim=-1)
    (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
      )
    )
    ```
    which can be found in `./model/vanilla_model.py`
#### Before training
This implementation runs well in a conda env, and make sure you get all the independencies right.  
A file named `environment.yml` is provided in the repo, for use refer to
* [Creating an environment from an environment.yml file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#id2)
* [Updating an environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#id4)
#### Training
Use the following command to get the training run.
```sh
  python trainer_vanilla.py \
--proj_name Vanilla \
--model_type Vanilla \
--data_type FashionMNIST \
--output_dir ./Results/Vanilla/ \
--gpu_devices 1 \
--data_dir ./dataset
```
Once training  starts or finishes with no error, results should be saved in the `output_dir`, where a timestamp is used for naming each training. The timestamp also appear on the logging output of the terminal.
#### Testing
Once the training finishes, get the path to the checkpoint file and run the following command to test
```txt
python eval_vanilla.py --model_path Path/To/Your/Resutl/TIMESTAMP/Vanilla_best_ckpt.pth
```
Once it finishes without any error, a set of prediction-lable pairs are printed to the terminal to show how the trained model performs.  
Note that since this is a simple classification demo, the test is kept simple as well. The more sophisticated `eval.py` is not involved yet.

### A more advanced segmentation demo
#### Introduction
Once the vanilla demo runs smoothly, it suggests that all the requirements are met and we can move onto more complicated tasks, say, segmentation.
* Dataset: [Carvana](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data), which should download by the user to the `--data_dir` 
* Model: [UNet](https://paperswithcode.com/paper/u-net-convolutional-networks-for-biomedical), a classic segmentation model
#### Before training
The sub directories of the dataset directory are named `train` and `train_masks` as
```txt
./Dataset/Carvana/
├── train
└── train_masks
```
Note that for demo purpose only, achieving higher performance against the state-of-the-art methods is not necessary here.
And here training, validation, and testing are only performed on partial (the rate is based on the option `--train_val_sample_rate` and `--test_sample_rate`) dataset of the overall **train** set provided by the official source. Since the default rate is small enough, and dataset is shuffled before fed to the model, this should not be an issue. 
⚠️But in real tasks, it is NOT good practice to train, validation, and test on the same dataset. Except for the case when you partition the dataset into three non-overlapping sets. It is a MUST to let the three be blind to each other.

#### Training
Use the following command to start training. 
```bash
python trainer_seg_on_carvana.py \
--proj_name SegOnCarvana \
--model_type UNet \
--data_type Carvana \
--output_dir Dataset/Carvana/ \
--gpu_devices 0 \
--data_dir ./Dataset/Carvana/
```
#### Testing
Once the training finishes, get the path to the checkpoint file and other required information, and run the following command to test
```bash
python eval_seg_on_carvana.py \
--model_path ./Results/Carvana/TIMESTAMP/UNet_best_ckpt.pth \
--save_path ./test_viz/ \
--data_dir ./Dataset/Carvana \
--model_type UNet \
--data_type Carvana \
--viz True
```
The viz flag is for visualization and the viz frequency is set to 50 by default, that is, visualizing the test results every 50 samples.  
The `--data_dir` and `--model_type` are actually better to be read from the `json` file previously saved in the training phase as in [victoresque](https://github.com/victoresque/pytorch-template)'s work.

#### Viz

Once the `--viz` is set to be `True`, the visualization can be obtained.
The following is a simple illustration.

|Original|Mask (GT/Target)|Prediction|
| :----: | :----: | :----: |
|![simple viz](./img/5d95d80e10a9_12_origin.png)|![simple viz](./img/5d95d80e10a9_12_mask.png)|![simple viz](./img/5d95d80e10a9_12_pred.png)|
