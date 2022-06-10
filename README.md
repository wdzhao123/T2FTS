# T2FTS in PyTorch

Implementation of "Teaching Teachers First and Then Student: Hierarchical Distillation to Improve Long-Tailed Object Recognition in Aerial Images" in PyTorch
## Datasets
- `train_data`: The data for training.
  - `FGSC-23`: Contains 3256 training images and it's GT.
  - `DOTA`: Contains 98906 training images and it's GT.
- `test_data`: The data for testing.
  - `FGSC-23`: Contains 825 testing images and it's GT.
  - `DOTA`: Contains 28853 testing images and it's GT.

baidu link: https://pan.baidu.com/s/1gIZZEakVLPoMoJzMdF5sdw  passward: jab5

> It should be noted that because we divide the dataset into three subsets, and the dataset and the ground truth in the link are not divided, users need to divide and make the ground truth by themselves.

## Test
You need to modify the path in the programs first, where baseline_cl_s_weight.py is for the FGSC-23 dataset and dota_baseline_cl_s_weight.py is for the DOTA dataset. In resnet50.py, ‘num_classes’ is determined by the number of categories in the dataset

Our test model has parameters of 23.5M and FLOPs of 1.56G and takes about 6.8 ms to generate the result for an image with a size of 224 × 224.

- `dataset.py`: process the dataset before passing to the network.
- `resnet50.py`: defines the model.
- `baseline_cl_s_weight.py, dota_baseline_cl_s_weight.py`: the entry point for testing.

You can use the following model to output results directly.
Here is our parameters： 

baidu link: https://pan.baidu.com/s/1hq-bJyNU_fIFrTWg1A1VoA   passward: qsp3

The "FGSC-23_save_model.zip" used for FGSC-23 and the "DOTA_save_model.zip" used for DOTA.

## Train
Our method is a two-stage training, first training the teacher model, then training the student model. Of course, the first thing to do is to modify the path in the relevant program.
### Teacher
Step-by-step training baseline_cl_t1.py, baseline_cl_t2.py and baseline_cl_t3.py to get three teacher models. When training the second and third teacher models, load the optimal parameters of the previously trained teacher model into the files.

Here is the pretrained weight of the teacher models. “t1.py, t2.py, t3.py” is used for FGSC-23, “dota_t1.py, dota_t2.py, dota_t3.py”is used for DOTA.

baidu link: https://pan.baidu.com/s/1b6uYSEYAmmGschm4MW7X3A passward: 4ifv


### Student
First select the optimal training parameters for t1, t2 and t3, then load them into three teacher model in the baseline_cl_s_weight.py. Training this code to get the student model.
If you want to train student model wtih DOTA dataset or other dataset, you can modify the relevant parameters and code in the network to run directly. Take DOTA as an example, you can use the dota_baseline_cl_t1.py, dota_baseline_cl_t2.py and dota_baseline_cl_t3.py step by step to train three teacher models. Then, loading the trained three DOTA teacher model parameter into dota_baseline_cl_s_weight.py 
You can choose whether to initialize the network model with the optimal parameters of the baseline network. If you want, you need to train the baseline network by baseline_resnet50.py.
- `baseline_cl_t1/2/3.py, dota_baseline_cl_t1/2/3.py`: training the three teacher models.
- `dataset.py`: process the dataset before passing to the network.
- `ClassAwareSampler.py`: initial sampler.
- `resnet50.py`: defines the model.
- `baseline_cl_s_weight.py, dota_baseline_cl_s_weight.py`: the entry point for training.


If our paper can bring you some help, please cite it:
> @article{zhao2022teaching, <br> title={Teaching Teachers First and Then Student: Hierarchical Distillation to Improve Long-Tailed Object Recognition in Aerial Images}, <br> author={Zhao, Wenda and Liu, Jiani and Liu, Yu and Zhao, Fan and He, You and Lu, Huchuan}, <br> journal={IEEE Transactions on Geoscience and Remote Sensing}, <br> year={2022}, <br> publisher={IEEE}}
