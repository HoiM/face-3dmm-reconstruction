# 3D face reconstruction

### Introduction

Implementation of the 3DMM deep face reconstruction method with train and test code:

[Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set](https://arxiv.org/abs/1903.08527)

This is an experiment that was done many years ago. Now I release the code if anyone needs it.

### Usage

##### Prerequisites

Libraries that are needed for running the code, with preferred versions:

```commandline
torch==2.2.1
torchvision==0.17.1
pytorch3d==0.7.6
tensorboard==2.14.0
opencv-python==3.4.11.43
dlib==19.17.0
```

Download the parameters that are needed for running the code:

* Please go to [this repo](https://github.com/ascust/3DMM-Fitting-Pytorch), download BFM and Expression Basis, and run the conversion program to get `BFM09_model_info.mat`. Then place the `BFM09_model_info.mat` file in `params/bfm` and run `parse.py` inside the `params` directory.
* If you want to train the model by yourself, please go to [this repo](https://github.com/TreB1eN/InsightFace_Pytorch) and download `model_ir_se50.pth`. Place the downloaded file inside the `params` directory. This is only needed for training.

##### Inference

* An example for inference is shown by running `python reconstruct_and_render.py`, which reconstructs the faces in `assets/images` and the rendered results are saved in `assets/results`.

##### Training

* Training dataset: if you want to construct your own dataset, please follow the following steps
  * Collect images with human faces and place them in `data/images`. It is preferred that each image contains only one face.
  * Inside the `data` directory, run `python main.py`. The program will crop the faces, get 68 facial landmarks and the corresponding face region masks. Processed training data will be saved in `data/data`. I use `dlib` to get facial landmarks. You can replace it with a better one. Face regions are detected by using [nasir6/face-segmentation](https://github.com/nasir6/face-segmentation) (Thanks to this work!)
* Train the model: run `python train.py`. Hyperparameters are written inside this file. 
  * TODO: Distributed Data Parallel is not used yet. 
  * TODO: 3DMM parameters can also be trained in order to get a better model. I haven't experiment with this. 

### References
Thanks to the following works:
* [ascust/3DMM-Fitting-Pytorch](https://github.com/ascust/3DMM-Fitting-Pytorch)
* [nasir6/face-segmentation](https://github.com/nasir6/face-segmentation)
* [TreB1eN/InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch)
* [Microsoft/Deep3DFaceReconstruction](https://github.com/Microsoft/Deep3DFaceReconstruction)
* [changhongjian/Deep3DFaceReconstruction-pytorch](https://github.com/changhongjian/Deep3DFaceReconstruction-pytorch)
* [Dlib installation on Linux](https://stackoverflow.com/questions/41912372/dlib-installation-on-windows-10)
