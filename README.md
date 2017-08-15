# TensorFCN

Tensorflow implementation of [Fully Convolutional Networks for Semantic Segmentation](http://arxiv.org/pdf/1605.06211v1.pdf) (FCNs) based on the code written by [shekkizh](https://github.com/shekkizh/FCN.tensorflow) and modified to be used with ease for any given task.

The model can be applied on the [Scene Parsing Challenge dataset](http://sceneparsing.csail.mit.edu/) provided by MIT straightaway after cloning this repo. I do not have any trained model available due to lack of computing resources.

1. [Prerequisites](#prerequisites)
2. [How to use](#howto)
3. [Datasets](#datasets)
4. [Differences](#differences)


## Prerequisites
- [numpy](https://github.com/numpy/numpy)
- [scipy](https://github.com/scipy/scipy)
- [opencv](https://github.com/opencv/opencv) (both 2.4.x and 3.x should work)
- Tested only with `tensorflow 1.1.0` and `python 2.7.12` on `Ubuntu 16.04`. I tried to make this `python 3` compatible but I haven't checked if it works.


## How to use
[example.py](https://github.com/dubvulture/tensor_fcn/blob/master/examples.py) should be self-explanatory for basic usage.

Things you can set while setting up the network :
- *Validation Set* (if you want to keep track of its loss)
- *Learning Rate*
- *Keep Probability* (1 - Dropout) for some layers
- *Training loss* summary frequency
- *Validation loss* summary frequency
- *Model saving* frequency
- *Maximum number of steps*

Things you can tweak in the code:
- [Optimizer](https://github.com/dubvulture/tensor_fcn/blob/master/fcn.py#L49) - default is **Adam**.
- [Loss function](https://github.com/dubvulture/tensor_fcn/blob/master/fcn.py#L58)
- [Number of models](https://github.com/dubvulture/tensor_fcn/blob/master/fcn.py#L72) to keep saved during training.
- [Choosing](https://github.com/dubvulture/tensor_fcn/blob/master/networks/fcn.py#L11) between VGG19 or VGG16 (and maybe others if implemented) as **encoders**.
