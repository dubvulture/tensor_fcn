# TensorFCN

Tensorflow implementation of [Fully Convolutional Networks for Semantic Segmentation](http://arxiv.org/pdf/1605.06211v1.pdf) (FCN-8 in particular) based on the code written by [shekkizh](https://github.com/shekkizh/FCN.tensorflow) and modified to be used with ease for any given task.

The model can be applied on the [Scene Parsing Challenge dataset](http://sceneparsing.csail.mit.edu/) provided by MIT straightaway after cloning this repo. I do not have any trained model available due to lack of computing resources.

## Table of contents
1. [Prerequisites](#prerequisites)
2. [Differences](#differences)
3. [Usage](#usage)
4. [Datasets](#datasets)


## Prerequisites
- [numpy](https://github.com/numpy/numpy)
- [scipy](https://github.com/scipy/scipy)
- [opencv](https://github.com/opencv/opencv) (both 2.4.x and 3.x should work)
- Tested only with `tensorflow 1.1.0` and `python 2.7.12` on `Ubuntu 16.04`. I tried to make this `python 3` compatible but I haven't checked yet if it works.


## Differences
As pointed out frequently in the issue tracker of [FCN.tensorflow](https://github.com/shekkizh/FCN.tensorflow/issues) there were some discrepancies between the caffe and the tensorflow implementation.
Here are the main ones and how I handled them:

- ### [Conv6 padding](https://github.com/shekkizh/FCN.tensorflow/issues/52)
In the original implementation there was no padding in order to shrink the tensor down to ` [batch_size, 1, 1, 4096]`.
This works well when the input images are 224x224 but for any other resolution it will break the deconvolution phase.
Since the padding did no harm and the results were still acceptable I decided to leave it as it is.

- ### [Average Pooling or Max Pooling](https://github.com/shekkizh/FCN.tensorflow/issues/26)
I just sticked to the original implementation using max pooling.

- ### [Final layer of VGG](https://github.com/shekkizh/FCN.tensorflow/issues/46)
I just sticked to the original implementation using the relu'd layer.

- ### VGG19 or VGG16
The original implementation used **VGG16**, [shekkizh](https://github.com/shekkizh) used **VGG19**. I leave you with the freedom of choice.


## Usage
[example.py](https://github.com/dubvulture/tensor_fcn/blob/master/examples.py) should be self-explanatory for basic usage. Note that a trained model can be run also on arbitrary sized images that will be accordingly padded to avoid information loss during pooling. 

Things you can set while setting up the network and training phase:
- *Number of classes*
- *Validation Set* (if you want to keep track of its loss)
- *Learning Rate*
- *Keep Probability* (1 - Dropout) for some layers
- *Training loss* summary frequency
- *Validation loss* summary frequency
- *Model saving* frequency
- *Maximum number of steps*
- [Choosing](./fcn.py#L23) between VGG19 or VGG16 (and maybe others if implemented) as **encoders**.

Things you can tweak in the code:
- [Optimizer](./fcn.py#L49) - default is **Adam**.
- [Loss function](./fcn.py#L58)
- [Number of models](./fcn.py#L72) to keep saved during training.

## Datasets

In [dataset_reader](./dataset_reader/) there are two classes, [BatchDataset](./dataset_reader/dataset_reader.py) which is supposed to be an abstract class, and [ADE_Dataset](./dataset_reader/ade_dataset.py) which is an example on how to specialize **BatchDataset** and is ready to be used for training.
Basic usage of a subclass:
```python
dt = MyDataset(*args, **kwargs)
images, annotations, weights, names = dt.next_batch()
```
where `images`, `annotations` and `weights` are numpy arrays of shape `[batch_size, height, width, channels]` (3 channels for images and 1 for both annotations and weights). In **ADE_Dataset** `weights` are not used but for other tasks with different datasets they might be useful.

When subclassing **BatchDataset** there are things to keep in mind:
- the argument `names` is required to identify its elements, but it might not be mandatory to be passed by the user to the subclass unless you want to be able to specify  a subset of the dataset (as I did with **ADE_Dataset** when creating the validation set)
- the argument `image_op` is often required to perform cropping and resizing when handling batch sizes greater than 1 unless you have a homogeneous dataset
- `image_size` has to be specified only if `batch_size` is greater than 1
