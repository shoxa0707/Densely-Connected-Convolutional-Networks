# DenseNet(Densely Connected Convolutional Networks)

We use and explain DenseNet-121 architecture in this repository

## <span style="color:red">DenseNet</span>

In a DenseNet architecture, each layer is connected to every other layer, hence the name Densely <br>
Connected Convolutional Network. For L layers, there are L(L+1)/2 direct connections. For each layer,<br>
the feature maps of all the preceding layers are used as inputs, and its own feature maps are used as <br>
input for each subsequent layers.

This is really it, as simple as this may sound, DenseNets essentially conect every layer to every other<br>
layer. This is the main idea that is extremely powerful. The input of a layer inside DenseNet is the <br>
concatenation of feature maps from previous layers.

<a><img src="images/densenet.png"/></a>

## <span style="color:green">Dense Blocks</span>

Now that we understand that a DenseNet architecture is divided into multiple dense blocks, let’s look<br>
at a single dense block in a little more detail. Essentially, we know, that inside a dense block, each<br>
layer is connected to every other layer and the feature map size remains the same.

<a><img src="images/denseblock.png"/></a>

Let’s try and understand what’s really going on inside a dense block. We have some gray input features <br>
that are then passed to LAYER_0. The LAYER_0 performs a non-linear transformation to add purple features<br>
to the gray features. These are then used as input to LAYER_1 which performs a non-linear transformation<br>
to also add orange features to the gray and purple ones. And so on until the final output for this 3 layer<br>
denseblock is a concatenation of gray, purple, orange and green features.

So, in a dense block, each layer adds some features on top of the existing feature maps.

Therefore, as you can see the size of the feature map grows after a pass through each dense layer and the<br>
new features are concatenated to the existing features. One can think of the features as a global state of<br>
the network and each layer adds K features on top to the global state.

This parameter K is referred to as growth rate of the network.

## <span style="color:blue">Types of DenseNet</span>

We already know by now from following figure, that DenseNets are divided into multiple DenseBlocks.<br>
The various architectures of DenseNets have been summarized in the paper:

<a><img src="images/densenets.png"/></a>

# Datasets

- In this project we use [ImageNet2012](https://www.image-net.org/download.php).

- And we use [oxford_iiit_pet](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet) dataset from tensorflow:<br>
  The Oxford-IIIT pet dataset is a 37 category pet image dataset with roughly 200 images for each class. <br>
  The images have large variations in scale, pose and lighting. All images have an associated ground truth annotation of breed.

# Installation

1. Install python

- requirement version 3.x <span style="color:yellow">**x = {7,8,9, 10}**</span>

2. Create virtual environment

- pip install virtualenv
- python -m venv {name of virtual environment}
- activate it<br>
- 1. On Windows: C:/path to your env/Scripts/activate
- 2. On Linux: path to your env/bin/activate

3. `pip install -r requirements.txt`

# Features

### Deep Learning

- architectures are built using [tensorflow](https://github.com/tensorflow/tensorflow.git)
- run in colab [![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QoarjGGkUXZMQ9eoztam4LY1LLRoXvYP?usp=sharing)
