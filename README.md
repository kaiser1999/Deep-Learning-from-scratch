# Deep Learning from scratch

We here provide the codes for the construction of a deep learning model from scratch in Python. Only **_numpy_** and **_scipy_** (cross-correlation in CNN for speed) libraries are used in the calculations for all layers in this project. The layers include:
## **_Intermediate_Layers.py_**
1. Activation (**_Activation_**) Layer
2. Batch Normalization (**_BatchNormalization_**) Layer
3. Instance Normalization (**_InstanceNormalization_**) Layer
4. Dropout (**_Dropout_**) Layer

## **_Dense.py_**
5. Fully-connected (**_Dense_**) layer

## **_CNN.py_**
6. Convolutional (**_Convolutional_**) Layer with fixed dilation rate (1, 1) 
7. Pooling (**_Pooling_**) Layer
8. Transposed Convolutional (**_Transposed_Conv_**) Layer with fixed dilation rate (1, 1)
9. Flattening (**_Flattening_**) Layer
10. Global Average Pooling (**_GlobalAveragePooling_**) Layer

## **_RNN.py_**
11. Simple Recurrent Neural Network (**_SimpleRNN_**) Layer
12. Gated Recurrent Unit (**_GRU_**) Layer
13. Long-Short Term Memory (**_LSTM_**) Layer

For the derivations and explanations of these codes, please refer to the book [1].


# Examples
There are two simple examples:
1. MNIST [2]: Require tensorflow
2. Bitcoin price prediction: downloaded from https://www.CryptoDataDownload.com

# Reference
[1] 1. Kaiser Fan, Phillip Yam (Expected 2024) . Statistical Deep Learning with Python and R.
[2] 2. Deng, L. (2012). The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6), 141–142.