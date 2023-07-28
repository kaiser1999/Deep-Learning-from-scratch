# Deep Learning from scratch

We here provide the codes for the construction of a deep learning model from scratch in Python. Only \texttt{numpy} and \texttt{scipy} (cross-correlation in CNN for speed) libraries are used in the calculations for all layers in this project. The layers include:
## \texttt{Intermediate_Layers.py}
1. Activation (\texttt{Activation}) Layer
2. Batch Normalization (\texttt{BatchNormalization}) Layer
3. Instance Normalization (\texttt{InstanceNormalization}) Layer
4. Dropout (\texttt{Dropout}) Layer

## \texttt{Dense.py}
5. Fully-connected (\texttt{Dense}) layer

## \texttt{CNN.py}
6. Convolutional (\texttt{Convolutional}) Layer with fixed dilation rate (1, 1) 
7. Pooling (\texttt{Pooling}) Layer
8. Transposed Convolutional (\texttt{Transposed_Conv}) Layer with fixed dilation rate (1, 1)
9. Flattening (\texttt{Flattening}) Layer
10. Global Average Pooling (\texttt{GlobalAveragePooling}) Layer

## \texttt{RNN.py}
11. Simple Recurrent Neural Network (\texttt{SimpleRNN}) Layer
12. Gated Recurrent Unit (\texttt{GRU}) Layer
13. Long-Short Term Memory (\texttt{LSTM}) Layer

For the derivations and explanations of these codes, please refer to the book:
Kaiser Fan, Phillip Yam (Expected 2024) . Statistical Deep Learning with Python and R.

# Examples
There are two simple examples:
1. MNIST: Require tensorflow
2. Bitcoin price prediction: downloaded from https://www.CryptoDataDownload.com

# Reference
Deng, L. (2012). The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6), 141–142.