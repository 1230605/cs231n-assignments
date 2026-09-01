from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss

        # 梯度（与 loss 同步计算）：dLi/dW = x_i^T (p - onehot(y_i))
        # 概率向量在正确类别处减 1，就等价于 p - onehot
        p[y[i]] -= 1
        # 外积 x_i (D,) ⊗ (p - onehot) (C,) -> (D, C)，累加进 dW
        dW += np.outer(X[i], p)

    # normalized softmax loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)

    # 梯度平均化 + L2 正则项的梯度 2λW
    dW = dW / num_train + 2 * reg * W

    return loss, dW
def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    num_train = X.shape[0]
    num_classes = W.shape[1]

    # 一次算出所有样本对所有类别的打分：X (N, D) @ W (D, C) -> (N, C)
    scores = X.dot(W)
    # 数值稳定：每行减去该行最大值；keepdims 保留 (N, 1) 以便广播
    scores -= np.max(scores, axis=1, keepdims=True)
    # softmax：逐行 exp，再除以该行的概率和
    p = np.exp(scores)
    p /= np.sum(p, axis=1, keepdims=True)

    # 交叉熵损失：取每行正确类别的概率，负对数，求平均 + L2 正则
    correct_logp = -np.log(p[np.arange(num_train), y])
    loss = correct_logp.sum() / num_train + reg * np.sum(W * W)

    # 梯度：dW = X^T (P - Y_onehot) / N + 2λW
    onehot = np.eye(num_classes)[y]          # (N, C)，正确类别处为 1
    dW = X.T.dot(p - onehot) / num_train + 2 * reg * W

    return loss, dW
