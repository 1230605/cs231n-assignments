from builtins import range
import numpy as np

# import numexpr as ne # ~~DELETE LINE~~


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################

    ###########################################################################
    # 把每个样本展平成 D 维行向量：(N, d1, ..., dk) -> (N, D)
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)
    # 仿射变换：out = x @ W + b，b 通过广播加到每一行
    out = x_reshaped.dot(w) + b

#                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################

    ###########################################################################
    # 恢复 x 的行数，并把 x 展平成 (N, D)
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)
    # dx = dout @ W^T，再还原成 x 的原始形状
    dx = dout.dot(w.T).reshape(x.shape)
    # dW = x^T @ dout
    dw = x_reshaped.T.dot(dout)
    # db = 沿 batch 维求和（b 广播到每一行的逆操作）
    db = dout.sum(axis=0)

#                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################

    ###########################################################################
    # ReLU：max(0, x)，逐元素把负数置 0
    out = np.maximum(0, x)

#                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################

    ###########################################################################
    # ReLU 的导数：x > 0 处为 1、其余为 0，用 mask 过滤上游梯度
    dx = dout * (x > 0)

#                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization (training + test)."""
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)
    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))
    out, cache = None, None
    if mode == "train":
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = gamma * x_hat + beta
        cache = (x_hat, gamma, beta, sample_mean, sample_var, eps)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
    elif mode == "test":
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_hat + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var
    return out, cache
def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization (unrolled computation graph)."""
    x_hat, gamma, beta, sample_mean, sample_var, eps = cache
    N = dout.shape[0]
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    sigma = np.sqrt(sample_var + eps)
    g = dout * gamma
    dvar = -0.5 * np.sum(g * x_hat, axis=0) / (sample_var + eps)
    dmu = -np.sum(g, axis=0) / sigma
    dx = g / sigma + dvar * 2 * (x_hat * sigma) / N + dmu / N
    return dx, dgamma, dbeta
def batchnorm_backward_alt(dout, cache):
    """Backward pass for batch normalization (collapsed closed form)."""
    x_hat, gamma, beta, sample_mean, sample_var, eps = cache
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    sigma = np.sqrt(sample_var + eps)
    g = dout * gamma
    dx = (g - np.mean(g, axis=0) - x_hat * np.mean(g * x_hat, axis=0)) / sigma
    return dx, dgamma, dbeta
def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization (per-sample over features)."""
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    sample_mean = np.mean(x, axis=1, keepdims=True)
    sample_var = np.var(x, axis=1, keepdims=True)
    x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
    out = gamma * x_hat + beta
    cache = (x_hat, gamma, beta, sample_mean, sample_var, eps)
    return out, cache
def layernorm_backward(dout, cache):
    """Backward pass for layer normalization (batchnorm math over axis=1)."""
    x_hat, gamma, beta, sample_mean, sample_var, eps = cache
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)
    sigma = np.sqrt(sample_var + eps)
    g = dout * gamma
    dx = (g - np.mean(g, axis=1, keepdims=True)
          - x_hat * np.mean(g * x_hat, axis=1, keepdims=True)) / sigma
    return dx, dgamma, dbeta
def dropout_forward(x, dropout_param):
    """Inverted dropout forward: train keeps each unit with prob p and scales by 1/p; test is identity."""
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])
    mask = None
    out = None
    if mode == "train":
        # Bernoulli mask scaled by 1/p so that E[train out] == test out
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == "test":
        out = x
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache
def dropout_backward(dout, cache):
    """Inverted dropout backward: gradient flows only through the (already 1/p scaled) mask."""
    dropout_param, mask = cache
    mode = dropout_param["mode"]
    dx = None
    if mode == "train":
        dx = dout * mask
    elif mode == "test":
        dx = dout
    return dx
def conv_forward_naive(x, w, b, conv_param):
    """Naive conv forward: loop over output pixels, vectorize over batch and filters."""
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    # zero-pad input symmetrically along height and width
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    Hp = 1 + (H + 2 * pad - HH) // stride
    Wp = 1 + (W + 2 * pad - WW) // stride
    out = np.zeros((N, F, Hp, Wp))
    for i in range(Hp):
        for j in range(Wp):
            x_patch = x_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]  # (N,C,HH,WW)
            # sum over (C, HH, WW) -> (N, F); add bias per filter
            out[:, :, i, j] = np.tensordot(x_patch, w, axes=([1, 2, 3], [1, 2, 3])) + b
    cache = (x, w, b, conv_param)
    return out, cache
def conv_backward_naive(dout, cache):
    """Naive conv backward: unroll the same windows as the forward pass."""
    x, w, b, conv_param = cache
    stride = conv_param["stride"]
    pad = conv_param["pad"]
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, Hp, Wp = dout.shape
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    db = np.sum(dout, axis=(0, 2, 3))
    dw = np.zeros_like(w)
    dx_pad = np.zeros_like(x_pad)
    for i in range(Hp):
        for j in range(Wp):
            x_patch = x_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]  # (N,C,HH,WW)
            dout_ij = dout[:, :, i, j]                                                     # (N,F)
            # dW += sum_n dout[n,f] * x_patch[n]  -> (F,C,HH,WW)
            dw += np.tensordot(dout_ij, x_patch, axes=([0], [0]))
            # dX_pad += sum_f dout[n,f] * w[f]  -> (N,C,HH,WW)
            dx_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += np.tensordot(dout_ij, w, axes=([1], [0]))
    # crop the padding back out of dx
    dx = dx_pad[:, :, pad:pad + H, pad:pad + W]
    return dx, dw, db
def max_pool_forward_naive(x, pool_param):
    """Naive max-pool forward: take the max over each non-overlapping-ish window."""
    ph = pool_param["pool_height"]
    pw = pool_param["pool_width"]
    stride = pool_param["stride"]
    N, C, H, W = x.shape
    Hp = 1 + (H - ph) // stride
    Wp = 1 + (W - pw) // stride
    out = np.zeros((N, C, Hp, Wp))
    for i in range(Hp):
        for j in range(Wp):
            window = x[:, :, i * stride:i * stride + ph, j * stride:j * stride + pw]
            out[:, :, i, j] = window.max(axis=(2, 3))
    cache = (x, pool_param)
    return out, cache
def max_pool_backward_naive(dout, cache):
    """Naive max-pool backward: route gradient only to the argmax location of each window."""
    x, pool_param = cache
    ph = pool_param["pool_height"]
    pw = pool_param["pool_width"]
    stride = pool_param["stride"]
    N, C, H, W = x.shape
    Hp = 1 + (H - ph) // stride
    Wp = 1 + (W - pw) // stride
    dx = np.zeros_like(x)
    for i in range(Hp):
        for j in range(Wp):
            window = x[:, :, i * stride:i * stride + ph, j * stride:j * stride + pw]
            mask = window == window.max(axis=(2, 3), keepdims=True)
            dx[:, :, i * stride:i * stride + ph, j * stride:j * stride + pw] += dout[:, :, i, j, None, None] * mask
    return dx
def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Spatial batchnorm: reuse vanilla batchnorm over the (N*H*W, C) view."""
    N, C, H, W = x.shape
    # move channels to the last axis and flatten to (N*H*W, C)
    x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)
    out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return out, cache
def spatial_batchnorm_backward(dout, cache):
    """Spatial batchnorm backward: unflatten the vanilla batchnorm gradient."""
    N, C, H, W = dout.shape
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
    dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta
def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    ###########################################################################
    num_train = x.shape[0]
    # 数值稳定：每行减去该行最大值，防止 exp 溢出
    shifted = x - np.max(x, axis=1, keepdims=True)
    # log-softmax：log p_j = s_j - log(sum(exp(s)))，数值更稳
    log_probs = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    probs = np.exp(log_probs)

    # 交叉熵损失：取每个样本正确类别的负对数概率，求平均
    loss = -np.sum(log_probs[np.arange(num_train), y]) / num_train

    # 梯度：dx = (p - onehot(y)) / N
    dx = probs.copy()
    dx[np.arange(num_train), y] -= 1
    dx /= num_train
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
