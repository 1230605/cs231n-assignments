from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.reg = reg
        self.dtype = dtype
        self.params = {}
        C, H, W = input_dim
        # conv layer: num_filters filters of size filter_size x filter_size
        self.params["W1"] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
        self.params["b1"] = np.zeros(num_filters)
        # 2x2 max pool halves the spatial size; flatten pool output
        pool_dim = num_filters * (H // 2) * (W // 2)
        self.params["W2"] = weight_scale * np.random.randn(pool_dim, hidden_dim)
        self.params["b2"] = np.zeros(hidden_dim)
        self.params["W3"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["b3"] = np.zeros(num_classes)
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        X = X.astype(self.dtype)
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        # ---------- forward ----------
        pool, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        N = pool.shape[0]
        flat = pool.reshape(N, -1)
        h, cache2 = affine_relu_forward(flat, W2, b2)
        scores, cache3 = affine_forward(h, W3, b3)

        if y is None:
            return scores

        # ---------- loss and backward ----------
        loss, grads = 0.0, {}
        loss, dscores = softmax_loss(scores, y)
        dh, grads["W3"], grads["b3"] = affine_backward(dscores, cache3)
        dflat, grads["W2"], grads["b2"] = affine_relu_backward(dh, cache2)
        dpool = dflat.reshape(pool.shape)
        dX, grads["W1"], grads["b1"] = conv_relu_pool_backward(dpool, cache1)

        for name in ("W1", "W2", "W3"):
            loss += 0.5 * self.reg * np.sum(self.params[name] ** 2)
            grads[name] += self.reg * self.params[name]
        return loss, grads
        return loss, grads
