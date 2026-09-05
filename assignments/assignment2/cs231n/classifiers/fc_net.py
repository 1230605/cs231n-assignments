from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################

        ############################################################################
        self.params["W1"] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["W2"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["b2"] = np.zeros(num_classes)

#                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        ############################################################################
        N = X.shape[0]
        X = X.reshape(N, -1)   # 鎶?(N, 32, 32, 3) 灞曞钩鎴?(N, D)
        # 绗竴灞傦細affine + relu
        h1, cache1 = affine_relu_forward(X, self.params["W1"], self.params["b1"])
        # 绗簩灞傦細affine锛屽緱鍒?C 涓被鍒殑鍒嗘暟
        scores, cache2 = affine_forward(h1, self.params["W2"], self.params["b2"])

#                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        ############################################################################
        # 鏁版嵁鎹熷け + 瀵瑰垎鏁扮殑姊害锛坰oftmax锛?        loss, grads = 0.0, {}
        loss, dscores = softmax_loss(scores, y)
        # 绗簩灞傚弽鍚戯細寰?dh1锛堢户缁洖浼狅級涓?W2/b2 鐨勬搴?        dh1, grads["W2"], grads["b2"] = affine_backward(dscores, cache2)
        # 绗竴灞傚弽鍚戯紙affine+relu锛夛細寰?W1/b1 鐨勬搴︼紙dx 涓嶉渶瑕侊級
        _, grads["W1"], grads["b1"] = affine_relu_backward(dh1, cache1)
        # L2 姝ｅ垯锛堢害瀹氬甫 0.5 绯绘暟锛夛細loss += 0.5*reg*危W虏锛屾搴?+= reg*W
        loss += 0.5 * self.reg * (np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2))
        grads["W1"] += self.reg * self.params["W1"]
        grads["W2"] += self.reg * self.params["W2"]

#                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True



class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        dims = [input_dim] + list(hidden_dims) + [num_classes]
        for l in range(1, self.num_layers + 1):
            self.params["W%d" % l] = weight_scale * np.random.randn(dims[l - 1], dims[l])
            self.params["b%d" % l] = np.zeros(dims[l])
        # gamma/beta for batch/layer normalization (hidden-layer outputs)
        if self.normalization is not None:
            for l in range(1, self.num_layers):
                self.params["gamma%d" % l] = np.ones(dims[l])
                self.params["beta%d" % l] = np.zeros(dims[l])
        # dropout config (only used when dropout_keep_ratio != 1)
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed
        # batchnorm keeps running stats per layer; layernorm needs no mode
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for _ in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for _ in range(self.num_layers - 1)]
        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        # ---------- forward ----------
        out = X
        caches = []
        for l in range(1, self.num_layers):
            out, aff_cache = affine_forward(out, self.params["W%d" % l], self.params["b%d" % l])
            norm_cache = None
            if self.normalization is not None:
                gamma = self.params["gamma%d" % l]
                beta = self.params["beta%d" % l]
                if self.normalization == "batchnorm":
                    out, norm_cache = batchnorm_forward(out, gamma, beta, self.bn_params[l - 1])
                elif self.normalization == "layernorm":
                    out, norm_cache = layernorm_forward(out, gamma, beta, self.bn_params[l - 1])
            out, relu_cache = relu_forward(out)
            if self.use_dropout:
                out, drop_cache = dropout_forward(out, self.dropout_param)
            else:
                drop_cache = None
            caches.append((aff_cache, norm_cache, drop_cache, relu_cache))
        scores, cache_last = affine_forward(
            out, self.params["W%d" % self.num_layers], self.params["b%d" % self.num_layers]
        )

        if mode == "test":
            return scores

        # ---------- loss and backward ----------
        loss, grads = 0.0, {}
        loss, dscores = softmax_loss(scores, y)
        dout, grads["W%d" % self.num_layers], grads["b%d" % self.num_layers] = affine_backward(dscores, cache_last)
        for l in range(self.num_layers - 1, 0, -1):
            aff_cache, norm_cache, drop_cache, relu_cache = caches[l - 1]
            if self.use_dropout:
                dout = dropout_backward(dout, drop_cache)
            dout = relu_backward(dout, relu_cache)
            if self.normalization is not None:
                if self.normalization == "batchnorm":
                    dout, grads["gamma%d" % l], grads["beta%d" % l] = batchnorm_backward(dout, norm_cache)
                elif self.normalization == "layernorm":
                    dout, grads["gamma%d" % l], grads["beta%d" % l] = layernorm_backward(dout, norm_cache)
            dout, grads["W%d" % l], grads["b%d" % l] = affine_backward(dout, aff_cache)

        for l in range(1, self.num_layers + 1):
            W = self.params["W%d" % l]
            loss += 0.5 * self.reg * np.sum(W * W)
            grads["W%d" % l] += self.reg * W

        return loss, grads

    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True