from typing import Sequence

import gin
import torch
from dopamine.jax import networks
from torch import nn
from torch.functional import F

cartpole_min_vals = torch.tensor(gin.query_parameter("jax_networks.CARTPOLE_MIN_VALS"))
cartpole_max_vals = torch.tensor(gin.query_parameter("jax_networks.CARTPOLE_MAX_VALS"))


def normalize_states(s: torch.Tensor) -> torch.Tensor:
    s -= cartpole_min_vals
    s /= cartpole_max_vals - cartpole_min_vals
    return 2.0 * s - 1.0


# NOTE no flattening, nor preprocessing right now...
def simple_mlp(layers: Sequence[int]) -> nn.Sequential:
    # compute intermediate dense sizes
    dense_sizes = list(zip(layers, layers[1:]))
    # interleave a dense layer with a ReLU activation function
    model_and_activations = [
        v
        for tup in zip(
            [nn.Linear(in_feat, out_feat) for in_feat, out_feat in dense_sizes[:-1]],
            [nn.ReLU()] * (len(dense_sizes) - 1),
        )
        for v in tup
    ]
    # put together in a sequential model
    return nn.Sequential(*model_and_activations, nn.Linear(*dense_sizes[-1]))


# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Our model will be a convolutional neural network that takes in the
# difference between the current and previous screen patches.
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
