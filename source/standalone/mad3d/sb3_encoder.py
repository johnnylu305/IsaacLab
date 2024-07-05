import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
       
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "pose":
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 32))
                total_concat_size += 32
            if key == "img":
                c, h, w = subspace.shape
                extractors[key] = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=2, padding=1), # First conv layer
                                                nn.ReLU(),  # First ReLU activation
                                                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # Second conv layer
                                                nn.ReLU(),   # Second ReLU activation
                                                nn.Flatten(),  # Flatten the output to feed into linear layer
                                                nn.Linear(in_features=(h//4) * (w//4) * 32, out_features=256)  # Linear layer to 256 units
                                            )
                total_concat_size += 256
            if key == "occ":
                c, x_res, y_res, z_res = subspace.shape
                x_out = (x_res + 2 - 3) // 2 + 1
                x_out = (x_out + 2 - 3) // 2 + 1
                y_out = (y_res + 2 - 3) // 2 + 1
                y_out = (y_out + 2 - 3) // 2 + 1
                z_out = (z_res + 2 - 3) // 2 + 1
                z_out = (z_out + 2 - 3) // 2 + 1
                extractors[key] = nn.Sequential(nn.Conv3d(in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1),
                                                nn.ReLU(),
                                                nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                                                nn.ReLU(),
                                                nn.Flatten(),  # Flatten the output to feed into linear layer
                                                nn.Linear(in_features=x_out * y_out * z_out * 32, out_features=256)  # Linear layer to 256 units
                                              )

                total_concat_size += 256          

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
