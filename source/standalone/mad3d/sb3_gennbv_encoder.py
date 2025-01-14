'''
import gymnasium as gym
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# self-attention layer for target (lookat xyz) proposal net
class SelfAttention3D(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(SelfAttention3D, self).__init__()
        self.query_conv = nn.Conv3d(in_channels, hidden_dim, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, hidden_dim, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, hidden_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.BatchNorm3d(hidden_dim)  # Use BatchNorm for 3D spatial data

    def forward(self, x):
        batch_size, C, D, H, W = x.size()
        query = self.query_conv(x).view(batch_size, -1, D * H * W).permute(0, 2, 1)  # B x N x C
        key = self.key_conv(x).view(batch_size, -1, D * H * W)  # B x C x N
        value = self.value_conv(x).view(batch_size, -1, D * H * W)  # B x C x N
        attention = torch.bmm(query, key)  # B x N x N
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, C, D, H, W)

        out = self.norm(out)

        out = self.gamma * out + x  # Residual connection
        return out


class ProposalNet(nn.Module):
    def __init__(self, input_channels=10, grid_size=(20, 20, 20)):
        super(ProposalNet, self).__init__()

        # (Unchanged) 3D CNN architecture
        #self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        #self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=2, dilation=2)
        #self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=4, dilation=4)

        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=2, dilation=2),
            nn.BatchNorm3d(32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=4, dilation=4),
            nn.BatchNorm3d(64)
        )

        self.self_attention = SelfAttention3D(in_channels=64, hidden_dim=64)
        
        flattened_size = 5 * 5 * 5 * 64
        
        # [CHANGE #1] Instead of a single fc1 -> xyz with sigmoid,
        # we define two separate linear layers for mean and log_std:
        self.fc_mean = nn.Linear(flattened_size, 256)  # Mean of xyz
        #self.fc_log_std = nn.Linear(flattened_size, 3)  # Log std of xyz
        self.log_std = nn.Parameter(th.ones(3) * 1, requires_grad=True)


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        
        x = F.leaky_relu(self.self_attention(x))
        
        x = x.view(x.size(0), -1)
        
        # [CHANGE #1 continued] Sample xyz from Gaussian distribution
        mean = self.fc_mean(x)
        #log_std = self.fc_log_std(x)
        #std = torch.exp(self.log_std)  
        #eps = torch.randn_like(std)
        #xyz = mean + std * eps  # Reparameterization trick

        # (Optional) If you need xyz in [0,1], do something like:
        #xyz = torch.sigmoid(xyz)
        #xyzmean = torch.sigmoid(mean)
        #xyz = torch.tanh(xyz)

        return mean#, xyzmean, self.log_std


class ViewEncoder(nn.Module):
    def __init__(self, input_channels=10):
        super(ViewEncoder, self).__init__()

        # 3D CNN layers with dilated convolutions
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),  # Add BatchNorm
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=2, dilation=2),
            nn.BatchNorm3d(32),  # Add BatchNorm
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=4, dilation=4),
            nn.BatchNorm3d(64),  # Add BatchNorm
            nn.LeakyReLU()
        )



    def trilinear_interpolation(self, features, xyz_coords):
        # Constrain xyz_coords between 0 and 1
        xyz_coords = torch.clamp(xyz_coords, 0, 1)
        
        # Reshape xyz_coords to fit the required shape for grid_sample
        batch_size = features.shape[0]
        xyz_coords = xyz_coords.view(batch_size, 1, 1, 1, 3)  # (B, D=1, H=1, W=1, C=3)
        xyz_coords = 2.0 * xyz_coords - 1.0  # Normalize to [-1, 1] for grid_sample
        
        # Perform trilinear interpolation using PyTorch's grid_sample
        interpolated_features = F.grid_sample(features, xyz_coords, mode='bilinear', align_corners=True)
        
        return interpolated_features.view(batch_size, -1)

    def forward(self, grid, xyz_coords):
        # Feature extraction at multiple levels using dilated convolutions
        level1 = self.conv1(grid)  # Regular convolution
        level2 = self.conv2(level1)  # Dilated convolution with dilation=2
        level3 = self.conv3(level2)  # Dilated convolution with dilation=4

        # Trilinear interpolation at each level
        interp1 = self.trilinear_interpolation(level1, xyz_coords)
        interp2 = self.trilinear_interpolation(level2, xyz_coords)
        interp3 = self.trilinear_interpolation(level3, xyz_coords)
        
        # Combine features
        combined_features = torch.cat([interp1, interp2, interp3], dim=1)
        
        return combined_features


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
       
        for key, subspace in observation_space.spaces.items():
           
            if key == "pose_step":
                extractors[key] = nn.Identity() 
                total_concat_size += 6
 
            if key == "img":               
                c, h, w = subspace.shape
                extractors[key] = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=2, padding=1),
                                                nn.BatchNorm2d(12),  # Add BatchNorm
                                                nn.LeakyReLU(),  # First ReLU activation
                                                nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=2, padding=1), 
                                                nn.BatchNorm2d(12),  # Add BatchNorm
                                                nn.LeakyReLU(),   # Second ReLU activation
                                                nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=2, padding=1), 
                                                nn.BatchNorm2d(12),  # Add BatchNorm
                                                nn.LeakyReLU(),   # Second ReLU activation
                                                nn.Flatten(),  # Flatten the output to feed into linear layer
                                                nn.Linear(in_features=972, out_features=256), # Linear layer to 256 units
                                                nn.LeakyReLU(),   # Second ReLU activation
                                            )
                total_concat_size += 256
            
            if key == "occ":
                proposal_net = ProposalNet()
                view_encoder = ViewEncoder()
                extractors["occ_pro"] = proposal_net
                extractors["occ_view"] = view_encoder
                total_concat_size += 16 + 32 + 64

        self.extractors = nn.ModuleDict(extractors)

        # fusion layer
        self.fuse = nn.Sequential(
                            nn.Linear(total_concat_size, 256),
                            nn.LeakyReLU(),
                            )

        self._features_dim = 256

    def forward(self, observations) -> th.Tensor:
        #encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            if "img" == key:
                xyz = self.extractors["img"](observations["img"])
            #elif "occ_pro" == key:
            #    xyz = self.extractors["occ_pro"](observations["occ"])
            #    #out = self.extractors["occ_view"](observations["occ"], xyz.detach())
            #    #encoded_tensor_list.append(out)
            #elif "occ" in key:
            #    pass
            #else:
            #    encoded_tensor_list.append(extractor(observations[key]))
        
        #out = th.cat(encoded_tensor_list, dim=1)
        #out = self.fuse(out)

        return xyz
'''
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
            
            if key == "pose_step":
                extractors[key] = nn.Sequential(
                                    nn.Linear(subspace.shape[0], 32),
                                    nn.LeakyReLU(),
                                    )
               
                #extractors[key] = nn.Sequential(
                #                    nn.Linear(subspace.shape[0], 64),
                #                    nn.LeakyReLU(),
                #                    nn.Linear(64, 64),
                #                    nn.LeakyReLU(),
                #                    )
                
                total_concat_size += 32
            
            if key == "img":
                small_net = True
                if small_net:               
                    c, h, w = subspace.shape
                    extractors[key] = nn.Sequential(nn.Conv2d(in_channels=5, out_channels=12, kernel_size=5, stride=2, padding=1),
                                                    nn.LeakyReLU(),  # First ReLU activation
                                                    nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=2, padding=1),
                                                    nn.LeakyReLU(),   # Second ReLU activation
                                                    nn.Flatten(),  # Flatten the output to feed into linear layer
                                                    nn.Linear(in_features=2700, out_features=256),  # Linear layer to 256 units
                                                    nn.LeakyReLU(),   # Second ReLU activation
                                                )

                   
                    total_concat_size += 256
                else: 
                    c, h, w = subspace.shape
                    extractors[key] = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=1),
                                                    nn.LeakyReLU(),  # First ReLU activation
                                                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=1),
                                                    nn.LeakyReLU(),   # Second ReLU activation
                                                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=1),
                                                    nn.LeakyReLU(),   # Second ReLU activation
                                                    nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=2, padding=1),
                                                    nn.LeakyReLU(),   # Second ReLU activation
                                                    nn.Flatten(),  # Flatten the output to feed into linear layer
                                                    nn.Linear(in_features=85264, out_features=256),  # Linear layer to 256 units
                                                    nn.LeakyReLU()
                                                )
                    total_concat_size += 256   
            
            if key == "occ":
                small_net = True
                if small_net:
                    c, x_res, y_res, z_res = subspace.shape
                    x_out = (x_res + 2 - 3) // 2 + 1
                    x_out = (x_out + 2 - 3) // 2 + 1
                    y_out = (y_res + 2 - 3) // 2 + 1
                    y_out = (y_out + 2 - 3) // 2 + 1
                    z_out = (z_res + 2 - 3) // 2 + 1
                    z_out = (z_out + 2 - 3) // 2 + 1
                    extractors[key] = nn.Sequential(nn.Conv3d(in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1),
                                                    nn.LeakyReLU(),
                                                    nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                                                    nn.LeakyReLU(),
                                                    nn.Flatten(),  # Flatten the output to feed into linear layer
                                                    nn.Linear(in_features=x_out * y_out * z_out * 32, out_features=256),
                                                    nn.LeakyReLU()
                                                  )
                else:
                    c, x_res, y_res, z_res = subspace.shape
                    extractors[key] = nn.Sequential(nn.Conv3d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
                                                    nn.LeakyReLU(),
                                                    nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                                                    nn.LeakyReLU(),
                                                    nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                                    nn.LeakyReLU(),
                                                    nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                                    nn.LeakyReLU(),
                                                    nn.Flatten(),  # Flatten the output to feed into linear layer
                                                    nn.Linear(in_features=x_res * y_res * z_res * 32, out_features=256),
                                                    nn.LeakyReLU(),
                                                  )
                total_concat_size += 256
            
            #if key == "env_step":
                #extractors[key] = nn.Sequential(nn.Linear(1, 2))
                #nn.Sequential(nn.Identity())
                #total_concat_size += 2

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            #print(key, extractor(observations[key]).shape, observations[key].shape)
            encoded_tensor_list.append(extractor(observations[key]))
            #print(key, extractor(observations[key]).shape, observations[key].shape)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
