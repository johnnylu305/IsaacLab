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
    def __init__(self, input_channels=4, grid_size=(20, 20, 20)):
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

        self.fc = nn.Linear(flattened_size, 256)  # Mean of xyz

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        
        x = F.leaky_relu(self.self_attention(x))
        
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

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
                                                nn.Linear(in_features=15552, out_features=256), # Linear layer to 256 units
                                                nn.LeakyReLU(),   # Second ReLU activation
                                            )
                total_concat_size += 256
            
            if key == "occ":
                proposal_net = ProposalNet()
                extractors["occ_pro"] = proposal_net
                total_concat_size += 256

        self.extractors = nn.ModuleDict(extractors)

        # fusion layer
        self.fuse = nn.Sequential(
                            nn.Linear(total_concat_size, 256),
                            nn.LeakyReLU(),
                            )

        self._features_dim = 256

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            if "occ_pro" == key:
                out = self.extractors["occ_pro"](observations["occ"])
                encoded_tensor_list.append(out)
            elif "occ" in key:
                pass
            else:
                encoded_tensor_list.append(extractor(observations[key]))
        
        out = th.cat(encoded_tensor_list, dim=1)
        out = self.fuse(out)

        return out
