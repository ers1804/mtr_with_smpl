# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi
# All Rights Reserved


import torch
import torch.nn as nn


def get_batch_offsets(batch_idxs, bs):
    '''
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    '''
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()])
            else:
                layers.extend(
                    [nn.Linear(c_in, mlp_channels[k], bias=False), nn.BatchNorm1d(mlp_channels[k]), nn.ReLU()])
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)


class PointNetPolylineEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None):
        super().__init__()
        self.pre_mlps = build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False
        )
        self.mlps = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False
        )

        if out_channels is not None:
            self.out_mlps = build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels],
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None

    def forward(self, polylines, polylines_mask):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):

        Returns:
        """
        batch_size, num_polylines, num_points_each_polylines, C = polylines.shape

        # pre-mlp
        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])  # (N, C)
        polylines_feature = polylines.new_zeros(batch_size, num_polylines, num_points_each_polylines,
                                                polylines_feature_valid.shape[-1])
        polylines_feature[polylines_mask] = polylines_feature_valid

        # get global feature
        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature = torch.cat(
            (polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)

        # mlp
        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(batch_size, num_polylines, num_points_each_polylines,
                                                      polylines_feature_valid.shape[-1])
        feature_buffers[polylines_mask] = polylines_feature_valid

        if self.out_mlps is not None:
            # max-pooling
            feature_buffers = feature_buffers.max(dim=2)[0]  # (batch_size, num_polylines, C)
            valid_mask = (polylines_mask.sum(dim=-1) > 0)
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])  # (N, C)
            feature_buffers = feature_buffers.new_zeros(batch_size, num_polylines, feature_buffers_valid.shape[-1])
            feature_buffers[valid_mask] = feature_buffers_valid

        return feature_buffers
    

class PoseEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=3, out_channels=None, time_encoder='rnn'):
        super().__init__()
        self.pre_mlps = build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=True
        )
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.time_encoder = time_encoder
        self.encoder = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
        if out_channels is not None:
            self.out_mlps = build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels], 
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None 

    def forward(self, trajectories, mask):
        """
        Args:
            trajectories (batch_size, num_timestamps, C):

        Returns:
        """
        num_center_agents, num_timesteps, C = trajectories.shape

        # pre-mlp
        valid_mask = torch.all(mask, dim=-1)
        buffer = torch.zeros((num_center_agents, self.hidden_dim), device=trajectories.device)
        trajectories_feature = self.pre_mlps(trajectories)  # (N, time, C)
        #trajectories_feature = trajectories.new_zeros(num_center_agents, num_agents,  num_timesteps, trajectories_feature_valid.shape[-1])
        #trajectories_feature[trajectories_mask] = trajectories_feature_valid


        _, hidden = self.encoder(trajectories_feature)
        feature_buffers = hidden[-1]
        buffer[valid_mask] = feature_buffers[valid_mask]
        
        # out-mlp 
        if self.out_mlps is not None:
            feature_buffers = self.out_mlps(feature_buffers)  # (N, C)
            masked_buffers = torch.zeros_like(feature_buffers)
            masked_buffers[valid_mask] = feature_buffers[valid_mask]
            return masked_buffers
        else:
            return buffer
