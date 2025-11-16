from typing import Sequence

import torch
import torch.nn as nn
from tqdm import tqdm
#
#
# class Encoder(nn.Module):
#     def __init__(self,
#                  input_dim: int,
#                  feature_dim: int,
#                  middle_dims: Sequence[int] = (1024, 512, 256),
#                  use_linear_projection: bool = False):
#         super(Encoder, self).__init__()
#         middle_dims = [input_dim] + list(middle_dims) + [feature_dim]
#         middle_layers = nn.ModuleList()
#         for i in range(len(middle_dims) - 2):
#             layer = [nn.Linear(middle_dims[i], middle_dims[i + 1])]
#             if not use_linear_projection:
#                 layer.extend([
#                     nn.BatchNorm1d(middle_dims[i + 1]),
#                     nn.ReLU(inplace=True)
#                 ])
#             middle_layers.append(nn.Sequential(*layer))
#         middle_layers.append(nn.Linear(middle_dims[-2], middle_dims[-1]))
#         # Completer在编码层加了一层Softmax
#         # middle_layers.append(nn.Softmax(dim=1))
#         self.middle_layers = middle_layers
#         self.middle_dims = middle_dims
#
#     def forward(self, x):
#         for layer in self.middle_layers:
#             x = layer(x)
#         return x
#
#
# class Decoder(nn.Module):
#     def __init__(self,
#                  feature_dim: int,
#                  output_dim: int,
#                  middle_dims: Sequence[int] = (256, 512, 1024),
#                  use_linear_projection: bool = False):
#         super(Decoder, self).__init__()
#         middle_dims = [feature_dim] + list(middle_dims) + [output_dim]
#         middle_layers = nn.ModuleList()
#         for i in range(len(middle_dims) - 2):
#             layer = [nn.Linear(middle_dims[i], middle_dims[i + 1])]
#             if not use_linear_projection:
#                 layer.extend([
#                     nn.BatchNorm1d(middle_dims[i + 1]),
#                     nn.ReLU(inplace=True)
#                 ])
#             middle_layers.append(nn.Sequential(*layer))
#         middle_layers.append(nn.Linear(middle_dims[-2], middle_dims[-1]))
#         # 控制输出范围在0-1之间
#         middle_layers.append(nn.Sigmoid())
#         self.middle_layers = middle_layers
#         self.middle_dims = middle_dims
#
#     def forward(self, x):
#         for layer in self.middle_layers:
#             x = layer(x)
#         return x
#
#
# # 重构和Prediction都可以使用自编码器的结构
# class AutoEncoder(nn.Module):
#     def __init__(self,
#                  input_dim: int,
#                  feature_dim: int,
#                  middle_dims: Sequence[int] = (1024, 512, 256),
#                  use_linear_projection: bool = False):
#         super(AutoEncoder, self).__init__()
#         output_dim = input_dim
#         self.middle_dims = [input_dim] + list(middle_dims) + [feature_dim]
#         self.encoder = Encoder(input_dim, feature_dim, middle_dims, use_linear_projection)
#         self.decoder = Decoder(feature_dim, output_dim, middle_dims[::-1], use_linear_projection)
#
#     def forward(self, x):
#         hidden = self.encoder(x)
#         x_rec = self.decoder(hidden)
#         return hidden, x_rec
#
#     def pretrain(self, x, epochs=100, lr=1e-2, weight_decay=0.):
#         optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
#         criterion = nn.MSELoss()
#         self.train()
#         for epoch in range(epochs):
#             h, x_rs = self(x)
#             loss = criterion(x, x_rs)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         # 返回最后一次的结果
#         return h, x_rs
#
#
# # 仅编码器
# class MultiviewPrediction(nn.Module):
#     def __init__(self, view_dims, latent_dim, middle_encoders, use_linear_projection=False):
#         super(MultiviewPrediction, self).__init__()
#         self.num_view = len(view_dims)
#         # 构建num_view个编码器
#         encoder_list = nn.ModuleList()
#         for i in range(self.num_view):
#             if middle_encoders is None:
#                 encoder = Encoder(view_dims[i], latent_dim, use_linear_projection=use_linear_projection)
#             else:
#                 encoder = Encoder(view_dims[i], latent_dim, middle_encoders[i], use_linear_projection)
#             encoder_list.append(encoder)
#         self.view_dims = view_dims
#         self.latent_dim = latent_dim
#         self.encoder_list = encoder_list
#
#     def __getitem__(self, i):
#         return self.encoder_list[i]
#
#     def forward(self, x):
#         hidden_list = []
#         for i in range(self.num_view):
#             encoder = self.encoder_list[i]
#             hidden = encoder(x[i])
#             hidden_list.append(hidden)
#         return hidden_list
#
#
# class MultiviewAutoEncoder(nn.Module):
#     def __init__(self, view_dims, latent_dim, middle_encoders, use_linear_projection=False):
#         super(MultiviewAutoEncoder, self).__init__()
#         num_view = len(view_dims)
#         # 构建num_view个自编码器
#         autoencoder_list = nn.ModuleList()
#         for i in range(num_view):
#             if middle_encoders is None:
#                 autoencoder = AutoEncoder(view_dims[i], latent_dim, use_linear_projection=use_linear_projection)
#             else:
#                 autoencoder = AutoEncoder(view_dims[i], latent_dim,
#                                           middle_dims=middle_encoders[i], use_linear_projection=use_linear_projection)
#             autoencoder_list.append(autoencoder)
#         self.num_view = num_view
#         self.view_dims = view_dims
#         self.latent_dim = latent_dim
#         self.autoencoder_list = autoencoder_list
#
#     def __getitem__(self, i):
#         return self.autoencoder_list[i]
#
#     def forward(self, x):
#         """
#         :param x: multi view data with shape num_view * [batch_size * feature_dim_v]
#         :return: multi view latent feature with view-specific autoencoders, and reconstructed view
#         """
#         hidden_list, x_rs = [], []
#         for i in range(self.num_view):
#             autoencoder = self.autoencoder_list[i]
#             hidden = autoencoder.encoder(x[i])
#             hidden_list.append(hidden)
#             x_r_view = autoencoder.decoder(hidden)
#             x_rs.append(x_r_view)
#         return hidden_list, x_rs
#
#     def pretrain(self, dataloader, epochs, device, lr=1e-2, weight_decay=1e-4):
#         optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
#         criterion = nn.MSELoss()
#         self.train()
#         for epoch in range(epochs):
#             loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
#             for bid, (x, _) in loop:
#                 for i in range(len(x)):
#                     x[i] = x[i].to(device)
#                 _, x_rs = self.forward(x)
#                 loss = torch.tensor(0., device=device)
#                 for i in range(len(x)):
#                     loss += criterion(x[i], x_rs[i])
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 loop.set_description(desc=f"Pretrain Epoch [{epoch}/{epochs}]")
#                 loop.set_postfix(loss=loss.item())
#
#
# class MultiviewAutoEncoderWithAvgpool(nn.Module):
#     def __init__(self, view_dims, latent_dim, middle_encoders, use_linear_projection=False):
#         super(MultiviewAutoEncoderWithAvgpool, self).__init__()
#         num_view = len(view_dims)
#         # 构建num_view个自编码器
#         autoencoder_list = nn.ModuleList()
#         for i in range(num_view):
#             if middle_encoders is None:
#                 autoencoder = AutoEncoder(view_dims[i], latent_dim, use_linear_projection=use_linear_projection)
#             else:
#                 autoencoder = AutoEncoder(view_dims[i], latent_dim,
#                                           middle_dims=middle_encoders[i], use_linear_projection=use_linear_projection)
#             autoencoder_list.append(autoencoder)
#         self.num_view = num_view
#         self.view_dims = view_dims
#         self.latent_dim = latent_dim
#         self.autoencoder_list = autoencoder_list
#
#     def forward(self, x):
#         hidden_list = []
#         for i in range(self.num_view):
#             autoencoder = self.autoencoder_list[i]
#             hidden = autoencoder.encoder(x[i])
#             hidden_list.append(hidden)
#         avg_hidden = torch.sum(torch.stack(hidden_list, dim=0), dim=0) / self.num_view
#         x_r = []
#         for i in range(self.num_view):
#             autoencoder = self.autoencoder_list[i]
#             x_r_view = autoencoder.decoder(avg_hidden)
#             x_r.append(x_r_view)
#         return avg_hidden, x_r, hidden_list
#
#
# class Normalize(nn.Module):
#     def __init__(self, p=2, dim=1):
#         super(Normalize, self).__init__()
#         self.normalize = nn.functional.normalize
#         self.p = p
#         self.dim = dim
#
#     def forward(self, x):
#         return self.normalize(x, p=self.p, dim=self.dim)


import math

import torch.nn as nn
from torch.nn.functional import normalize
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class MultiHead(nn.Module):
    def __init__(self, feature_dim, high_feature_dim, dropout=None):
        super(MultiHead, self).__init__()
        self.head = feature_dim // high_feature_dim
        self.feature_dim = feature_dim
        self.high_feature_dim = high_feature_dim
        self.multi_lines_module = []
        self.dropout = None
        if(dropout is not None):
            self.dropout = nn.Dropout(dropout)

        for i in range(self.head):
            self.multi_lines_module.append(
                nn.Sequential(
                    nn.Linear(feature_dim, high_feature_dim),
                )
            )
        self.multi_lines_module = nn.ModuleList(self.multi_lines_module)
        self.attenLinear = nn.Linear(self.feature_dim, self.feature_dim)
    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if(mask is not None):
            scores = scores.masked_fill(mask=mask, value=torch.tensor(-1e9))

        p_attn = F.softmax(scores, dim=-1)
        if(self.dropout is not None):
            p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

    def forward(self, x, k):
        """
        x: B,V,L
        k: B,V,L
        """
        B,V,L = x.shape
        # print(f"B V L: {B},{V},{L}")#  256,6,512
        attens = []
        for i in range(self.head):
            q = self.multi_lines_module[i](x)
            v = (q+k)/2.0
            atten, p_attn = self.attention(q,k,v)
            attens.append(atten) # (H, B, V, L)
        xAtten = torch.stack(attens, dim=0).permute(1, 2, 0, 3)
        # print(f"xAtten shape: {xAtten.shape}") # xAtten shape: torch.Size([256, 6, 4, 128])
        xAtten = xAtten.transpose(1,2).contiguous().view(B,-1,self.head*self.high_feature_dim)

        return self.attenLinear(xAtten)



class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device, dropout = 0.2):
        super(Network, self).__init__()
        self.view = view
        self.encoders = []
        self.decoders = []
        # self.nDecoders = []
        self.feature_contrastive_modules = []
        self.noise_modules = []
        self.private_modules = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim+high_feature_dim))
            # self.nDecoders.append(Decoder(input_size[v], feature_dim+high_feature_dim))
            self.feature_contrastive_modules.append(
                nn.Sequential(
                    nn.Linear(feature_dim, high_feature_dim),
                    # nn.ReLU(),
                )
            )
            self.noise_modules.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(high_feature_dim, high_feature_dim),
                )
            )
            self.private_modules.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(high_feature_dim, high_feature_dim),
                )
            )
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_contrastive_modules = nn.ModuleList(self.feature_contrastive_modules)
        self.noise_modules = nn.ModuleList(self.noise_modules)
        self.private_modules = nn.ModuleList(self.private_modules)
        #
        self.multiAttn = MultiHead(feature_dim,high_feature_dim,dropout=dropout)
        #
        self.PNs = []
        self.PPs = []

    def forward(self, xs):
        hs = []
        qs = []
        xLNs = []
        xLPs = []
        xHPs = []
        xLPss = []
        xHPss = []
        zs = []
        zNs = []
        zPs = []
        lHZs = []
        hHZs = []

        for v in range(self.view):
            x = xs[v]
            h = self.encoders[v](x)
            z = normalize(self.feature_contrastive_modules[v](h), dim=1)
            # print("z device:", z.data.device)
            zNoise = self.noise_modules[v](z)
            zPrivate = self.private_modules[v](z)
            # print(f"zNoise shape: {zNoise.shape}, h shape: {h.shape}")# zNoise shape: torch.Size([256, 128]), h shape: torch.Size([256, 512])
            # lowProcessHidNoise = torch.cat([h,zNoise], dim=1)
            lowProcessHidPrivate = torch.cat([h,zPrivate], dim=1)
            # print(f"lowProcessHidNoise shape: {lowProcessHidNoise.shape}")# lowProcessHidNoise shape: torch.Size([256, 640])
            # xLN = self.nDecoders[v](lowProcessHidNoise)
            # xLP = self.decoders[v](lowProcessHidPrivate)
            # xLNs.append(xLN)
            # xLPs.append(xLP)
            hs.append(h)
            zs.append(z)
            zNs.append(zNoise)
            zPs.append(zPrivate)
            lHZs.append(lowProcessHidPrivate)
            self.PNs.append(zNoise)
            self.PPs.append(zPrivate)

        # print(f"hs[0] shape: {hs[0].shape}")# hs[0] shape: torch.Size([256, 512])
        query = torch.stack(hs, dim=0).permute(1, 0, 2)
        # print(f"query shape: {query.shape}")# query shape: torch.Size([256, 6, 512])
        key = torch.stack(zPs, dim=0).permute(1, 0, 2)
        # print(f"key shape: {key.shape}") # key shape: torch.Size([256, 6, 128])
        xAttn = self.multiAttn(query, key)
        xAttn = xAttn.permute(1, 0, 2)
        # print(f"xAttn shape: {xAttn.shape}") # xAttn shape: torch.Size([6, 256, 512])
        for v in range(self.view):
            tem_xLPs = []
            #
            # tem_xHPs = []
            # x = xAttn[v]
            for w in range(self.view):
                crossFea = torch.cat([hs[v],zPs[w]], dim=1)
                xLP = self.decoders[w](crossFea)
                tem_xLPs.append(xLP)
                #
                # zPrivate = zPs[w]
                # highProcessHidPrivate = torch.cat([x, zPrivate], dim=1)
                # xHP = self.decoders[w](highProcessHidPrivate)
                # tem_xHPs.append(xHP)
            xLPss.append(tem_xLPs)
            #
            # xHPss.append(tem_xHPs)

            x = xAttn[v]
            zPrivate = zPs[v]
            highProcessHidPrivate = torch.cat([x, zPrivate], dim=1)
            xHP = self.decoders[v](highProcessHidPrivate)
            xHPs.append(xHP)
            hHZs.append(highProcessHidPrivate)

        # return zs, lHZs, hHZs, xLNs, xLPs, xHPs, hs, zNs, zPs
        return zs, lHZs, hHZs, xLNs, xLPss, xHPs, hs, zNs, zPs
        # return zs, lHZs, hHZs, xLNs, xLPss, xHPss, hs, zNs, zPs

    def getZLoss(self):
        # 互信息量
        # for
        # 相似度
        loss_list = []
        for i in range(self.view):
            x1, x2 = self.PNs[i], self.PPs[i]
            cos_sim = F.cosine_similarity(x1, x2, dim=1)  # shape: (B,)

            cos_sim = torch.abs(cos_sim.view(-1, 1))  # shape: (B, 1)
            loss_list.append(cos_sim.mean())
        loss = sum(loss_list)
        return loss
