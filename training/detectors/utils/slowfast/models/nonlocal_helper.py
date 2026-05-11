                      
                                                                       

                      

import torch
import torch.nn as nn


class Nonlocal(nn.Module):
\
\
\
\
\
\
\
       

    def __init__(
        self,
        dim,
        dim_inner,
        pool_size=None,
        instantiation="softmax",
        zero_init_final_conv=False,
        zero_init_final_norm=True,
        norm_eps=1e-5,
        norm_momentum=0.1,
        norm_module=nn.BatchNorm3d,
    ):
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
           
        super(Nonlocal, self).__init__()
        self.dim = dim
        self.dim_inner = dim_inner
        self.pool_size = pool_size
        self.instantiation = instantiation
        self.use_pool = (
            False
            if pool_size is None
            else any((size > 1 for size in pool_size))
        )
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum
        self._construct_nonlocal(
            zero_init_final_conv, zero_init_final_norm, norm_module
        )

    def _construct_nonlocal(
        self, zero_init_final_conv, zero_init_final_norm, norm_module
    ):
                                                     
        self.conv_theta = nn.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_phi = nn.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )
        self.conv_g = nn.Conv3d(
            self.dim, self.dim_inner, kernel_size=1, stride=1, padding=0
        )

                                   
        self.conv_out = nn.Conv3d(
            self.dim_inner, self.dim, kernel_size=1, stride=1, padding=0
        )
                                                         
        self.conv_out.zero_init = zero_init_final_conv

                                         
        self.bn = norm_module(
            num_features=self.dim,
            eps=self.norm_eps,
            momentum=self.norm_momentum,
        )
                                         
        self.bn.transform_final_bn = zero_init_final_norm

                                                       
        if self.use_pool:
            self.pool = nn.MaxPool3d(
                kernel_size=self.pool_size,
                stride=self.pool_size,
                padding=[0, 0, 0],
            )

    def forward(self, x):
        x_identity = x
        N, C, T, H, W = x.size()

        theta = self.conv_theta(x)

                                                                     
        if self.use_pool:
            x = self.pool(x)

        phi = self.conv_phi(x)
        g = self.conv_g(x)

        theta = theta.view(N, self.dim_inner, -1)
        phi = phi.view(N, self.dim_inner, -1)
        g = g.view(N, self.dim_inner, -1)

                                                             
        theta_phi = torch.einsum("nct,ncp->ntp", (theta, phi))
                                                                            
                              
                                                   
                                         
        if self.instantiation == "softmax":
                                                                       
            theta_phi = theta_phi * (self.dim_inner ** -0.5)
            theta_phi = nn.functional.softmax(theta_phi, dim=2)
        elif self.instantiation == "dot_product":
            spatial_temporal_dim = theta_phi.shape[2]
            theta_phi = theta_phi / spatial_temporal_dim
        else:
            raise NotImplementedError(
                "Unknown norm type {}".format(self.instantiation)
            )

                                                             
        theta_phi_g = torch.einsum("ntg,ncg->nct", (theta_phi, g))

                                           
        theta_phi_g = theta_phi_g.view(N, self.dim_inner, T, H, W)

        p = self.conv_out(theta_phi_g)
        p = self.bn(p)
        return x_identity + p
