import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'#0,1,2,3

import torch
#torch.cuda.set_device(3)
from torch.utils.data import Dataset, TensorDataset, DataLoader
import socket

import numpy as np
import scipy
import os
import torch.cuda
import torch.backends.cudnn as cudnn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


import pandas as pd
import graphviz
import dowhy
from dowhy import CausalModel
import networkx as nx

import matplotlib
# %matplotlib notebook
# %matplotlib inline  
from matplotlib import pyplot as plt

import socket
import datetime
import math
import random
import os
import sys
# temp_argv = sys.argv
import timeit

import pandas as pd
import seaborn as sns


import networkx as nx

# Set manual seed for reproduction of experiments
# seed = 1
# seed = random.randint(1, 20000)

# random.seed(seed)
# np.random.seed(seed=seed)
# torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
# else:
#     print("GPU device not available!")
    
from torch.distributions.normal import Normal

from timeit import default_timer as timer
import lib.utils as utils
from datetime import datetime
import yaml
import os
import UCIdatasets
import numpy as np
from models.Normalizers import *
from models.Conditionners import *
from models.NormalizingFlowFactories import buildFCNormalizingFlow, buildFCNormalizingFlow_UC
from models.NormalizingFlow import *
import math
import re


from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

# init_printing(use_unicode=True)


class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        
    def __getitem__(self, index):
        x = self.data[index]
        
        return x
    
    def __len__(self):
        return len(self.data)


def batch_iter(X, batch_size, shuffle=False):
    """
    X: feature tensor (shape: num_instances x num_features)
    """
    if shuffle:
        idxs = torch.randperm(X.shape[0])
    else:
        idxs = torch.arange(X.shape[0])
    if X.is_cuda:
        idxs = idxs.cuda()
    for batch_idxs in idxs.split(batch_size):
        yield X[batch_idxs]


def load_data(name,
        rho = 0.0,
        y_dim = 8# the dimension of the outcome to be used. 1:education, 2:health, 3:information, 4: malnutrition, 5: sanitization, 6: shelter, 7: water, 8: total degree.
             ):

    if name == 'bsds300':
        return UCIdatasets.BSDS300()

    elif name == 'power':
        return UCIdatasets.POWER()

    elif name == 'gas':
        return UCIdatasets.GAS()

    elif name == 'hepmass':
        return UCIdatasets.HEPMASS()

    elif name == 'miniboone':
        return UCIdatasets.MINIBOONE()

    elif name == "digits":
        return UCIdatasets.DIGITS()
    
    elif name == "proteins":
        return UCIdatasets.PROTEINS()
        
    elif name == "real_imf_categorical":
        return UCIdatasets.REAL_IMF_CATEGORICAL(
        rho = rho,
        y_dim=y_dim# the dimension of the outcome to be used. 1:education, 2:health, 3:information, 4: malnutrition, 5: sanitization, 6: shelter, 7: water, 8: total degree.
                                         )
    
    else:
        raise ValueError('Unknown dataset')


cond_types = {"DAG": DAGConditioner, "Coupling": CouplingConditioner, "Autoregressive": AutoregressiveConditioner}
norm_types = {"affine": AffineNormalizer, "monotonic": MonotonicNormalizer}


def train(dataset="toy_simulated_binary", load=True, nb_step_dual=100, nb_steps=20, path="", l1=.1, nb_epoch=10000,
          int_net=[200, 200, 200], emb_net=[200, 200, 200], b_size=100, all_args=None, file_number=None, train=True,
          solver="CC", nb_flow=1, weight_decay=1e-5, learning_rate=1e-3, cond_type='DAG', norm_type='affine', n_mce_samples=20000, mce_b_size=20000, nb_estop=50,seed=None, 
        rho = 0.0,
        y_dim = 8# the dimension of the outcome to be used. 1:education, 2:health, 3:information, 4: malnutrition, 5: sanitization, 6: shelter, 7: water, 8: total degree.
         ):
                                                
#     logger = utils.get_logger(logpath=os.path.join(path, 'logs'), filepath=os.path.abspath('/home/user/prj/rhoGNF/RealIMFCategorical.py'))
    logger = utils.get_logger(logpath=os.path.join(path, 'logs'), filepath=__file__)
    logger.info(str(all_args))


    device = "cpu" if not(torch.cuda.is_available()) else "cuda:0"
    
    # Set manual seed for reproduction of experiments
    if seed is None:
        seed = random.randint(1, 20000)
    logger.info(f"Running simulation with seed {seed}")


    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        print("GPU device not available!")

    if load:
        file_number_str = '_%05d' % file_number if file_number is not None else "_"

    if file_number is None:
        file_number = 0
    batch_size = b_size

    logger.info("Loading data...")
    data = load_data(dataset,
        rho = rho,
        y_dim=y_dim# the dimension of the outcome to be used. 1:education, 2:health, 3:information, 4: malnutrition, 5: sanitization, 6: shelter, 7: water, 8: total degree.
                    )
    data_mu = torch.from_numpy(data.mu).float().to(device)
    data_sigma = torch.from_numpy(data.sig).float().to(device)
    logger.info(f"data_mu = \n{data_mu.cpu().numpy()}, \ndata_sigma = \n{data_sigma.cpu().numpy()}")

    
#     logger.info(f"Data loaded from DGP {n_dgp}/{n_DGPs} with ATE = {data.ATE.item():.5f}.")
    
    d_trn = TensorDataset(data.trn.x.float())
    d_val = TensorDataset(data.val.x.float())
    d_tst = TensorDataset(data.tst.x.float())
    
    logger.info(f"Number of samples = trn:{len(d_trn):7d},  val:{len(d_val):7d},  tst:{len(d_tst):7d}")

    workers = 4
    pin_memory = True
    
    batch_size = b_size#128#2000#125
#     batch_size = int(len(d_trn)**(3/4))#2000#125
#     batch_size = len(d_trn)//16
#     batch_size = min(batch_size, len(d_trn)//10)
    
    logger.info(f"Batch size = trn:{batch_size:7d},  val:{len(d_val):7d},  tst:{len(d_tst):7d}")
    l_trn = DataLoader(d_trn, 
#                        batch_size=batch_size,
                       batch_size=batch_size,
#                              sampler=sampler_src,
                             num_workers=int(workers),
                             shuffle=True,
                             pin_memory=pin_memory,
                             drop_last=False) # create your dataloader
    l_val = DataLoader(d_val, 
#                        batch_size=batch_size,
                       batch_size=len(d_val)//4,
#                              sampler=sampler_src,
                             num_workers=int(workers),
                             shuffle=False,
                             pin_memory=pin_memory,
                             drop_last=False) # create your dataloader
    l_tst = DataLoader(d_tst, 
#                        batch_size=batch_size,
                       batch_size=len(d_tst)//4,
#                              sampler=sampler_src,
                             num_workers=int(workers),
                             shuffle=False,
                             pin_memory=pin_memory,
                             drop_last=False) # create your dataloader

    logger.info(f"Number of batches = trn:{len(l_trn):7d},  val:{len(l_val):7d},  tst:{len(l_tst):7d}")
    epoch_iters = len(l_trn)
    
    logger.info(f"Dataset_mean = {data.mu}")
    logger.info(f"Dataset_sigma = {data.sig}")
    
    
#     logger.info(f'\n\n# {len(d_trn)} ++ ORI - EY0 = {data.EY0.item():.5f}')
#     logger.info(f'# {len(d_trn)} ++ ORI - EY1 = {data.EY1.item():.5f}')
#     logger.info(f'# {len(d_trn)} ++ ORI - lambda_ATE = {data.ATE.item():.5f}\n\n')

#     # Open a file with access mode 'a'
#     with open(path + f"/_IPWRWR_ATE_lambda.txt", "a") as file_object:
#         file_object.write(f'  ,  {data.ATE.item():.5f}  # {len(d_trn)} ++ ORI - lambda_ATE =\n')

#     # Open a file with access mode 'a'
#     with open(path + "/_best_ATE_lambda_params.txt", "a") as file_object:
# #         data.p_U = p_U = torch.rand(1)#from_numpy(p_A1).float()
# #         print(f':: p_U = {p_U}')
#         file_object.write(f':: seed = {seed:.5f}\n')
#         file_object.write(f':: rho = {data.Z_Sigma[0,1].item():.5f}\n')
#         file_object.write(f':: n_dgp = {data.n_dgp}\n')
#         file_object.write(f':: p_U = {data.p_U.item():.5f}\n')

# #         data.p_AU0 = p_AU0 = torch.rand(1)#from_numpy(p_A1).float()
# #         data.p_AU1 = p_AU1 = torch.rand(1)#from_numpy(p_A1).float()
# # #             p_AU1 = 1-p_AU0#torch.rand()#from_numpy(p_A1).float()
# #         print(f':: p_AU0 = {p_AU0}, :: p_AU1 = {p_AU1}')
#         file_object.write(f':: p_AU0 = {data.p_AU0.item():.5f}, :: p_AU1 = {data.p_AU1.item():.5f}\n')

# #         data.p_YA0U0 = p_YA0U0 = torch.rand(1)#from_numpy(p_A1).float()
# #         data.p_YA0U1 = p_YA0U1 = torch.rand(1)#from_numpy(p_A1).float()
# #         data.p_YA1U0 = p_YA1U0 = torch.rand(1)#from_numpy(p_A1).float()
# #         data.p_YA1U1 = p_YA1U1 = torch.rand(1)#from_numpy(p_A1).float()
# # #             p_AU1 = 1-p_AU0#torch.rand()#from_numpy(p_A1).float()
# #         print(f':: p_YA0U0 = {p_YA0U0}, :: p_YA0U1 = {p_YA0U1}')
# #         print(f':: p_YA1U0 = {p_YA1U0}, :: p_YA1U1 = {p_YA1U1}')
#         file_object.write(f':: p_YA0U0 = {data.p_YA0U0.item():.5f}, :: p_YA0U1 = {data.p_YA0U1.item():.5f}\n')
#         file_object.write(f':: p_YA1U0 = {data.p_YA1U0.item():.5f}, :: p_YA1U1 = {data.p_YA1U1.item():.5f}\n')

# # #             p(y,a,u)=p(u)1(a)p(y|a,u)
# # #             p(y)=sum u p(u)1(a)p(y|a,u)
# # #             p(y|a1)= p(u0)1(a1)p(y|a1,u0)+p(u1)1(a1)p(y|a1,u1)
# # #             p(y|a0)= p(u0)1(a0)p(y|a0,u0)+p(u1)1(a0)p(y|a0,u1)
# #         data.EY1 = EY1 = (1-p_U)*p_YA1U0+p_U*p_YA1U1
# #         data.EY0 = EY0 = (1-p_U)*p_YA0U0+p_U*p_YA0U1
# #         data.ATE = ATE = EY1-EY0
# #         print(f'ATE : {ATE}')
#         file_object.write(f':: EY0 = {data.EY0.item():.5f}, :: EY1 = {data.EY1.item():.5f}, :: ATE = {data.ATE.item():.5f}\n')

#         file_object.write(f':: EY0 (lower <= true <= upper) : {data.EY0_l.item():.5f} <= {data.EY0.item():.5f} <= {data.EY0_u.item():.5f}\n')
#         file_object.write(f':: EY1 (lower <= true <= upper) : {data.EY1_l.item():.5f} <= {data.EY1.item():.5f} <= {data.EY1_u.item():.5f}\n')
#         file_object.write(f':: ATE (lower <= true <= upper) : {data.ATE_l.item():.5f} <= {data.ATE.item():.5f} <= {data.ATE_u.item():.5f}\n')
        
    
    
    # Logging results ends

    logger.info("Creating model...")
    
    beta = rho    
    dim = data.trn.x.shape[1]
    conditioner_type = cond_types[cond_type]
#     print(cond_type)
#     print(conditioner_type)
    conditioner_args = {"in_size": dim, "hidden": emb_net[:-1], "out_size": emb_net[-1]}
    if conditioner_type is DAGConditioner:
        conditioner_args['l1'] = l1
        conditioner_args['gumble_T'] = .5
        conditioner_args['nb_epoch_update'] = nb_step_dual
        conditioner_args["hot_encoding"] = False#True
        conditioner_args['A_prior'] = data.A.to(device)
        conditioner_args['Z_Sigma'] = data.Z_Sigma.to(device)
        
    normalizer_type = norm_types[norm_type]
#     print(norm_type)
#     print(normalizer_type)
    if normalizer_type is MonotonicNormalizer:
        normalizer_args = {"integrand_net": int_net, "cond_size": emb_net[-1], "nb_steps": nb_steps,
                           "solver": solver, "mu": data_mu, "sigma": data_sigma,
#                            "cat_dims": None}
                           "cat_dims": data.cat_dims}#, 3:2
#                            "cat_dims": {0:2, 1:2}}#, 3:2
    else:
        normalizer_args = {}

    model = buildFCNormalizingFlow_UC(nb_flow, conditioner_type, conditioner_args, normalizer_type, normalizer_args)
#     print(model)
    print(model.getConditioners()[0])
    print(model.getNormalizers()[0])
#     print(model.getConditioners()[0].soft_thresholded_A().detach().cpu())
    _best_valid_loss = np.inf

#     opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    if load:
        logger.info("Loading model...")
        model = torch.load(path + '/model%s.pt' % file_number_str, map_location={"cuda:0": device})
        model.train()
        if os.path.isfile(path + '/ADAM%s.pt'):
            opt.load_state_dict(torch.load(path + '/ADAM%s.pt' % file_number_str, map_location={"cuda:0": device}))
            if device != "cpu":
                for state in opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

    n_mce_samples = n_mce_samples#2000
    Z_do = MultivariateNormal(torch.zeros(dim), data.Z_Sigma)#torch.eye(dim))
    z_do = Z_do.sample(torch.Size([n_mce_samples]))#.to(device)
#     print(z_do.shape)

#     l_cat_dims = list(model.cat_dims.keys())
#     l_n_cats = list(model.cat_dims.values())
    l_cat_dims = list([0])# specify the location of treatment tensor
    l_n_cats = list([2])# specify the number of categories of treatment for ATE calculation.

    if len(l_cat_dims) > 1:
        all_a = torch.cartesian_prod(*[torch.arange(0, n_cat).float() for n_cat in l_n_cats])#.to(device)
    else:
        all_a = torch.cartesian_prod(*[torch.arange(0, n_cat).float() for n_cat in l_n_cats]).unsqueeze(-1)#.to(device)
    
    z_do_n = z_do.unsqueeze(1).expand(-1, all_a.shape[0], -1).clone()#.to(device)
    all_a_n=all_a.unsqueeze(0).expand(n_mce_samples,-1,-1)#.to(device)

    z_do_n[:,:,l_cat_dims] = all_a_n
    z_do_n = z_do_n.transpose_(1,0).reshape(-1,dim).to(device)#.view(-1,n_samples,dim)
    cur_x_do_n_inv = torch.zeros_like(z_do_n)
    
    d_z_do_n = TensorDataset(z_do_n)
    l_z_do_n = DataLoader(d_z_do_n, batch_size=mce_b_size,
                             num_workers=int(workers),
                             shuffle=False,
                             pin_memory=pin_memory,
                             drop_last=False) # create your dataloader

    n_iters_val = (epoch_iters//4)+1 #50 # check validation loss after 20% of epoch iterations
    n_ckpt_cycle = (epoch_iters//4)+1 #50 # check validation loss after 20% of epoch iterations
    n_ckpt_cycle_10 = n_ckpt_cycle*10


    x_val_diff_norm_best = np.inf
    x_tst_diff_norm_best = np.inf
    x_val_do_diff_norm_best = np.inf
    x_tst_do_diff_norm_best = np.inf

    model.to(device)
    nb_iters = nb_epoch*epoch_iters

    # create dataloader-iterator
    l_trn_iter = iter(l_trn) 

    n_estop = 0
    n_iters = 0
    
    for epoch in range(file_number+1, file_number+1+nb_epoch):
                
        ll_tot = 0
        start = timer()

        # Update constraints
        if conditioner_type is DAGConditioner:
            with torch.no_grad():
                for conditioner in model.getConditioners():
                    conditioner.constrainA(zero_threshold=0.)

        # Training loop
        if train:
            model.train()
            if n_estop > nb_estop:
                break
            for i, cur_x in enumerate(l_trn):
                n_iters += 1
                cur_x=cur_x[0].to(device)
                if normalizer_type is MonotonicNormalizer:
                    for normalizer in model.getNormalizers():
                        normalizer.nb_steps = nb_steps + torch.randint(0, 10, [1])[0].item()
                z, jac = model(cur_x)
                
                loss = model.loss(z, jac)
                if math.isnan(loss.item()) or math.isinf(loss.abs().item()):
                    torch.save(model, path + '/NANmodel.pt')
                    logger.info("Error NAN in loss")
                    exit()
                ll_tot += loss.detach()
                opt.zero_grad()
                loss.backward(retain_graph=True)
                opt.step()
                if n_iters % n_iters_val==0:
                    n_estop += n_iters_val/epoch_iters
                    # Valid loop
                    model.eval()
                    ll_val = 0.
                    x_val_diff_norm = 0
                    x_val_do_diff_norm = 0
                    z_val_diff_norm = 0
                    z_val_do_diff_norm = 0
                    with torch.no_grad():
                        if normalizer_type is MonotonicNormalizer:
                            for normalizer in model.getNormalizers():
                                normalizer.nb_steps = nb_steps + 20
                        for i, cur_x in enumerate(l_val):
                            cur_x=cur_x[0].to(device)
                            z, jac = model(cur_x)

                            ll = (model.z_log_density(z) + jac)
                            ll_val += ll.mean().item()
                        ll_val /= i + 1

                        end = timer()
                        dagness = max(model.DAGness())
                        logger.info("epoch: {:d} - Train loss: {:4f} - Valid log-likelihood: {:4f} - <<DAGness>>: {:4f} - Elapsed time per epoch {:4f} (seconds)".
                                    format(epoch, ll_tot.item(), ll_val, dagness, end-start))


                        if dagness < 1e-20 and -ll_val < _best_valid_loss:
                            n_estop = 0
                            _best_valid_loss = -ll_val
                            logger.info("------- New best validation loss --------")
                            logger.info("++ epoch: {:d} - Train loss: {:4f} - Valid log-likelihood: {:4f} - <<DAGness>>: {:4f} - Elapsed time per epoch {:4f} (seconds)".
                                    format(epoch, ll_tot.item(), ll_val, dagness, end-start))
                            torch.save(model, path + '/_best_model.pt')
                            torch.save(opt.state_dict(), path + '/_best_optimizer.pt')

                            x_val_diff_norm = 0
                            x_val_do_diff_norm = 0
                            z_val_diff_norm = 0
                            z_val_do_diff_norm = 0
                            for i, cur_x in enumerate(l_val):
                                cur_x=cur_x[0].to(device)
                                z, jac = model(cur_x)

                                cur_x_do_inv = model.invert(z, do_idx=[0], do_val=torch.narrow(cur_x,1,0,1))
                                z_do_inv, _ = model(cur_x_do_inv)#.to(device)
                                cur_x_inv = model.invert(z)
                                z_inv, _ = model(cur_x_inv)#.to(device)
                                x_val_diff_norm += torch.norm(cur_x - cur_x_inv)
                                x_val_do_diff_norm += torch.norm(cur_x - cur_x_do_inv)
                                z_val_diff_norm += torch.norm(z - z_inv)
                                z_val_do_diff_norm += torch.norm(z - z_do_inv)

                            if x_val_diff_norm > 1e-8 or z_val_diff_norm > 1e-8 or x_val_do_diff_norm > 1e-8 or z_val_do_diff_norm > 1e-8:
                                logger.info(f'# {len(d_trn)} e{epoch}: i{n_iters} - ||x_val_diff||={x_val_diff_norm:8.5f}, ||z_val_diff||={z_val_diff_norm:8.5f}, \ne{epoch}: i{n_iters} - ||x_val_do_diff||={x_val_do_diff_norm:8.5f}, ||z_val_do_diff||={z_val_do_diff_norm:8.5f}')
                            elif x_val_diff_norm > 1e-4 or z_val_diff_norm > 1e-4 or x_val_do_diff_norm > 1e-4 or z_val_do_diff_norm > 1e-4:
                                logger.info(f'# {len(d_trn)} x_val=\n{cur_x[:2,:]},\nx_val_inv=\n{cur_x_inv[:2,:]},\nx_val_do_inv=\n{cur_x_do_inv[:2,:]}')
                                logger.info(f'# {len(d_trn)} e{epoch}: i{n_iters} - ||x_val_diff||={x_val_diff_norm:8.5f}, ||z_val_diff||={z_val_diff_norm:8.5f}, \ne{epoch}: i{n_iters} - ||x_val_do_diff||={x_val_do_diff_norm:8.5f}, ||z_val_do_diff||={z_val_do_diff_norm:8.5f}')
                            else:
                                pass    
#                                 logger.info('full invertibility achieved!!!')

                            if x_val_diff_norm < x_val_diff_norm_best:
                                x_val_diff_norm_best = x_val_diff_norm
                                torch.save(model, path + '/_best_val_consistent_model.pt')


                            if x_val_do_diff_norm < x_val_do_diff_norm_best:
                                x_val_do_diff_norm_best = x_val_do_diff_norm
                                torch.save(model, path + '/_best_val_do_consistent_model.pt')

                            # perform counterfactual inference by doing all the treatments for the full population
                            cur_x_do_inv = model.invert(z_do_n, do_idx=l_cat_dims, do_val=torch.narrow(z_do_n,1,min(l_cat_dims),len(l_cat_dims)))
                            cur_x_do_inv = cur_x_do_inv.view(-1,n_mce_samples,dim)
                            cur_x_do_n_inv_mean=cur_x_do_inv.mean(1).cpu().numpy()
                            logger.info(f'# {len(d_trn)} ++ cur_x_do_n_inv_mean=\n{cur_x_do_n_inv_mean}')
                            lambda_ate = cur_x_do_n_inv_mean[1,-1]-cur_x_do_n_inv_mean[0,-1]
                            logger.info(f'# {len(d_trn)} ++ e{epoch}: i{n_iters} - lambda_01 = [{lambda_ate:.5f}]')

                            # Open a file with access mode 'a'
                            with open(path + f"/_best_ATE_lambda_{beta:.4f}.txt", "a") as file_object:
                                file_object.write(f'  ,  [{lambda_ate:.5f}]  # {len(d_trn)} ++ e{epoch}: i{n_iters} - lambda_ate = #-# EY1 = {cur_x_do_n_inv_mean[1,-1]:.5f} #-# EY0 = {cur_x_do_n_inv_mean[0,-1]:.5f} #-# ll_val = {ll_val:.5f}\n')

                    torch.save(model, path + '/model.pt')
                    torch.save(opt.state_dict(), path + '/ADAM.pt')


                    model.train()
            ll_tot /= i + 1
            model.step(epoch, ll_tot)
        else:
            ll_tot = 0.

    with open(path + f"/_best_ATE_lambda_{beta:.4f}.txt", "a") as file_object:
        file_object.write(f'# {len(d_trn)} xx e{epoch}: i{n_iters} - lambda_ate =  ,  [{lambda_ate:.5f}] #-# EY1 = {cur_x_do_n_inv_mean[1,-1]:.5f} #-# EY0 = {cur_x_do_n_inv_mean[0,-1]:.5f} #-# ll_val = {ll_val:.5f}\n')
                


def make_graph(adjacency_matrix, labels=None):
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
#     print(dirs)
    d = graphviz.Digraph(engine='dot')
#     print(len(adjacency_matrix))
#     print(labels)
    names = labels if labels else [f'x{i}' for i in range(len(adjacency_matrix))]
    for name in names:
        d.node(name)
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(names[from_], names[to], label=str(coef))
    return d

def str_to_dot(string):
    '''
    Converts input string from graphviz library to valid DOT graph format.
    '''
    graph = string.replace('\n', ';').replace('\t','')
    graph = graph[:9] + graph[10:-2] + graph[-1] # Removing unnecessary characters from string
    return graph


import argparse
datasets = ["real_imf_categorical", "power", "gas", "bsds300", "miniboone", "hepmass", "digits", "proteins"]

parser = argparse.ArgumentParser(description='')
parser.add_argument("-load_config", default=None, type=str)
# General Parameters
parser.add_argument("-dataset", default='real_imf_categorical', choices=datasets, help="Which problem ?")
# parser.add_argument("-dataset", default='proteins', choices=datasets, help="Which toy problem ?")
parser.add_argument("-load", default=False, action="store_true", help="Load a model ?")
parser.add_argument("-folder", default="", help="Folder")
parser.add_argument("-f_number", default=None, type=int, help="Number of heating steps.")
parser.add_argument("-test", default=False, action="store_true")
parser.add_argument("-nb_flow", type=int, default=1, help="Number of steps in the flow.")

parser.add_argument("-rho", default=0.0, type=float, help="The strength of confounding from the Gaussian copula")# rho=0 => unconfoundedness
parser.add_argument("-y_dim", default=8, type=int, help="The dimension of child poverty to be used as the outcome")# the dimension of the outcome to be used. 1:education, 2:health, 3:information, 4: malnutrition, 5: sanitization, 6: shelter, 7: water, 8: total degree.

# Optim Parameters
parser.add_argument("-weight_decay", default=1e-5, type=float, help="Weight decay value")
parser.add_argument("-learning_rate", default=3e-4, type=float, help="Weight decay value")
parser.add_argument("-nb_epoch", default=50000, type=int, help="Number of epochs")
parser.add_argument("-b_size", default=128, type=int, help="Batch size")
# 
# Conditioner Parameters
parser.add_argument("-conditioner", default='DAG', choices=['DAG', 'Coupling', 'Autoregressive'], type=str)
# parser.add_argument("-emb_net", default=[60, 60, 60, 30], nargs="+", type=int, help="NN layers of embedding")
parser.add_argument("-emb_net", default=[20, 15, 10], nargs="+", type=int, help="NN layers of embedding")
# Specific for DAG:
parser.add_argument("-nb_steps_dual", default=50, type=int, help="number of step between updating Acyclicity constraint and sparsity constraint")
parser.add_argument("-l1", default=0.5, type=float, help="Maximum weight for l1 regularization")
parser.add_argument("-gumble_T", default=0.5, type=float, help="Temperature of the gumble distribution.")

# Normalizer Parameters
parser.add_argument("-normalizer", default='monotonic', choices=['affine', 'monotonic'], type=str)
# parser.add_argument("-int_net", default=[100, 100, 100], nargs="+", type=int, help="NN hidden layers of UMNN")
parser.add_argument("-int_net", default=[15, 10, 5], nargs="+", type=int, help="NN hidden layers of UMNN")
parser.add_argument("-nb_steps", default=50, type=int, help="Number of integration steps.")
parser.add_argument("-nb_estop", default=20, type=int, help="Number of epochs for early stopping.")
parser.add_argument("-n_mce_samples", default=2000, type=int, help="Number of Monte-Carlo mean estimation samples.")
parser.add_argument("-mce_b_size", default=2000, type=int, help="Monte-Carlo mean estimation Batch size")
parser.add_argument("-solver", default="CC", type=str, help="Which integral solver to use.",
                    choices=["CC", "CCParallel"])

args = parser.parse_args()

# try:
#     sys.argv = ['']
#     print(sys.argv)
#     args = parser.parse_args()
# finally:
#     sys.argv = temp_argv
#     print(sys.argv)
                          
    
now = datetime.now()
# loader = yaml.SafeLoader
# loader.add_implicit_resolver(
#     u'tag:yaml.org,2002:float',
#     re.compile(u'''^(?:
#      [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
#     |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
#     |\\.[0-9_]+(?:[eE][-+][0-9]+)?
#     |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
#     |[-+]?\\.(?:inf|Inf|INF)
#     |\\.(?:nan|NaN|NAN))$''', re.X),
#     list(u'-+0123456789.'))
# if args.load_config is not None:
#     with open("UCIExperimentsConfigurations.yml", 'r') as stream:
#         try:
#             configs = yaml.load(stream, Loader=loader)[args.load_config]
#             for key, val in configs.items():
#                 setattr(args, key, val)
#         except yaml.YAMLError as exc:
#             print(exc)


dir_name = args.dataset if args.load_config is None else args.load_config
machinename = socket.gethostname()
path = "NeurIPS2022_rhoGNF/" + dir_name + "/" + now.strftime("%Y_%m_%d_%H_%M_%S") + '_' + machinename + '_rho_' +  f'{args.rho:.4f}' if args.folder == "" else args.folder
if not(os.path.isdir(path)):
    os.makedirs(path)
train(args.dataset, load=args.load, path=path, nb_step_dual=args.nb_steps_dual, l1=args.l1, nb_epoch=args.nb_epoch, nb_estop=args.nb_estop,
      int_net=args.int_net, emb_net=args.emb_net, b_size=args.b_size, all_args=args,
      nb_steps=args.nb_steps, file_number=args.f_number,  solver=args.solver, nb_flow=args.nb_flow,
      train=not args.test, weight_decay=args.weight_decay, learning_rate=args.learning_rate,
      cond_type=args.conditioner,  norm_type=args.normalizer,  n_mce_samples=args.n_mce_samples, mce_b_size=args.mce_b_size,
        rho = args.rho,
        y_dim = args.y_dim# the dimension of the outcome to be used. 1:education, 2:health, 3:information, 4: malnutrition, 5: sanitization, 6: shelter, 7: water, 8: total degree.
     )

# sed -i 's/#.*= / /g' *.log & sed -i 's/==>/# ==>/g' *.log &
