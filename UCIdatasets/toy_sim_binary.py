import numpy as np
import torch
import os
from torch.distributions.bernoulli import Bernoulli
import pandas as pd
import graphviz
import dowhy
from dowhy import CausalModel
import networkx as nx

import UCIdatasets as datasets

# n_DGPs = 20#10000
# n_samples = 50000

# # p_U=torch.rand(torch.Size([n_DGPs,1]))
# # p_AU0=torch.rand(torch.Size([n_DGPs,1]))
# # p_AU1=torch.rand(torch.Size([n_DGPs,1]))
# # p_YA0U0=torch.rand(torch.Size([n_DGPs,1]))
# # p_YA0U1=torch.rand(torch.Size([n_DGPs,1]))
# # p_YA1U1=torch.rand(torch.Size([n_DGPs,1]))
# # p_YA1U0=torch.rand(torch.Size([n_DGPs,1]))

# p_Y1A0 = p_Y1A0 = p_YA0U0*(1-p_AU0)*(1-p_U)+p_YA0U1*(1-p_AU1)*(p_U)
# p_Y1A1 = p_Y1A1 = p_YA1U0*(p_AU0)*(1-p_U)+p_YA1U1*(p_AU1)*(p_U)
# p_A0 = p_A0 = (1-p_AU0)*(1-p_U)+(1-p_AU1)*(p_U)
# p_A1 = p_A1 = (p_AU0)*(1-p_U)+(p_AU1)*(p_U)

# EY0_l = EY0_l = p_Y1A0
# EY0_u = EY0_u = p_Y1A0+p_A1
# EY1_l = EY1_l = p_Y1A1
# EY1_u = EY1_u = p_Y1A1+p_A0
# ATE_l = ATE_l = EY1_l - EY0_u
# ATE_u = ATE_u = EY1_u - EY0_l
# EY1 = (1-p_U)*p_YA1U0+p_U*p_YA1U1
# EY0 = (1-p_U)*p_YA0U0+p_U*p_YA0U1
# ATE = EY1-EY0

# data = torch.cat((p_U,p_AU0,p_AU1,p_YA0U0,p_YA0U1,p_YA1U0,p_YA1U1, EY0, EY1, ATE, EY0_l, EY1_l, ATE_l, EY0_u, EY1_u, ATE_u),dim=1).float()
# os.system(f'mkdir -p {datasets.dataroot}toy_sim_binary')
# torch.save(data, datasets.dataroot + f'toy_sim_binary/toy_sim_binary_nDGPs_{n_DGPs}.' + 'pt')

# U = Bernoulli(p_U)
# u = U.sample(sample_shape=torch.Size([n_samples]))  # Bernoulli sampling
# u=u.squeeze_(0).float()

# p_A = u*p_AU1 + (1-u)*p_AU0
# A = Bernoulli(p_A)
# a = A.sample(sample_shape=torch.Size([1]))  # Bernoulli sampling
# a=a.squeeze_(0).float()

# p_Y = u*a*p_YA1U1 + (1-u)*a*p_YA1U0 + u*(1-a)*p_YA0U1 + (1-u)*(1-a)*p_YA0U0
# Y = Bernoulli(p_Y)
# y = Y.sample(sample_shape=torch.Size([1]))  # Bernoulli sampling
# y=y.squeeze_(0).float()

# data = torch.cat((a, y),dim=-1).bool()#.float()#n_samples x n_dgp x d

# os.system(f'mkdir -p {datasets.dataroot}toy_sim_binary')
# torch.save(data, datasets.dataroot + f'toy_sim_binary/toy_sim_binary_{n_DGPs}_{n_samples}.' + 'pt')




class TOY_SIM_BINARY:

    class Data:

        def __init__(self, data):

            self.x = data.float()#.double()#.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, n_samples=50000, new=False,
        case='Binary',# experiment name
        p_U = 0.4,
        p_AU0 = 0.4,
        p_AU1 = 0.6,
        p_YA0U0 = 0.3,
        p_YA0U1 = 0.4,
        p_YA1U0 = 0.6,
        p_YA1U1 = 0.7,
        n_dgp = 0,
        n_DGPs = 20,
        rho=0.0
        ):
        self.case = case

        if not new:
            try:
        #         p_ = torch.cat((p_U,p_AU0,p_AU1,p_YA0U0,p_YA0U1,p_YA1U0,p_YA1U1, EY0, EY1, ATE, EY0_l, EY1_l, ATE_l, EY0_u, EY1_u, ATE_u),dim=1).float()
                p_ = torch.load(datasets.dataroot + f'toy_sim_binary/toy_sim_binary_{case}_nDGPs_{n_DGPs}.' + 'pt')#[n_dgp,:]#torch.cat((a, y),dim=1).float()
                print(f'DGP Parameters loaded from the saved file '+datasets.dataroot + f'toy_sim_binary/toy_sim_binary_{case}_nDGPs_{n_DGPs}.' + 'pt')
            except:
                with torch.no_grad():

    #                 n_DGPs = 20#10000
    #                 n_samples = 50000

                    p_U=torch.rand(torch.Size([n_DGPs,1]))
                    p_AU0=torch.rand(torch.Size([n_DGPs,1]))
                    p_AU1=torch.rand(torch.Size([n_DGPs,1]))
                    p_YA0U0=torch.rand(torch.Size([n_DGPs,1]))
                    p_YA0U1=torch.rand(torch.Size([n_DGPs,1]))
                    p_YA1U1=torch.rand(torch.Size([n_DGPs,1]))
                    p_YA1U0=torch.rand(torch.Size([n_DGPs,1]))

                    p_Y1A0 = p_Y1A0 = p_YA0U0*(1-p_AU0)*(1-p_U)+p_YA0U1*(1-p_AU1)*(p_U)
                    p_Y1A1 = p_Y1A1 = p_YA1U0*(p_AU0)*(1-p_U)+p_YA1U1*(p_AU1)*(p_U)
                    p_A0 = p_A0 = (1-p_AU0)*(1-p_U)+(1-p_AU1)*(p_U)
                    p_A1 = p_A1 = (p_AU0)*(1-p_U)+(p_AU1)*(p_U)

                    EY0_l = EY0_l = p_Y1A0
                    EY0_u = EY0_u = p_Y1A0+p_A1
                    EY1_l = EY1_l = p_Y1A1
                    EY1_u = EY1_u = p_Y1A1+p_A0
                    ATE_l = ATE_l = EY1_l - EY0_u
                    ATE_u = ATE_u = EY1_u - EY0_l
                    EY1 = (1-p_U)*p_YA1U0+p_U*p_YA1U1
                    EY0 = (1-p_U)*p_YA0U0+p_U*p_YA0U1
                    ATE = EY1-EY0

                    p_ = torch.cat((p_U,p_AU0,p_AU1,p_YA0U0,p_YA0U1,p_YA1U0,p_YA1U1, EY0, EY1, ATE, EY0_l, EY1_l, ATE_l, EY0_u, EY1_u, ATE_u),dim=1).float()
                    os.system(f'mkdir -p {datasets.dataroot}toy_sim_binary')
                    torch.save(p_, datasets.dataroot + f'toy_sim_binary/toy_sim_binary_{case}_nDGPs_{n_DGPs}.' + 'pt')
                    print(f'DGP Parameters file generated !!!')

                    U = Bernoulli(p_U)
                    u = U.sample(sample_shape=torch.Size([n_samples]))  # Bernoulli sampling
                    u=u.squeeze_(0).float()

                    p_A = u*p_AU1 + (1-u)*p_AU0
                    A = Bernoulli(p_A)
                    a = A.sample(sample_shape=torch.Size([1]))  # Bernoulli sampling
                    a=a.squeeze_(0).float()

                    p_Y = u*a*p_YA1U1 + (1-u)*a*p_YA1U0 + u*(1-a)*p_YA0U1 + (1-u)*(1-a)*p_YA0U0
                    Y = Bernoulli(p_Y)
                    y = Y.sample(sample_shape=torch.Size([1]))  # Bernoulli sampling
                    y=y.squeeze_(0).float()

                    data = torch.cat((a, y),dim=-1).bool()#.float()#n_samples x n_dgp x d

                    os.system(f'mkdir -p {datasets.dataroot}toy_sim_binary')
                    torch.save(data, datasets.dataroot + f'toy_sim_binary/toy_sim_binary_{case}_{n_DGPs}_{n_samples}.' + 'pt')
                    print(f'Observational samples generated from DGP Parameters file !!!')
               
        self.n_dgp = n_dgp
 
 
        self.p_U = p_U = p_[n_dgp,0]#torch.rand(1)#from_numpy(p_A1).float()
        print(f':: p_U = {p_U}')

        self.p_AU0 = p_AU0 = p_[n_dgp,1]#torch.rand(1)#from_numpy(p_A1).float()
        self.p_AU1 = p_AU1 = p_[n_dgp,2]#torch.rand(1)#from_numpy(p_A1).float()
#             p_AU1 = 1-p_AU0#torch.rand()#from_numpy(p_A1).float()
        print(f':: p_AU0 = {p_AU0}, :: p_AU1 = {p_AU1}')

        self.p_YA0U0 = p_YA0U0 = p_[n_dgp,3]#torch.rand(1)#from_numpy(p_A1).float()
        self.p_YA0U1 = p_YA0U1 = p_[n_dgp,4]#torch.rand(1)#from_numpy(p_A1).float()
        self.p_YA1U0 = p_YA1U0 = p_[n_dgp,5]#torch.rand(1)#from_numpy(p_A1).float()
        self.p_YA1U1 = p_YA1U1 = p_[n_dgp,6]#torch.rand(1)#from_numpy(p_A1).float()
#             p_AU1 = 1-p_AU0#torch.rand()#from_numpy(p_A1).float()
        print(f':: p_YA0U0 = {p_YA0U0}, :: p_YA0U1 = {p_YA0U1}')
        print(f':: p_YA1U0 = {p_YA1U0}, :: p_YA1U1 = {p_YA1U1}')

#             p(y,a,u)=p(u)1(a)p(y|a,u)
#             p(y)=sum u p(u)1(a)p(y|a,u)
#             p(y|a1)= p(u0)1(a1)p(y|a1,u0)+p(u1)1(a1)p(y|a1,u1)
#             p(y|a0)= p(u0)1(a0)p(y|a0,u0)+p(u1)1(a0)p(y|a0,u1)

        self.EY0 = EY0 = p_[n_dgp,7]#(1-p_U)*p_YA0U0+p_U*p_YA0U1
        self.EY1 = EY1 = p_[n_dgp,8]#(1-p_U)*p_YA1U0+p_U*p_YA1U1
        self.ATE = ATE = p_[n_dgp,9]#EY1-EY0
        
        self.p_Y1A0 = p_Y1A0 = p_YA0U0*(1-p_AU0)*(1-p_U)+p_YA0U1*(1-p_AU1)*(p_U)
        self.p_Y1A1 = p_Y1A1 = p_YA1U0*(p_AU0)*(1-p_U)+p_YA1U1*(p_AU1)*(p_U)
        self.p_A0 = p_A0 = (1-p_AU0)*(1-p_U)+(1-p_AU1)*(p_U)
        self.p_A1 = p_A1 = (p_AU0)*(1-p_U)+(p_AU1)*(p_U)
        
        self.EY0_l = EY0_l = p_[n_dgp,10]#p_Y1A0
        self.EY1_l = EY1_l = p_[n_dgp,11]#p_Y1A1
        self.ATE_l = ATE_l = p_[n_dgp,12]#self.EY1_l - self.EY0_u
        self.EY0_u = EY0_u = p_[n_dgp,13]#p_Y1A0+p_A1
        self.EY1_u = EY1_u = p_[n_dgp,14]#p_Y1A1+p_A0
        self.ATE_u = ATE_u = p_[n_dgp,15]#self.EY1_u - self.EY0_l
        
        print(f'EY0 (lower <= true <= upper) : {EY0_l} <= {EY0} <= {EY0_u}')
        print(f'EY1 (lower <= true <= upper) : {EY1_l} <= {EY1} <= {EY1_u}')
        print(f'ATE (lower <= true <= upper) : {ATE_l} <= {ATE} <= {ATE_u}')
        
        
        
        # boundsAFfun <- function(p0, p1, q0, q1){
        # l0 <- q0*p0
        # u0 <- q0*p0+p1
        # l1 <- q1*p1
        # u1 <- q1*p1+p0
        # }
        # c(l0, u0, l1, u1)        
                

        data = load_data(n_samples=n_samples, new=new, case=case,
        p_U = self.p_U,
        p_AU0 = self.p_AU0,
        p_AU1 = self.p_AU1,
        p_YA0U0 = self.p_YA0U0,
        p_YA0U1 = self.p_YA0U1,
        p_YA1U0 = self.p_YA1U0,
        p_YA1U1 = self.p_YA1U1,
        n_DGPs = n_DGPs,
        n_dgp = n_dgp,
        )
        
        N = data.shape[0]

        N_test = int(0.1 * data.shape[0])
        data_test = data[-N_test:]
    #     data = data[0:-N_test]
        N_validate = int(0.1 * data.shape[0])
        data_validate = data[-N_validate-N_test:-N_test]
        data_train = data[0:-N_validate-N_test]


        self.trn = data_train
        self.val = data_validate
        self.tst = data_test

        data = np.vstack((self.trn, self.val))
        self.mu = mu = data.mean(axis=0)
        self.sig = s = data.std(axis=0)
            

        self.trn = self.Data(self.trn)
        self.val = self.Data(self.val)
        self.tst = self.Data(self.tst)

        self.n_dims = self.trn.x.shape[1]
        self.cat_dims = {0:2, 1:2}
        self.A = get_adj_matrix()
        self.Z_Sigma = get_cov_matrix(rho)# + get_adj_matrix().T/2.0
        
        self.dataset_filepath = datasets.dataroot + f'toy_sim_binary/toy_sim_binary_{case}_{n_DGPs}_{n_samples}_{n_dgp}.'
        print(f'Observational data samples from the specific DGP={n_dgp} is loaded as the dataset from '+self.dataset_filepath)

        return 
        

def get_adj_matrix():
    A = np.zeros((2, 2))

    A[1,0] = 1

    return torch.from_numpy(A).float()

def get_cov_matrix(rho=0.0):
    Z_Sigma = np.eye(2)

    Z_Sigma[0, 1] = rho
    Z_Sigma[1, 0] = rho

    return torch.from_numpy(Z_Sigma).float()


def load_data(n_samples=50000, new=False, 
    case='Binary',
    p_U = 0.4,
    p_AU0 = 0.4,
    p_AU1 = 0.6,
    p_YA0U0 = 0.3,
    p_YA0U1 = 0.4,
    p_YA1U0 = 0.6,
    p_YA1U1 = 0.7,
    n_DGPs = 10000,
    n_dgp = -1
    ):

    if not new:
        try:
            data = torch.load(datasets.dataroot + f'toy_sim_binary/toy_sim_binary_{case}_{n_DGPs}_{n_samples}.' + 'pt')[:,n_dgp,:].float()#torch.cat((a, y),dim=1).float()
            return data
        except:
            
            print(f'Parameter file not available !!! Please use argument new=True to include parameter file generation !!!')
            exit()
#             data = load_data(n_samples=n_samples, new=True
#             p_U = p_U,
#             p_AU0 = p_AU0,
#             p_AU1 = p_AU1,
#             p_YA0U0 = p_YA0U0,
#             p_YA0U1 = p_YA0U1,
#             p_YA1U0 = p_YA1U0,
#             p_YA1U1 = p_YA1U1,
#             n_dgp = n_dgp,
#             )

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
    f'''
    Converts input string from graphviz library to valid DOT graph format.
    f'''
    graph = string.replace('\n', f';').replace('\t','')
    graph = graph[:9] + graph[10:-2] + graph[-1] # Removing unnecessary characters from string
    return graph
