import numpy as np
import torch
import os
import pandas as pd
import graphviz
import dowhy
from dowhy import CausalModel
import networkx as nx

import UCIdatasets as datasets


class REAL_IMF_CATEGORICAL:

    class Data:

        def __init__(self, data):

            self.x = data.float()#.double()#.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self,
            rho = 0.0,
            y_dim=8# the dimension of the outcome to be used. 1:education, 2:health, 3:information, 4: malnutrition, 5: sanitization, 6: shelter, 7: water, 8: total degree.
        ):

#         import pickle
#         with open(datasets.dataroot + 'IMF_CP/IMF_CP_d32.' + 'pkl',"rb") as f:
#             pickle_objects = pickle.load(f)

#         print('Old dataset loaded normally from saved file.')

#         df_imf = pickle_objects['df_imf']
#         df_imf_train = pickle_objects['df_imf_train']
#         df_imf_val = pickle_objects['df_imf_val']
#         df_imf_test = pickle_objects['df_imf_test']

        n_samples = 1941734#imf_data.shape[0]
#         print(imf_data.shape[0])
#         print(imf_data.shape[1])

        imf_data = torch.load(datasets.dataroot + f'real_imf_categorical/real_imf_categorical_{n_samples}_A_Y1_Y7.' + 'pt')
        print(f'IMF dataset with only treatment and binary outcomes of 7 individual dimensions of child poverty loaded from file '+datasets.dataroot + f'real_imf_categorical/real_imf_categorical_{n_samples}_A_Y1_Y7.' + 'pt')


#         imf_data = torch.from_numpy(df_imf.to_numpy())
# #         imf_data = imf_data[:,[14].append(range(8))]
#         imf_data = imf_data[:,[14,1,2,3,4,5,6,7]]
#         imf_data = imf_data.to(torch.bool)

        
#         os.system(f'mkdir -p {datasets.dataroot}real_imf_categorical')
#         torch.save(imf_data, datasets.dataroot + f'real_imf_categorical/real_imf_categorical_{n_samples}_y8.' + 'pt')

        
#         self.trn, self.val, self.tst = df_imf_train.to_numpy(), df_imf_val.to_numpy(), df_imf_test.to_numpy()
#         self.trn, self.val, self.tst = torch.from_numpy(df_imf_train.to_numpy()), torch.from_numpy(df_imf_val.to_numpy()), torch.from_numpy(df_imf_test.to_numpy())
        
    
        data_dict = torch.load(datasets.dataroot + f'real_imf_categorical/real_imf_categorical_{n_samples}_A_Y1_Y7_splits.' + 'pt')
        print(f'IMF dataset predefined train, validation, test splits with only treatment and binary outcomes of 7 individual dimensions of child poverty loaded from file '+datasets.dataroot + f'real_imf_categorical/real_imf_categorical_{n_samples}_A_Y1_Y7_splits.' + 'pt')
    
#         print(self.trn.shape)
        self.trn = data_dict['trn']#self.trn[:,[14,1,2,3,4,5,6,7]].to(torch.bool)
        self.val = data_dict['val']#self.val[:,[14,1,2,3,4,5,6,7]].to(torch.bool)
        self.tst = data_dict['tst']#self.tst[:,[14,1,2,3,4,5,6,7]].to(torch.bool)
        
#         os.system(f'mkdir -p {datasets.dataroot}real_imf_categorical')
#         torch.save(imf_data, datasets.dataroot + f'real_imf_categorical/real_imf_categorical_{n_samples}_y8.' + 'pt')
        
        if y_dim==8:
            self.cat_dims = {0:2, 1:8}
            self.trn = torch.cat((self.trn[:,0].unsqueeze(-1),self.trn[:,1:].sum(-1).unsqueeze(-1)),dim=1)
            self.val = torch.cat((self.val[:,0].unsqueeze(-1),self.val[:,1:].sum(-1).unsqueeze(-1)),dim=1)
            self.tst = torch.cat((self.tst[:,0].unsqueeze(-1),self.tst[:,1:].sum(-1).unsqueeze(-1)),dim=1)
            
        elif y_dim==9:
            self.cat_dims = {0:2, 1:2}
            self.trn = torch.cat((self.trn[:,0].unsqueeze(-1),self.trn[:,1:].sum(-1).unsqueeze(-1)),dim=1)
            self.val = torch.cat((self.val[:,0].unsqueeze(-1),self.val[:,1:].sum(-1).unsqueeze(-1)),dim=1)
            self.tst = torch.cat((self.tst[:,0].unsqueeze(-1),self.tst[:,1:].sum(-1).unsqueeze(-1)),dim=1)
            
            self.trn[:,1] = torch.where(self.trn[:,1]<=1.0, 0.0, 1.0)
            self.val[:,1] = torch.where(self.val[:,1]<=1.0, 0.0, 1.0)
            self.tst[:,1] = torch.where(self.tst[:,1]<=1.0, 0.0, 1.0)
            
        else:
            self.cat_dims = {0:2, 1:2}
            self.trn = self.trn[:,[0,y_dim]]
            self.val = self.val[:,[0,y_dim]]
            self.tst = self.tst[:,[0,y_dim]]

        data = np.vstack((self.trn, self.val))
        self.mu = mu = data.mean(axis=0)
        self.sig = s = data.std(axis=0)


        self.trn = self.Data(self.trn)
        self.val = self.Data(self.val)
        self.tst = self.Data(self.tst)
#         self.trn = self.Data(torch.from_numpy(self.trn))
#         self.val = self.Data(torch.from_numpy(self.val))
#         self.tst = self.Data(torch.from_numpy(self.tst))
                
        
        self.n_dims = 2
    
#         self.p_Y1A0 = p_Y1A0 = p_YA0U0*(1-p_AU0)*(1-p_U)+p_YA0U1*(1-p_AU1)*(p_U)
#         self.p_Y1A1 = p_Y1A1 = p_YA1U0*(p_AU0)*(1-p_U)+p_YA1U1*(p_AU1)*(p_U)
#         self.p_A0 = p_A0 = (1-p_AU0)*(1-p_U)+(1-p_AU1)*(p_U)
#         self.p_A1 = p_A1 = (p_AU0)*(1-p_U)+(p_AU1)*(p_U)

#         self.EY0_l = EY0_l = p_Y1A0
#         self.EY0_u = EY0_u = p_Y1A0+p_A1
#         self.EY1_l = EY1_l = p_Y1A1
#         self.EY1_u = EY1_u = p_Y1A1+p_A0
#         self.ATE_l = ATE_l = EY1_l - EY0_u
#         self.ATE_u = ATE_u = EY1_u - EY0_l
#         # EY1 = (1-p_U)*p_YA1U0+p_U*p_YA1U1
#         # EY0 = (1-p_U)*p_YA0U0+p_U*p_YA0U1
        self.ATE = ATE = -1.2#EY1-EY0


        self.A = get_adj_matrix()
        self.Z_Sigma = get_cov_matrix(rho)
        
        self.dataset_filepath = datasets.dataroot + f'real_imf_categorical/real_imf_categorical_y{y_dim}.'
        return


def get_adj_matrix():
    A = np.zeros((2, 2))

    A[1,0] = 1 # A->Y

    return torch.from_numpy(A).float()

def get_cov_matrix(rho=0.0):
    Z_Sigma = np.eye(2)


    Z_Sigma[0, 1] = rho
    Z_Sigma[1, 0] = rho

    return torch.from_numpy(Z_Sigma).float()
