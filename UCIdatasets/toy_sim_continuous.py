import numpy as np
import torch
import os
from torch.distributions.multivariate_normal import MultivariateNormal
import pandas as pd
import graphviz
import dowhy
from dowhy import CausalModel
import networkx as nx

import UCIdatasets as datasets


class TOY_SIM_CONTINUOUS:

    class Data:

        def __init__(self, data):

            self.x = data.float()#.double()#.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, n_samples=50000, new=False,
        case='Hoover',#experimental setting name
        alpha = 0.2,
        beta = -0.6,
        delta = 0.72,
        rho = 0.0,#unconfounded
        ):
        self.case = case

        self.trn, self.val, self.tst = load_data_split_with_noise(n_samples=n_samples, new=new, 
        case=case, 
        alpha = alpha,
        beta = beta,
        delta = delta,
        )

        data = np.vstack((self.trn, self.val))
        self.mu = mu = data.mean(axis=0)
        self.sig = s = data.std(axis=0)
            

        self.trn = self.Data(self.trn)
        self.val = self.Data(self.val)
        self.tst = self.Data(self.tst)

        self.n_dims = self.trn.x.shape[1]
        self.n_samples = n_samples# = self.trn.x.shape[0]
        self.cat_dims = {}
        self.A = get_adj_matrix()
        self.Z_Sigma = get_cov_matrix(rho)
        
        self.ATE = ate = alpha
        self.alpha = alpha
        self.delta = delta
        self.rho = rho
        
        self.dataset_filepath = datasets.dataroot + f'toy_sim_continuous/toy_sim_continuous_{case}_{n_samples}_{alpha}_{beta}_{delta}.'


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
    case='Hoover',
    alpha = 0.2,
    beta = -0.6,
    delta = 0.72,
    ):

    if not new:
        try:
            data = torch.load(datasets.dataroot + f'toy_sim_continuous/toy_sim_continuous_{case}_{n_samples}_{alpha}_{beta}_{delta}.' + 'pt')
        except:
            data = load_data(n_samples=n_samples, new=True, case=case,
            alpha = alpha,
            beta = beta,
            delta = delta,
            )
    else:
        with torch.no_grad():
                             
#             cov_mat = torch.Tensor([[1,-0.6],[-0.6,0.72]])#+0.2,-0.71,-0.55
#             cov_mat = torch.Tensor([[1,-0.4],[-0.4,0.52]])#0.0,-0.55,-0.55
#             cov_mat = torch.Tensor([[1,-0.2],[-0.2,0.40]])#-0.2,-0.32,-0.55
#             cov_mat = torch.Tensor([[1,0.2],[0.2,0.40]])#+0.2,0.32,0.55
#             cov_mat = torch.Tensor([[1,0.4],[0.4,0.52]])#0.0,0.55,0.55
#             cov_mat = torch.Tensor([[1,0.6],[0.6,0.72]])#-0.2,0.71,0.55
            cov_mat = torch.Tensor([[1,beta],[beta,delta]])#alpha,rho,rho_P

            Eay = MultivariateNormal(torch.zeros(2), cov_mat)
            eay = Eay.sample(torch.Size([n_samples]))  # normally distributed with mean=`[0,0]` and covariance_matrix=`I`
            ea = eay[:,0].unsqueeze(-1)
            ey = eay[:,1].unsqueeze(-1)
            a = ea
            y = alpha*a + ey
                               
        data = torch.cat((a, y),dim=1).float()
        
        ate = alpha
        
        n_samples = data.shape[0]
        
        os.system(f'mkdir -p {datasets.dataroot}toy_sim_continuous')
        torch.save(data, datasets.dataroot + f'toy_sim_continuous/toy_sim_continuous_{case}_{n_samples}_{alpha}_{beta}_{delta}.' + 'pt')
    

        df = pd.DataFrame()
        df['A']= pd.Series(a.squeeze(dim=1).numpy())
        df['Y'] = pd.Series(y.squeeze(dim=1).numpy())

        # write a pandas dataframe to gzipped CSV file
        df.to_csv(datasets.dataroot + f'toy_sim_continuous/toy_sim_continuous_{case}_{n_samples}_{alpha}_{beta}_{delta}.' + 'csv.gz', 
        index=False, 
        compression="gzip"
        )   

        graph = make_graph(adjacency_matrix=get_adj_matrix().numpy(), labels=list(df.columns))
    #     print(graph)
    #     graph_dot = str_to_dot(graph)
    #     nx.drawing.nx_pydot.write_dot(graph, datasets.dataroot + f'toy_sim_continuous/toy_sim_continuous.dot')
        graph.save(filename='toy_sim_continuous.dot', directory=datasets.dataroot + f'toy_sim_continuous/')

        print(f'graph saved to {datasets.dataroot}toy_sim_continuous/toy_sim_continuous.dot')



        model=CausalModel(
        data = df,
        treatment=['A'],
        outcome=['Y'],
        graph=datasets.dataroot + f'toy_sim_continuous/toy_sim_continuous.dot',
        )


    #     model.view_model()
    #     from IPython.display import Image, display
        out_filename = datasets.dataroot + f'toy_sim_continuous/toy_sim_continuous_causal_graph.png'#"causal_model.png"
        try:
            import pygraphviz as pgv
            agraph = nx.drawing.nx_agraph.to_agraph(model._graph._graph)
            agraph.draw(out_filename, format="png", prog="dot")
        except:
            model._graph.logger.warning("Warning: Pygraphviz cannot be loaded. Check that graphviz and pygraphviz are installed.")
            model._graph.logger.info("Using Matplotlib for plotting")
            import matplotlib.pyplot as plt

            solid_edges = [(n1,n2) for n1,n2, e in model._graph._graph.edges(data=True) if f'style' not in e ]
            dashed_edges =[(n1,n2) for n1,n2, e in model._graph._graph.edges(data=True) if ('style' in e and e['style']=="dashed") ]
            plt.clf()

            pos = nx.layout.shell_layout(model._graph._graph)
            nx.draw_networkx_nodes(model._graph._graph, pos, node_color='yellow',node_size=400 )
            nx.draw_networkx_edges(
            model._graph._graph,
            pos,
            edgelist=solid_edges,
            arrowstyle="-|>",
            arrowsize=12
            )
            nx.draw_networkx_edges(
            model._graph._graph,
            pos,
            edgelist=dashed_edges,
            arrowstyle="-|>",
            style="dashed",
            arrowsize=12
            )

            labels = nx.draw_networkx_labels(model._graph._graph, pos)

            plt.axis('off')
            plt.savefig(out_filename)
            plt.draw()
    
    return data

def make_graph(adjacency_matrix, labels=None):
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
    d = graphviz.Digraph(engine='dot')
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

def load_data_split_with_noise(n_samples=2500, new=False, 
    case='Hoover',
    alpha = 0.2,
    beta = -0.6,
    delta = 0.72,
    ):
    rng = np.random.RandomState(42)

    data = load_data(n_samples=n_samples, new=new, case=case, 
    alpha = alpha,
    beta = beta,
    delta = delta,
    )
#     rng.shuffle(data)
    N = data.shape[0]

#     N_test = int(0.1 * data.shape[0])
#     data_test = data[-N_test:]
    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
#     data = data[0:-N_test]
    N_validate = int(0.1 * data.shape[0])
#     data_validate = data[-N_validate-N_test:-N_test]
    data_validate = data[-N_validate:]
    data_test = data_validate
    data_train = data[0:-N_validate]


    return data_train, data_validate, data_test
