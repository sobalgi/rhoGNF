# $\rho$-GNF : $\rho$-GNF: A Copula-based Sensitivity Analysis to Unobserved Confounding Using Normalizing Flows. [[arxiv]](https://arxiv.org/abs/2209.07111)

$\rho$ represents the sensitivity parameter of the Gaussian copula that represents the non-causal associaition/dependence between the Gaussian noise of the $\rho$-GNF, $Z_A$ and $Z_Y$. Since the transformations of $Z_A \rightarrow A$ and $Z_Y \rightarrow Y$ are monotonic by design, the non-causal association due to unobserved confounding modeled by the copula represents the non-causal association between $A$ and $Y$ thanks to scale-invariance property of $\rho$ to monotonically increasing transformations. 

The implementation of rho-GNF is done by extending the offical codes for the paper: `Graphical Normalizing Flows,  Antoine Wehenkel and Gilles Louppe.  (May 2020)`. [[arxiv]](https://arxiv.org/abs/2006.02548) [[github]](https://github.com/AWehenkel/Graphical-Normalizing-Flows)

This implemnetation is an adaptation of the c-GNF (Causal-Graphical Normalizing Flows) for causal effect identification and estimation for sensitivity analysis to relax the unconfoundedness assumptions. 

`Balgi, S., Peña, J. M., & Daoud, A. (2022). Personalized Public Policy Analysis in Social Sciences Using Causal-Graphical Normalizing Flows. Proceedings of the AAAI Conference on Artificial Intelligence, 36(11), 11810-11818.` [[paper]](https://doi.org/10.1609/aaai.v36i11.21437)

The $\rho_{curve}$ for a given dataset is generated by running multiple values of $\rho \in [-1,+1]$ and plotting the line with the points $(\rho,ACE_{\Phi_\rho})$.


# Dependencies
The list of dependencies can be found in requirements.txt text file and installed with the following command:
```bash
pip install -r requirements.txt
```
# Code architecture
This repository provides some code to build diverse types normalizing flow models in PyTorch. The core components are located in the **models** folder. The different flow models are described in the file **NormalizingFlow.py** and they all follow the structure of the parent **class NormalizingFlow**.
A flow step is usually designed as a combination of a **normalizer** (such as the ones described in Normalizers sub-folder) with a **conditioner** (such as the ones described in Conditioners sub-folder). Following the code hierarchy provided makes the implementation of new conditioners, normalizers or even complete flow architecture very easy.
#  $\rho$-GNF PGM2024 submission experiments
## Three different settings of datasets and experiments presented in our $\rho$-GNF PGM2024 submission
### 1. Simulated dataset with  ``Continuous variables`` with the code `python ToySimulatedContinuous.py -h`.
```bash
python ToySimulatedContinuous.py -alpha ALPHA -beta BETA -delta DELTA -rho RHO
```

For example, 
```bash
python ToySimulatedContinuous.py -alpha 0.2 -beta -0.6 -delta 0.72 -rho -0.55

```
to run the rho-GNF corresponding to the observational dataset generated from the SCM in Eq. 8.

### 2. Simulated dataset with ``Categorical variables`` with the code `python ToySimulatedBinary.py -h`.
```bash
python ToySimulatedBinary.py -rho RHO -p_U P_U -p_AU0 P_AU0 -p_AU1 P_AU1 -p_YA0U0 P_YA0U0 -p_YA0U1 P_YA0U1 -p_YA1U0 P_YA1U0 -p_YA1U1 P_YA1U1 -n_DGPs N_DGPS -n_dgp N_DGP
```

For example, 
```bash
python ToySimulatedBinary.py -rho 0.0 -n_DGPs 20 -n_dgp 0

```
to run the rho-GNF corresponding to the observational dataset with the binary treatment A, outcome Y and unobserved confounder U.

### 3. Real-world IMF child poverty dataset with ``Categorical variables`` with the code `python RealIMFCategorical.py -h`.
```bash
python RealIMFCategorical.py -rho RHO -y_dim Y_DIM
```

For example, 
```bash
python RealIMFCategorical.py -y_dim 8 -rho 0.0

```
to run the rho-GNF corresponding to the real-world observational dataset with the binary treatment A, categorical outcome Y and unobserved confounder Political-will.

y_dim represents the dimension of the outcome to be used. 1:education, 2:health, 3:information, 4: malnutrition, 5: sanitization, 6: shelter, 7: water, 8: total degree of all the 7 individual dimensions of child poverty.

If you use c-GNF or $\rho $-GNF, please cite 
  1. `Balgi, S., Peña, J. M., & Daoud, A. (2022). Personalized Public Policy Analysis in Social Sciences Using Causal-Graphical Normalizing Flows. In Proceedings of the AAAI Conference on Artificial Intelligence, 36(11), 11810-11818.` [[paper]](https://doi.org/10.1609/aaai.v36i11.21437)
and
  2. `Balgi, S., Peña, J. M., & Daoud, A. (2022). $\rho $-GNF: A Novel Sensitivity Analysis Approach Under Unobserved Confounders. In Proceedings of the 12th International Conference on Probabilistic Graphical Models (PGM 2024) - Proceedings of Machine Learning Research, (to appear).` [[arxiv]](https://arxiv.org/abs/2209.07111)
