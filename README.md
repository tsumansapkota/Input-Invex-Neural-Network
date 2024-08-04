# Input-Invex-Neural-Network
[Paper](https://arxiv.org/abs/2106.08748)


The experiments and plots were conducted with the code provided.

#### Experiment: K-Lipschitz
The experiment comparing the capacity of various K-Lipschitz constraints method was conducted by file:
`00.0_2D_GP_vs_LP_vs_SN_vs_GCLP-Benchmark-All.ipynb`

#### Visualizations: Constructing Invex Function

*Category of Functions:* `02.0_Plots_from_data.ipynb`
*Invex Function Visual Proof:* `03.0_Invex_Visual_Proof.ipynb`
*GC-GP Function visualization:* `04.0_Penalty and clipping function.ipynb`

#### Experiment: N -> 1 Invex
The experiment comparing the capacity of Invex function alongside Linear, Convex and Ordinary Neural Networks was conducted by:

*2D Classification:* `01.0_2D_cls_Logistic_vs_Convex_vs_Invex_vs_NN_Benchmark.ipynb`

*2D Regression:* `01.1_2D_reg_Linear_vs_Convex_vs_Invex_vs_NN_Benchmark.ipynb`

*MNIST : MLP* `05.0_Logistic_vs_Convex_vs_Invex_vs_NN_Mnist-Benchmark.ipynb`

*MNIST : CNN* `05.1_Logistic_vs_Convex_vs_Invex_vs_NN_MnistCNN-Benchmark.ipynb`

*F-MNIST : CNN* `05.2_Convex_vs_Invex_vs_NN_FMnistCNN-Benchmark.ipynb`

#### Experiment: Energy minimization PICNN and PIINN

*1D-partial input (in/con)-vex neural network:* `06.0_Energy_minimization_PIINN_vs_PICNN_diagrams.ipynb`

#### Experiment: Multi-Connected-Set Classification

*Gaussian Mixture Model disconnected decision boundary:* `07.0_GMM_decision_boundary_viz.ipynb`

*Convex vs Connected set for regions in 2D classification:* `08.0_Toy_MultiConnected_set_classification_benchmark.ipynb`

*Invertible + Connected Classifier for Network Morphism in 2D classification:* `08.1_Toy_MultiConnected_set_classification_add_remove_centers.ipynb`

*2D Invex embedding visualization of F-MNIST:* `09.0_2D_Multi_Invex_vs_Ordinary_fMNIST.ipynb`

*2D Invex embedding visualization of CIFAR-10:* `09.1_2D_Multi_Invex_vs_Ordinary_Cifar.ipynb`

*Multi-Invex classifier region interpretation F-MNIST:* `10.0_Multi_Invex_Visualize_Centers_(BNvsAN)_fMNIST.ipynb` 

*Multi-Invex classifier region interpretation CIFAR-10:* `10.1_Multi_Invex_Visualize_Centers_(BNvsAN)_Cifar.ipynb` 

*MNIST/FMNIST/CIFAR-10/CIFAR-100 invertible/ordinary + Connected-clf/MLP-clf experiment* `python bench_script.py`

#### Experiment: Connected Uncertainity

*Uncertainity representation with local classifier:* `11.0_Multi_Invex_MSE_classifier_2D_with_Uncertainity_(NotQuiteWorking).ipynb`