# NeuralCollapse
This is the repo for Neural Collapse project.

<p align="center">
  <img src="https://github.com/10258392511/NeuralCollapse/blob/main/cifar100_params/epoch_30_K_5_True_M_2_True_dymanics.gif" alt="dynamics">
 </p>
 
# Notes
Please install autograd & pymanopt following [this page](https://www.pymanopt.org/):
```bash
python3 -m pip install autograd, pymanopt
```
When I run the solver it throws an error saying "TensorFlow" has no "Session". My TF has been upgraded to 2.x where static graph in 1.x has been replaced 
by dynamic graph as PyTorch, so there's no "Session". I changed "problem.py" where the error comes from a little forcing the library to use autograd. If anything alike occurs, maybe 
you can use the "problem.py" in the repo.

# CIFAR100 Colab Notebook [here](https://drive.google.com/file/d/1DHzJKXrbV7E07r9E3SSDTa4CoehdGQ19/view?usp=sharing)
