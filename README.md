# NeuralCollapse
This is the repo for Neural Collapse project.

# Notes
Please install autograd & pymanopt following [this page](https://www.pymanopt.org/):
```bash
python3 -m pip install autograd, pymanopt
```
When I run the solver it throws an error saying "TensorFlow" has no "Session". My TF has been upgraded to 2.x where static graph in 1.x has been replaced 
by dynamic graph as PyTorch, so there's no "Session". I changed "problem.py" where the error comes from a little forcing the library to use autograd. If anything alike occurs, maybe 
you can use the "problem.py" in the repo.
