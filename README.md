# CMSC422 - Project 3 - Perceptrons, SVMs, Neural networks
---

In this project, you will work with perceptrons, support vector machines, and neural networks. By completing this assignment, you will develop an intuition for how each algorithm works, understand some key aspects of decision boundary geometry, the connection between theory and model development, hyperparameter optimization, kernels, methods for regularization, interpreting models, and more. Finally you will apply these algorithms to canonical prediction tasks, by training, evaluating, and interpreting your own models.

As usual, we expect you to complete the notebooks in this homework
and fill in the missing implementations.

Please do not edit function names or parameters as your code will be tested
using a test harness.

I recommend that you complete notebooks in the following order:

1. Perceptrons
2. SVMs
3. Neural networks



We will write clarifications and updates below, so check this page regularly.

__Updates:__

__Nov 24 -__ For the neural network, use tanh as the activation function for the first layer, and your last layer will be a softmax layer. Softmax function is defined [here](https://en.wikipedia.org/wiki/Softmax_function), and here's a quick definition to make sure you understand it correctly: for each output node i, compute its pre-activation value z_i, take exp(z_i), and divide it by the sum of exp(z_j) for all pre-activations z_j in the output layer. This is the value we expect you to return for each output.
