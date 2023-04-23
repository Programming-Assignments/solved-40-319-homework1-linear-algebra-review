Download Link: https://assignmentchef.com/product/solved-40-319-homework1-linear-algebra-review
<br>
<h1>0.    Linear Algebra Review</h1>

Let <em>θ </em>:= (<em>θ</em><sub>1</sub><em>,…,θ<sub>d</sub></em>) ∈ R<em><sup>d </sup></em>be a vector, and <em>θ</em><sub>0 </sub>∈ R be a scalar. Let the hyperplane H be the set of all points <em>x </em>:= (<em>x</em><sub>1</sub><em>,…,x<sub>d</sub></em>) <sup>∈ </sup>R<em><sup>d </sup></em>such that 0 = <em>θ</em><sup>&gt;</sup><em>x </em>+ <em>θ</em><sub>0</sub>, where

<em>θ</em><sup>&gt;</sup><em>x </em>= <em>θ</em><sub>1</sub><em>x</em><sub>1 </sub>+ ··· + <em>θ<sub>d</sub>x<sub>d</sub></em>

is the dot product. The goal is to find the shortest distance between H and a point <em>y </em>∈ R<em><sup>d</sup></em>. There are many ways to solve this problem, but we will be using Lagrange multipliers to familiarize ourselves with this powerful method.

Let ˜<em>x </em>be the point on H that is closest to <em>y</em>. Then ˜<em>x </em>solves the optimization problem

minimize (<em>x </em>− <em>y</em>)<sup>&gt;</sup>(<em>x </em>− <em>y</em>) <em>x</em>∈R<em><sup>d </sup></em>subject to <em>θ</em><sup>&gt;</sup><em>x </em>+ <em>θ</em><sub>0 </sub>= 0<em>.</em>

The Lagrangian for this optimization problem is

<em>L</em>(<em>x,λ</em>) = (<em>x </em>− <em>y</em>)<sup>&gt;</sup>(<em>x </em>− <em>y</em>) + <em>λ</em>(<em>θ</em><sup>&gt;</sup><em>x </em>+ <em>θ</em><sub>0</sub>)

where <em>λ </em>is the Lagrange multiplier.

1.1.               Write down the derivatives of <em>L</em>(<em>x,λ</em>) with respect to <em>x</em><sub>1</sub><em>,…,x<sub>d </sub></em>and <em>λ</em>.

1.2.        Equate the derivatives to zero, and solve the equations to find ˜<em>x</em>.

1.3.          Use ˜<em>x </em>to find the distance of <em>y </em>to the hyperplane H.

<h1>1.    Probability Review</h1>

Let <em>X </em>and <em>Y </em>be independent Poisson random variables, i.e.

<em>,         </em>for all <em>x,y </em>≥ 0<em>.</em>

for some rates <em>α,β &gt; </em>0. Let the random variable <em>Z </em>= <em>X </em>+ <em>Y </em>be their sum.

2.1.               Write P(<em>Z </em>= <em>z</em>) as a sum of products of P(<em>X </em>= <em>x</em>) and P(<em>Y </em>= <em>y</em>).

2.2.          Show that <em>Z </em>is also Poisson, and find its rate <em>γ</em>.

40.319 STATISTICAL AND MACHINE LEARNING SPRING 2021 HOMEWORK 1 3 3. Linear Regression [20 Points]

We will use PyTorch to perform linear regression using gradient descent. Import the Boston housing data from the following link.

<a href="https://www.dropbox.com/s/kkeu8nvto35n0dt/boston.csv?dl=1">https://www.dropbox.com/s/kkeu8nvto35n0dt/boston.csv?dl=1</a>

We will train a linear model that predicts the prices of houses MEDV using three inputs:

(i)        average number of rooms per dwelling RM; (ii)                index of accessibility to radial highways RAD; (iii)         per capita crime rate by town CRIM.

You can access the selected inputs and target variables using the following code:

import matplotlib.pyplot as plt import numpy csv = ’boston.csv’

data = numpy.genfromtxt(csv,delimiter=’,’)

The data contains 506 observations on housing prices in suburban Boston. The first three columns are the inputs RM, RAD and CRIM. The last column is the target MEDV.

Convert the data to PyTorch tensors using the following code.

import torch inputs = data[:, [0,1,2]] inputs = inputs.astype(numpy.float32) inputs = torch.from_numpy(inputs) target = data[:,3] target = target.astype(numpy.float32) target = torch.from_numpy(target)

3.1. Write the code to generate (random) weights <em>w</em><sub>RM</sub><em>,w</em><sub>RAD</sub><em>,w</em><sub>CRIM </sub>and bias <em>b</em>. After that, write a function to compute the linear model.

3.2.         Write a function that computes the mean squared error (MSE).

3.3. Complete the loop below to update the weights and bias using a fixed learning rate (try different values from 0.01 to 0.0001) over 200 iterations/epochs.

4                                                                                                       DUE 14 FEB. TOTAL 40 POINTS.

for i in range(200): print(“Epoch”, i, “:”)

# compute the model predictions # compute the loss and its gradient print(“Loss=”, loss) with torch.no_grad():

# update the weights # update the bias

w.grad.zero_()

b.grad.zero_()

(We use w.grad.zero () and b.grad.zero () to reset the gradients to zero because PyTorch accumulates gradients.)

3.4. Use the matplotlib library to plot the MSE against the number of iterations. Print the output to the PDF file that you are submitting on Gradescope.

For this problem, DO NOT use the in-built functions for the loss or the linear model in the torch library. Upload the final script as a file named [student-id].py using the Dropbox link at the start of this assignment.