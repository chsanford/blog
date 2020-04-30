---
layout: post
title: "Why Is Deep Learning Deep?"
author: Clayton Sanford
tags: learning-theory depth-separation research
---
As a first year computer science PhD student, I've spent a lot of time exploring problems in learning theory, trying to figure out which research problems to work on. Learning theory as a field strives to understand mathematically why machine learning works by studying the computational and statistical abilities and limitations of ML algorithms. I am broadly interested in better understanding why deep learning works so well in practice. 

__Deep learning__ is a machine learning model that represents functions with a circuit-like structure called a __neural network__. A neural network takes an input vector from $$\mathbb{R}^d$$ and transforms that by applying several "layers" of computation to the input, which finally results in the output of the function. Each layer takes a linear combinations of the outputs of the previous layer and applies some non-linear __activation function__ to those values. The number of values (or __neurons__) computed in each layer is known as the __width__ of the network, and the number of layers is called the __depth__. The "deep" in "deep learning" refers to neural networks with many layers.

The neural network is parameterized by the the weights assigned to each neuron in each linear combination. The network learns these parameters by attempting to minimize the training error for some given sample, with the hope that good performance on that sample will indicate good performance on never-before-seen data. (This strategy is known as __Empirical Risk Minimization__ in machine learning theory.) It does so by applying gradient based optimization methods, such as stochastic gradient descent.

![](/assets/power-of-depth/nn.jpeg)

There are numerous research questions one can ask in pursuit of a more complete understanding of these problems.

* __Optimization:__ Training a neural network involves running the stochastic gradient descent (SGD) algorithm to minimize some objective function. We know that SGD will converge to a globally optimal solution when the function it optmizes is convex ("bowl-like"). However, because the objective function for deep neural networks is non-convex, there's no obvious reason we should expect neural networks trained with SGD to converge to a solution that even minimizes training error. 

  Yet they do so in practice. Why is that? There are some papers that examine how such a convergence can happen in simple settings, like two-layer neural networks. Some papers (like [this one](https://arxiv.org/abs/1909.12292) and [that one](https://arxiv.org/abs/1805.02677)) have proved that gradient descent can give us solutions that perform well, but they tend to use extremely limited settings, and their convergence rate guarantees are orders of magnitude more conservative than empirical performance on real models.

* __Generalization:__ Even if a neural network can minimize its training objective, that only guarantees to us that the network has low training error -- it doesn't necessarily tell us anything about how well the network will perform on new data. Statistical intuition tells us that it should perform very poorly, because the neural network has tons of parameters and would thus need a much larger amount of data to learn anything meaningful. This intuition is suppored by sample complexity measurements like VC-dimension and Rademacher complexity, which indicate that deep neural networks should be extremely prone to overfitting. 

  Yet their generalization performance is far better than these pessimistic bounds tell us. Are there other ways to analyze generalization to explain why networks perform so well in practice? 

  Some recent works suggest that interpolating or overfitting the data in certain ways may actually yield good results in some scenarios. Interpolation methods have been proved to work well in certain settings for [linear regression](https://arxiv.org/abs/1806.05161) and [nearest neighbor algorithms](https://arxiv.org/abs/1903.07571), where they end up yielding functions that are smoothed implicitly by other factors. It's possible that neural networks are in this same regime, and that we can dodge the issues posed by classical statistical intuition.

* __Network Design:__ How should neural networks should be designed to optimize performance? How many layers should one use? Which activation function works best? Practitioners have developed intuition for how to answer those questions, but few of those solutions are backed up by rigorous mathematical arguments about why they work.

There's a huge gap in complexity between the networks analyzed by theorists and the networks implemented in practice for comptuer vision or natural language processing models. Neural networks for these applications may have hundreds of layers. On the other hand, most theoretical work is limited to the study of networks of depth 2 (which means a network with only one hidden layer) with bounded weights. 

A goal of my research so far is to better understand what powers depth grants neural networks. For now, we're focusing on those problems through from the approximability angle. Along these lines, one might ask the following question: 
>What kinds of mathematical functions can 3-layer networks closely approximate that 2-layer networks cannot?

Or, more generally: 
>What kinds of mathematical functions can $$(k+1)$$-layer networks closely approximate that $$k$$-layer networks cannot?

![](/assets/power-of-depth/two-vs-three.jpeg)


There's key problem with this question. We already know from [a famous 1989 paper by Hornik, Stinchcombe, and White](https://www.sciencedirect.com/science/article/pii/0893608089900208) that depth-2 networks are already universal approximators! For any continuous function from $$\mathbb{R}^d$$ to $$\mathbb{R}$$ and for any $$\epsilon > 0$$, there exists some 2-layer neural network $$g$$ such that $$g$$ $$\epsilon$$-approximates $$f$$. Therefore, we're not going to be able to prove that there are any reasonable functions that depth-3 networks can represent that depth-2 networks cannot.

So instead, we'll refine our question to ask about how whether there are functions that are "easy to represent" with depth-3 networks and "hard to represent" with depth-2 networks: 
>For which functions $$f$$ does there exist a depth-3 approximation **of polynomial width in d** such that all depth-2 approximations require **exponential width in d**?

Answers to this question are known as _depth-separation bounds_.

![](/assets/power-of-depth/two-vs-three-width.jpeg)

There are a few answers to this question, but all of them have certain limitations.
* A [2016 paper by Telgarsky](http://proceedings.mlr.press/v49/telgarsky16.html) shows that for any positive integer $$k$$, there exists a function $$f$$ that can be approximated by networks of depth $$\Theta(k^3)$$ with polynomial width, but which require exponential width in $$k$$ to be approximated by networks of depth $$\Theta(k)$$. 

  These functions are only of one variable and are very "spiky" because they're created by applying a triangle map $$\Theta(k^3)$$ times, which means the $$f(x)$$ has $$2^{O(k^3)}$$ oscillations when $$0 \leq x \leq 1$$. They're hard to approximate by less deep networks any function $$g$$ represented by a ReLU networks of depth $$\Theta(k)$$ and width $$\text{poly}(k)$$ is incapable of oscillating so many times on that interval. 

  ![](/assets/power-of-depth/telgarsky.jpeg) 

  However, this steepness of $$f$$ limits the usefulness of the result; $$f$$ has a very high Lipschitz constant (a number $$\ell$$ such that for all $$x, x' \in \mathbb{R}$$, $$\lvert f(x) - f(x')\rvert \leq \ell \lvert x - y\rvert$$). Many learning theory guarantees require that $$\ell$$ be small, and learning algorithms rarely resturn functions that are so spiky.

* Papers by [Eldan and Shamir (2016)](https://arxiv.org/abs/1512.03965) and [Daniely (2017)](https://arxiv.org/abs/1702.08489) both showed depth separation bounds between depth-2 and depth-3 networks, for $$d$$-dimensional radial functions and inner-product functions respectively. The image below is a plot of the function used by Daniely to obtain 2-vs-3 depth-separation bounds [Source: [SES19](https://arxiv.org/abs/1904.06984)]. 

  ![](/assets/power-of-depth/ses.png)

  Again, however, their functions not ideal with high Lipschitz constants. A [2019 follow-up by Safran, Eldan, and Shamir](https://arxiv.org/abs/1904.06984) shows that such an improvement in this setting may be impossible, because all Lipschitz-1 radial functions can be approximated by a depth-2 network with width polynomial in $$d$$. This means that either (1) there are no 1-Lipschitz functions that exhibit 2-vs-3 depth seprations, or (2) there are 1-Lipschitz functions with depth-separation, but they're not radial.

* A [2013 paper by Martens, Chattopadhyay, Pitassi, and Zemel](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.644.7315) succeeds in showing a depth-separation result for a 1-Lipschitz function! The problem is that the function is the discrete inner product, $$f(x, y) = (x_1 \wedge y_1) \oplus \dots \oplus (x_d \wedge y_d)$$, whose input is over the Boolean cube $$\{-1, +1\}^{2d}$$, rather than $$\mathbb{R}^d$$. While this result is theoretically illuminating, its discrete domain isn't great for studying neural networks; methods like gradient descent are optimization methods for continuous spaces, which means it's harder to think about how one would optimize a non-continuous neural network.

An aspirational goal for work in this area, would then be to answer the following updated question: 
>For which **1-Lipschitz** functions $$f$$ does there exist a depth-3 approximation of polynomial width in d such that all depth-2 approximations require exponential width in d?

Over the next few weeks, I'm going to write blog posts discussing work in this field in a little more detail. Admittedly, this is a partially selfish endeavor; it'll be good to familiarize myself with previous works on these problems by summarizing their results in my own words. But I also hope that these posts will help other people understand what kinds of results are out there on the approximation powers of neural networks which techniques are used to obtain those results.

**What did you think about the post? I'd appreciate any questions, thoughts, or feedback -- feel free to email me.**