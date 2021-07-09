---
layout: post
title: "[OPML#0] A series of posts on over-parameterized machine learning models"
author: Clayton Sanford
tags: candidacy technical learning-theory over-parameterized
---

_Hello, and welcome to the blog! I've been wanting to start this for awhile, and I've finally jumped in. This is the introduction to a series of blog posts I'll be writing over the course of the summer and (possibly) early in the fall. I hope these posts are informative, and I welcome any feedback on their technical content and writing quality._

I've recently finished the second year of my computer science PhD program, in which I study the overlap between theoretical comptuer science and machine learning.
Over the next few months, I'm going to read a lot of papers about machine learning models which have a much larger number of parameters than the number of samples.
I'll write summaries of them on the blog, with the goal of making this line of work accessible to people in and out of my research community.

## Why are you doing this?

While reading papers is a lot of fun (sometimes), I kind of am required to read this set.
CS PhD students at Columbia must take a [candidacy exam](https://www.cs.columbia.edu/education/phd/requirements/candidacy/){:target="_blank"} sometime in their third (or occasionally fourth) year, which requires the student to read 20-25 papers in their subfield in order to better understand the what is known and what questions are asked in their research landscape.
It culminates with an oral examination, where several faculty members question the student on the papers and future research directions in the subfield. 

I'm starting the process of reading the papers now, and I figured that it wouldn't be such a bad idea to write about what I learn, so that's what this is going to be.

## Why this research area?
>Also, what even is an "over-parameterized machine learning model?"

The core motivation of all of my graduate research is to understand why deep learning works so well in practice.
For the uninitiated, deep learning is a family of machine learning models that uses complex hierarchical neural networks or circuits to represent complicated functions.
In the past decade, deep learning has been applied to tasks like object recognition, language translation, and game playing with wild degrees of success.
However, the theory of deep learning has lagged far behind these practical successes, which means that we can't answer simple questions like the following in a mathematically precise way: 
"How well do we expect this trained model to perform on new data?"
"What are the most important factors in determining how a sample is classified?"
"How will changing the size of the model affect model performance?"

ML theory researchers have formulated those questions mathematically and produced impressive results about performance guarantees for broad categories of ML models.
However, these don't apply very well to deep learning.
In order to get a mathematical understanding of the kinds of questions these researchers ask, I define some terminology about ML models, which I refer back to when describing why theoretical approaches tend to fall short in for deep neural networks.

As a motivating example, consider a prototypical toy machine learning problem: training a classifer that distinguishes images of cats from images of dogs.
* One does so by "learning" a function $$f_\theta: \mathbb{R}^d \to \mathbb{R}$$ that takes as input the values of pixels of a photo $$x$$ (which we can think of as a $$d$$-dimensional vector) and has the goal of returning $$f_\theta(x) = 1$$ if the photo contains a dog and $$f_\theta(x) = -1$$ if the photo contains a cat.
* In particular, if we assume that the pixels $$x$$ and labels $$y$$ are drawn from some probability distribution $$\mathcal{D}$$, then our goal is to find some parameter vector $$\theta \in \mathbb{R}^p$$ such that the _population error_ $$\mathbb{E}_{(x, y) \sim \mathcal{D}}[\ell(f_\theta(x), y)]$$ is small, where $$\ell: \mathbb{R} \times \mathbb{R} \to \mathbb{R}_+$$ is some _loss function_ that we want to minimize.
(For instance, the _squared loss_ is $$\ell(\hat{y}, y) = (\hat{y} - y)^2$$.)
* $$f_\theta$$ is parameterized by the $$p$$-dimensional vector $$\theta$$, and _training_ the model is the process of choosing a value of $$\theta$$ that we expect to perform well and have small population error.
In the case of deep learning, $$f_\theta$$ is the function produced by computing the output of a neural network with connection weights determined by $$\theta$$.
In simpler models like linear regression, $$\theta$$ directly represents the weights of a linear combination of the inputs: $$f_\theta(x) = \theta^T x$$.
* This training process occurs by observing $$n$$ training samples $$(x_1, y_1), \dots, (x_n, y_n)$$ and choosing a set of parameters $$\theta$$ such that $$f_\theta(x_i) \approx y_i$$ for all $$i= 1, \dots, n$$. 
That is, we find a vector $$\theta$$ that yields a small _training error_ $$\frac{1}{n} \sum_{i=1}^n \ell(f_\theta(x_i, y_i))$$.
The hope is that the learning rule will _generalize_, meaning that a small training error will lead to a small population error.

This is obviously an over-simplified model that excludes broad categories of ML.
It pertains to batch supervised regression problems, where the data provided are labeled, all data is given at once, and the labels are real numbers.
While there's a broad array of topics that we could discuss, we focus on this simple setting in order to motivate the line of research without introducing too much complexity.

### Classical learning theory
Statisticians and ML theorists over the years have studied the conditions necessary for a trained model to perform well on new data.
They developed an elegant set of theories to explain when we should expect this to occur.
The core idea is that more complex models require more data; without enough data, the model will pick up only spurious correlations and noise, learning nothing of value.

To think about this mathematically, we decompose the population error term into two terms---the training error and the generalization error---and analyze how they change with the model complexity $$p$$ and the number of samples $$n$$.

$$\underbrace{\mathbb{E}_{(x, y) \sim \mathcal{D}}[\ell(f_\theta(x), y)]}_{\text{population error}} = \underbrace{\frac{1}{n} \sum_{i=1}^n \ell(f_\theta(x_i, y_i))}_{\text{training error}} + \underbrace{\left(\mathbb{E}_{(x, y) \sim \mathcal{D}}[\ell(f_\theta(x), y)] - \frac{1}{n} \sum_{i=1}^n \ell(f_\theta(x_i, y_i)) \right)}_{\text{generalization error}}$$

This classical model gives rigorous mathematical bounds that specify when the two components should be small. 
The below image represents the core trade-off that this principle implies; you can find something like it in most ML text books.
![](/assets/images/2021-06-15-candidacy-overview/classical-err.jpeg)

* If I choose a very small number of parameters $$p$$ relative to the number of samples $$n$$, then my model will perform poorly because it's too simplistic.
There will be no function $$f_\theta$$ that can classify most of the training data, the training error will be large for any choice of $$\theta$$, even though the generalization error is small.
![](/assets/images/2021-06-15-candidacy-overview/samples1.jpeg)

* There's then a "sweet spot" for $$p$$, where it's large enough to capture the complexity of the data distribution, but not too large to overfit. Here, we have a small training error _and_ a generalization error.
![](/assets/images/2021-06-15-candidacy-overview/samples2.jpeg)

* If I choose $$p$$ to be large, then I can expect _overfitting_ to occur, where the model has a training error near zero, but the generalization error is very large. 
In this setting, the model performs poorly because it only memorizes the data, without actually learning the underlying trend.
![](/assets/images/2021-06-15-candidacy-overview/samples3.jpeg)



Classical learning theory offers several tools (like VC dimension and Rademacher complexity) to quantify the complexity of a model and provide guarantees about how well we expect a model to perform.
Based on this theory, _over-parameterized models_ (which have $$p \gg n$$) are expected to be the in the Very Bad second regime, with lots of overfitting and a very large generalization error.
However, models of this form often perform much better than this theory anticipates.
As a result, there's a push to develop new theory that better captures what happens when we have more parameters than samples.

### The gap between theory and practice
The most prominent case where the classical model fails to explain good performance is for deep learning.
Deep neural networks are typically trained with large quantities of data, but they'll also have more than enough parameters needed to perfectly fit the data; often, there are more parameters than samples.
Then, they're typically trained to obtain zero training error --- and purposefully overfit the data --- using gradient descent.

Standard approaches to proving generalization bounds provide a bleak picture. For instance:
* The VC-dimension of a neural network with $$w$$ different weights and binary output is known to be $$O(w \log w)$$, which implies that it can be learned with $$O(w \log w)$$ samples.
* [BFT17](https://arxiv.org/abs/1706.08498) and [GRS17](https://arxiv.org/abs/1712.06541) gives generalization error bounds on deep neural networks based on the magnitudes of the weights; however, these bounds grow exponentially with the depth of the network unless the weights are forced to be much smaller than they are in neural networks that succeed in practice.

These approaches give no reason to suspect that the generalization error will be small at all in realistic neural networks with a large number of parameters and unrestricted weights.
But neural networks _do_ generalize to new data in practice, which leaves open the question of why that works.

This gap between application and theory indicates that there are more phenomena that are not currently accounted for in our theoretical understanding of deep learning.
This also means that much of the practical work in deep learning is not informed at all by theoretical principles.
Training neural networks has been dubbed "[alchemy](https://www.youtube.com/watch?v=ORHFOnaEzPc){:target="_blank"}" rather than science because the field is built on best practices that were learned by tinkering that are not understand on any first-principles level.
Ideally, a more explanatory theory of how neural networks work could lead to more informed practice and eventually, more interpretable models.

### Over-parameterization and double-descent

However, this series of posts is (mostly) not about deep learning.
Deep neural networks are notoriously difficult to understand mathematically, so most researchers who attempt to do so (including yours truly) must instead study highly simplified variants.
For instance, [my paper](https://arxiv.org/abs/2102.02336){:target="_blank"} about the approximation capabailities of neural networks with random weights only applies to networks of depth two, because anything else is too complex for our methodology to characterize. 
So instead of studying deep neural networks, we'll consider a broader family of ML methods (in particular, linear methods like _least-squares regression_ and _support vector machines_) when they're in over-parameterized settings with $$p \gg n$$.
The hope is that similar principles will explain both the success of simpler over-parameterized models and more complicated deep neural networks.

Broadly, this line of work challenges the famous curve above about the tradeoffs between model complexity and generalization.
It does so by suggesting that increasing the complexity of a model (or the number of parameters) can lead to situations where the generalization error decreases once more.
This idea is referred to _double descent_ and was popularized by [my advisor](https://www.cs.columbia.edu/~djhsu/){:target="_blank"} and his collaborators in papers like [BHM18](https://arxiv.org/abs/1806.05161) and [BHX19](https://arxiv.org/abs/1903.07571){:target="_blank"}. 
This augments what is referred to as the _classical regime_---where $$p \leq n$$ and choosing the right model is equivalent to choosing the "sweet spot" for $$p$$---with the _interpolation regime_.
(Interpolation means rougly the same thing as overfitting; it describes methods that bring the training loss to zero.)

![](/assets/images/2021-06-15-candidacy-overview/double-err.jpeg)

In the interpolation regime, $$p \gg n$$ and the learning algorithm selects a hypothesis that perfectly fits the training data (i.e. the training error is zero) and is somehow "smooth," which leads to some other kind of "simplicity" that then yields a good generalization error.
Here's a way to think about it: 
* When $$p \approx n$$, it's likely that there's exactly one or very few candidate functions $$f_\theta$$ that perfectly fit the data, and we have no reason to expect that this function won't be overly "bumpy" and fail to learn any underlying pattern. (See image (3) above.)
* Instead, if $$p \gg n$$, then there will be many hypotheses to choose from that have a training error of zero.
If the algorithm is somehow biased in favor of "smooth" hypotheses, then it's more likely to pick up on the underlying structure of the data.
![](/assets/images/2021-06-15-candidacy-overview/samples4.jpeg)

Of course, this is a very hand-wavy way to describe what's going on.
Also, it's not always the case that having $$p \gg n$$ leads to a nice situation like the one in image (4).
This kind of success only occurs when the learning algorithm and training distribution meet certain properties.
Through this series of posts, I'll make this more precise and describe the specific mechanisms that enable good generalization in these settings.

## What will you cover?

This series aims to have both breadth and depth.
I'll explore a wide range of settings where over-parameterized models perform well and then hone in on linear regression to understand the literature on a very granular level.
The following are topics and papers that I'll read and write about. 
I'll add links to the corresponding blog posts once they exist.

This list is subject to change, especially in the next few weeks.
If you, the reader, think there's anything important missing, please let me know!

* **Double-descent in linear models:**
These papers --- which study over-parameterization in the one of the most simple domains possible --- will be the main focus of this survey.
	* _Least-squares regression:_
		* **[CD07](https://link.springer.com/article/10.1007/s10208-006-0196-8){:target="_blank"}.** Caponnetto and De Vito. "Optimal rates for the regularized least-squares algorithm." 2007. 
		* **[AC10](https://arxiv.org/abs/1010.0072){:target="_blank"}.** Audibert and Catoni. "Linear regression through PAC-Bayesian truncation." 2010.
		* **[WF17](https://projecteuclid.org/journals/annals-of-statistics/volume-45/issue-3/Asymptotics-of-empirical-eigenstructure-for-high-dimensional-spiked-covariance/10.1214/16-AOS1487.full){:target="_blank"}.** Wang and Fan. "Asymptotics of empirical eigenstructure for high dimensional spiked covariance." 2017.
		* **[BHX19](https://arxiv.org/abs/1903.07571){:target="_blank"}. [[OPML#1]]({% post_url 2021-07-05-bhx19 %}){:target="_blank"}.** Belkin, Hsu, and Xu. "Two models of double descent for weak features." 2019. 
		* **[BLLT19](https://arxiv.org/abs/1906.11300){:target="_blank"}.** Bartlett, Long, Lugosi, and Tsigler. "Benign overfitting in linear regression." 2019.
		* **[HMRT19](https://arxiv.org/abs/1903.08560){:target="_blank"}.** Hastie, Montanari, Rosset, and Tibshirani. "Surprises in high-dimensional ridgeless least squares interpolation." 2019.
		* **[Mit19](https://arxiv.org/abs/1906.03667){:target="_blank"}.** Mitra. "Understanding overfitting peaks in generalization error: Analytical risk curves for $$\ell_2$$ and $$\ell_1$$ penalized interpolation." 2019.
		* **[MN19](https://arxiv.org/abs/1912.13421){:target="_blank"}.** Mahdaviyeh and Naulet. "Risk of the least squares minimum norm estimator under the spike covariance model." 2019.
		* **[MVSS19](https://ieeexplore.ieee.org/document/9051968){:target="_blank"}.** Muthukumar, Vodrahalli, Subramanian, and Sahai. "Harmless interpolation of noisy data in regression." 2019.		
		* **[XH19](https://proceedings.neurips.cc/paper/2019/file/e465ae46b07058f4ab5e96b98f101756-Paper.pdf){:target="_blank"}.** Xu and Hsu. "On the number of variables to use in principal component regression." 2019.
		* **[BL20](https://arxiv.org/abs/2010.08479){:target="_blank"}.** Bartlett and Long. "Failures of model-dependent generalization bounds for least-norm interpolation." 2020.
		* **[HHV20](https://arxiv.org/abs/2011.11477){:target="_blank"}.** Huang, Hogg, and Villar. "Dimensionality reduction, regularization, and generalization in overparameterized regressions." 2020.
	* _Ridge regression:_
		* **[DW15](https://arxiv.org/abs/1507.03003){:target="_blank"}.** Dobriban and Wagner. "High-dimensional asymptotics of prediction: ridge regression and classification." 2015.
		* **[TB20](https://arxiv.org/abs/2009.14286){:target="_blank"}.**Tsigler and Bartlett. "Benign overfitting in ridge regression." 2020.
	* _Kernel regression:_
		* **[Zha05](https://direct.mit.edu/neco/article/17/9/2077/7007/Learning-Bounds-for-Kernel-Regression-Using){:target="_blank"}.** Zhang. "Learning bounds for kernel regression using effective data dimensionality." 2005.
		* **[RZ19](http://proceedings.mlr.press/v99/rakhlin19a.html){:target="_blank"}.** Rakhlin and Zhai. "Learning bounds for kernel regression using effective data dimensionality." 2019.
		* **[LRZ20](http://proceedings.mlr.press/v125/liang20a.html){:target="_blank"}.** Liang, Rakhlin, and Zhai. "On the multiple descent of minimum-norm interpolants and restricted lower isometry of kernels." 2020.
	* _Support Vector Machines:_
		* **[BHMZ20](https://arxiv.org/abs/2005.11818){:target="_blank"}.** Bousquet, Hanneke, Moran, and Zhivotovskiy. "Proper learning, Helly number, and an optimal SVM bound." 2020.
		* **[MNSBHS20](https://arxiv.org/abs/2005.08054){:target="_blank"}.** Muthukumar, Narang, Subramanian, Belkin, Hsu, and Sahai. "Classification vs regression in overparameterized regimes: Does the loss function matter?" 2020.
		* **[WT20](https://arxiv.org/abs/2011.09148){:target="_blank"}.** Wang and Thrampoulidis. "Binary classification of Gaussian mixtures: abundance of support vectors, benign overfitting and regularization." 2020.
		* **[CGB21](https://arxiv.org/abs/2104.13628){:target="_blank"}.** Cao, Gu, and Belkin. "Risk bounds for over-parameterized maximum margin classification on sub-Gaussian mixtures." 2021.
	* _Random feaures models:_
		* **[MM19](https://arxiv.org/abs/1908.05355){:target="_blank"}.** Mei and Montanari. "The generalization error of random features regression: Precise asymptotics and double descent curve." 2019.


* **Training beyond zero training error in boosting:** 
While not exactly an over-parameterized model, one well-known example of when training a model past the point of perfectly fitting the data can produce better population errors is with the AdaBoost algorithm.
I'll discuss the original AdaBoost paper and how arguments about the margins of the resulting classifiers suggest that there's more to training ML models than finding the "sweet spot" discussed above and avoiding overfitting.
	* **[FS97](https://www.sciencedirect.com/science/article/pii/S002200009791504X){:target="_blank"}.** Freund and Schapire. "A decision-theoretic generalization of online learning and an application to boosting." 1997.
	* **[BFLS98](https://projecteuclid.org/journals/annals-of-statistics/volume-26/issue-5/Boosting-the-margin--a-new-explanation-for-the-effectiveness/10.1214/aos/1024691352.full){:target="_blank"}.** Bartlett, Freund, Lee, and Schapire. "Boosting the margin: a new explanation for the effectiveness of voting methods." 1998.

* **Interpolation of arbitary data in neural networks and kernel machines:**
These papers show that both neural networks and kernel machines can interpret data with arbitrary labels and still generalize, even when some fraction of data are noisy.
The former challenges the narrative that classical learning theory that's oriented around avoiding overfitting can explain deep learning generalization.
The latter suggests that the phenomena are not unique to deep neural networks and that simpler linear models are ripe for study as well.
	* **[ZBHRV17](https://arxiv.org/abs/1611.03530){:target="_blank"}.** Zhang, Bengio, Hardt, Recht, and Vinyals. "Understanding deep learning requires rethinking generalization." 2017.
	* **[BMM18](https://arxiv.org/abs/1802.01396){:target="_blank"}.** Belkin, Ma, and Mandal. "To understand deep learning we need to understand kernel learning." 2018.

* **Empirical evidence of double descent in deep neural networks:**
I'll survey a variety of papers that relate the above theoretical ideas to experimental results in deep learning.
	* **[BHMM19](https://www.pnas.org/content/116/32/15849){:target="_blank"}.** Belkin, Hsu, Ma, and Mandal. "Reconciling modern machine-learning practice and the classical bias–variance trade-off." 2019.
	* **[NKBYBS19](https://arxiv.org/abs/1912.02292){:target="_blank"}.** Nakkiran, Kaplun, Bansal, Yang, Barak, and Sutskever. "Deep double descent: Where bigger models and more data hurt." 2019.
	* **[SGDSBW19](https://iopscience.iop.org/article/10.1088/1751-8121/ab4c8b){:target="_blank"}.** Spigler, Geiger, d'Ascoli, Sagun, Biroli, and Wyart. "A jamming transition from under- to over-parametrization affects generalization in deep learning." 2019.
	* **[NVKM20](https://arxiv.org/abs/2003.01897){:target="_blank"}.** Nakkiran, Venkat, Kakade, and Ma. "Optimal regularization can mitigate double descent." 2020.


* **Smoothness of interpolating neural networks as a function of width:**
Since one of the core benefits of over-parameterized interpolation models is obtaining very smooth functions $$f_\theta$$, there's an interest in understanding how the number of parameters of a neural network can be translated to the smoothnuess of $$f_\theta$$.
These papers attempt to establish that relationship.
	* **[BLN20](https://arxiv.org/abs/2009.14444){:target="_blank"}.** Bubeck, Li, and Nagaraj. "A law of robustness for two-layers neural networks." 2020.
	* **[BS21](https://arxiv.org/abs/2105.12806){:target="_blank"}.** Bubeck and Sellke. "A universal law of robustness via isoperimetry." 2021.

## What's next?

This overview post is published alongside the first paper summary about [BHX19](https://arxiv.org/abs/1903.07571){:target="_blank"}, which is [here]({% post_url 2021-07-05-bhx19 %}){:target="_blank"}.
This paper nicely explains how linear regression can perform well in an over-parameterized regime by interpolating the data.
It relies on a fairly straight-forward mathematical argument that focuses on a toy model with samples drawn from a nice data distribution, and it's helpful for seeing the kinds of results that we can expect to prove about models with more parameters than samples.
<!-- This paper provides a detailed understanding of the performance of linear regression models under a broad set of distributional assumptions.
Notably, this paper does not actually handle the over-paramaterized regime; they have a fixed number of parameters $$p$$ and consider the limiting case where $$n \to \infty$$.
The purpose of starting with this one is to understand how researchers typically thought about why linear models worked before the last few years.
I'll shift to focusing on papers about over-parameterization in the weeks to come.
 -->I'll aim to write a new blog post about a different paper each week until I'm done.

Because research papers are by nature highly technical and because I'm trying to understand them in their full depth, most of these posts will only be accessible to readers with some background in my field.
However, I also don't want to write posts that'll be useless to anyone who isn't pursuing a PhD in machine learning theory.
My intention is that readers with some amount of background in ML will be able to read them to understand what kinds of questions learning theorists ask; if someone who's taken an undergraduate-level ML and algorithms course can't udnerstand what I'm writing, then that's on me.
I'll periodically give more technical asides into proof techniques that'll only make sense to people who work directly on research in this area, but I'll flag them so they'll be easily skippable.

Maybe this blog will become something more with time... I'm trying to get my feet wet by talking mostly about technical topics that will be primarily of interest to people in my research field, but I may end up branching out to broader subjects that will interest people who aren't theoretical CS weirdos.
