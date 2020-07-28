---
layout: post
mathjax: true
title:  "A Non-Parametric Test to Detect Data-Copying in Generative Models" 
date:   2020-07-13 10:00:00 -0700
categories: jekyll update
tags: generative modeling, generalization, overfitting
author: Casey Meehan 
excerpt: What does it mean for a generative model to overfit? We formalize the notion of 'data-copying' ':' when a generative model produces only slight variations of the training set and fails to express the diversity of the true distribution. To catch this form of overfitting, we propose a three-sample hypothesis test that is entirely model agnostic. Our experiments indicate that several standard tests condone data-copying, and contemporary generative models like VAEs and GANs can commit data-copying. 
---

In our [AISTATS 2020 paper](https://arxiv.org/abs/2004.05675), professors [Kamalika Chaudhuri](https://cseweb.ucsd.edu/~kamalika/), [Sanjoy Dasgupta](https://cseweb.ucsd.edu/~dasgupta/), and I propose some new definitions and test statistics for conceptualizing and measuring overfitting by generative models. 

Overfitting is a basic stumbling block of any learning process. Take learning to cook for example. In quarantine, I've attempted ~60 new recipes and can recreate ~45 of them reliably decently. The recipes are my training set and the fraction I can recreate is a sort of training error. While this training error is not exactly impressive, if you ask me to riff on these recipes and improvise, the result (i.e. dinner) will be dramatically worse. 

{:refdef: style="text-align: center;"}
<img src="/assets/2020-07-13-data-copying/supervised_overfitting_2.png" width="75%">
{:refdef}

It is well understood that our models tend to do the same -- deftly regurgintating their training data, yet struggling to generalize to unseen examples similar to the training data. Learning theory has nicely formalized this in the supervised setting. Our classification and regression models start to overfit when we observe a gap between training and (held-out) test prediction error, as in the above figure for the overly complex models.  

This notion of overfitting relies on being able to measure prediction error or perhaps log likelihood, which is rarely a barrier in the supervised setting; supervised models generally output low dimensional, simple predictions. Such is not the case in the generative setting where we ask models to output original, high dimensional, complex entities like images or natural language. Here, we certainly have no notion of prediction error and likelihoods are intractible for many models: VAEs only provide a lower bound of the data likelihood, and GANs only leave us with their samples.

{:refdef: style="text-align: center;"}
<img src="/assets/2020-07-13-data-copying/unsupervised_setting_2.png" width="75%">
{:refdef}

Without a tractable likelihood, we evaluate generative models by comparing their generated samples with those of the true distribution. This practice is well established by existing generative model tests like the [Frechet Inception Distance](https://arxiv.org/abs/1706.08500), [Kernel MMD](https://arxiv.org/abs/1611.04488), and [Precision & Recall test](https://arxiv.org/abs/1806.00035). But in absence of ground truth labels, what exactly are we testing for? We argue that unlike supervised models, generative models exhibit two varieties of overfitting: **over-representation** and **data-copying**. 

### Data-copying vs. Over-representation

Most generative model tests like those listed above check for over-representation: the tendency of a model to over-emphasize certain regions of the instance space by assigning more probability mass there than it should. Consider a data distribution $P$ over an instance space $\mathcal{X}$ of cat cartoons. Region $\mathcal{C} \subset \mathcal{X}$ specifically contains cartoons of cats with bats. Using training set $T \sim P$, we train a generative model $Q$ from which we draw a sample $Q_m \sim Q$. 

{:refdef: style="text-align: center;"}
<img src="/assets/2020-07-13-data-copying/overrepresentation_2.png" width="95%">
{:refdef}

Evidently, the model $Q$ really likes region $\mathcal{C}$, generating an undue share of cats with bats. More formally, we say $Q$ is over-representing some region $\mathcal{C}$ when 

\\[ \Pr_{x \sim Q}[x \in \mathcal{C}] \gg \Pr_{x \sim P}[x \in \mathcal{C}]  \\]

This can be measured with a simple two-sample hypothesis test, as was done in Richardson & Weiss's [2018 paper](https://arxiv.org/abs/1805.12462) demonstrating the efficacy of Gaussian mixture models in high dimension. 

Data-copying, on the other hand, occurs when $Q$ produces samples that are *closer to training set $T$* than they should be. To test for this, we equip ourselves with a held-out test sample $P_n \sim P$ in addition to some distance metric $d(x,T)$ that measures proximity to the training set of any $x \in \mathcal{X}$. We then say that '$Q$ is data-copying training set $T$' when $x \sim Q$ are on average closer to $T$ than are $x \sim P$.  


{:refdef: style="text-align: center;"}
<img src="/assets/2020-07-13-data-copying/data_copying_1_.png" width="95%">
{:refdef}

We define proximity to training set $d(x,T)$ to be the distance between $x$ and its nearest neighbor in $T$ according to some metric $d_\mathcal{X}:\mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$. Specifically 

\\[ d(x,T) = \min_{t \in T}d_\mathcal{X}(x,t) \\]

At a first glance, the generated samples in the above figure look perfectly fine, representing the different regions nicely. But taken alongside its training and test sets, we see that it has effectively copied the cat with bat in the lower right corner (for visualization, we let euclidean distance $d_\mathcal{X}$ be a proxy for similarity).  

{:refdef: style="text-align: center;"}
<img src="/assets/2020-07-13-data-copying/data_copying_2_.png" width="95%">
{:refdef}

More formally, $Q$ is data-copying $T$ in some region $\mathcal{C} \subset \mathcal{X}$ when

\\[ \Pr_{x \sim Q, z \sim P}[d(x,T) < d(z,T) \mid x,z \in \mathcal{C}] \gg \frac{1}{2}\\] 

The key takeaway here is that data-copying and over-representation are *orthogonal failure modes* of generative models. A model that exhibits over-representation may or may not data-copy and vice versa. As such, it is critical that we test for both failure modes when desigining and training models. 

{:refdef: style="text-align: center;"}
<img src="/assets/2020-07-13-data-copying/orthogonal_concepts_2_.png" width="70%">
{:refdef}

Returning to my failed culinary ambitions, I tend to both data-copy recipes I've tried *and* over-represent certain types of cuisine. If you look at the 'true distribution' of recipes online, you will find that there is a tremendous diversity of cooking styles and cuisines. However, put in the unfortunate circumstance of having me cook for you, I will most likely produce some slight variation of a recipe I've recently tried. And, even though I have attempted a number of Indian, Mexican, Italian, and French dishes, I tend to over-represent bland pastas and salads when left to my own devices. To cook truly original food, one must both be creative enough to go beyond the recipes they've seen *and* versatile enough to make a variety of cuisines. So, be sure to test for both data-copying and over-representation, and do not ask me to cook for you.  

### A Three-Sample Test for Data-Copying

Adding another test to one's modeling pipeline is tedious. The good news is that data-copying can be tested with a single snappy three-sample hypothesis test. It is non-parametric, and concentrates nicely with increasing test-set and generated samples. 

As described in the previous section, we use a training sample $T \sim P$, a held-out test sample $P_n \sim P$, and a generated sample $Q_m \sim Q$. We additionally need some distance metric $d_\mathcal{X}(x,z)$. In practice, we choose $d_\mathcal{X}(x,z)$ to be the euclidean distance between $x$ and $z$ after being embedded by $\phi$ into some lower-dimensional perceptual space: $d_\mathcal{X}(x,z) = \\| \phi(x) - \phi(z) \\|_2$. The use of such embeddings is common practice in testing generative models as exhibited by several existing over-representation tests like  [Frechet Inception Distance](https://arxiv.org/abs/1706.08500)  and [Precision & Recall](https://arxiv.org/abs/1806.00035).  

Following intuition, it is tempting to check for data-copying by simply differencing the expected distance to training set:

<div>
$$
\mathbb{E}_{x \sim Q} [d(x,T)] - \mathbb{E}_{x \sim P} [d(x,T)] \approx  \frac{1}{m} \sum_{x_i \in Q_m} d(x_i, T) - \frac{1}{n} \sum_{x_i \in P_n}d(x_i, T) \ll 0 
$$
</div>

where, to reiterate, $d(x,T)$ is the distance $d_\mathcal{X}$ between $x$ and its nearest neighbor in $T$. This statistic --- an expected distance --- is a little too finicky: the variance is far out of our control, influenced by both the choice of distance metric and by outliers in both $P_n$ and $Q_m$. So, instead of probing for how *much* closer $Q$ is to $T$ than $P$ is, we probe for how *often* $Q$ is closer to $T$ than $P$ is: 

<div>
$$
\mathbb{E}_{x \sim Q, z \sim P} [\mathbb{1}_{d(x,T) > d(z,T)}] \approx  \frac{1}{nm} \sum_{x_i \in Q_m, z_j \in P_n} \mathbb{1} \big( d(x_i, T) > d(z_j, T) \big) \ll \frac{1}{2} 
$$
</div>

This statistic --- a probability --- is closer to what we want to measure, and more stable. It tells us how much more likely samples in $Q_m$ are to fall near samples in $T$ relative to the held out samples in $P_n$. If it is much less than a half, then significant data-copying is occuring. This statistic is much more robust to outliers and lower variance. Additionally, by measuring a probability instead of an expected distance, this statistic is transferrable between different data domains and distance metrics: less than half is always overfit, half is always good, and over half is underfit (in the sense that the generated samples are further from the training set than they should be). We are also able to show that this indicator statistic has nice concentration properties agnostic to the chosen distance metric.

It turns out that the above test is an instantiation of the [Mann Whitney hypothesis test](https://en.wikipedia.org/wiki/Mann-Whitney_U_test), proposed in 1947, for which there are computationally efficient implementations in packages like [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html). By $Z$-scoring the Mann Whitney statistic, we normalize its mean to zero and variance to one. We call this statistic $Z_U$. As such, a generative model $Q$ with $Z_U \ll 0$ is heavily data-copying and a score $Z_U \gg 0$ is underfitting. Near 0 is ideal.  

### Handling Heterogeneity
An operative phrase that you may have noticed in the above definition of data-copying is "on average". Is the generative model closer to the training data than it should be *on average*? This, unfortunately, is prone to false negatives. If $Z_U \ll 0$, then $Q$ is certainly data-copying in some region $\mathcal{C} \subset \mathcal{X}$. However, if $Z_U \geq 0$, it may still be excessively data-copying in one region and significantly underfitting in another, leading to a test score near 0. 

{:refdef: style="text-align: center;"}
<img src="/assets/2020-07-13-data-copying/bins_1.png" width="33%">
{:refdef}

For example, let the $\times$'s denote training samples and the red dots denote generated samples. Even without observing a held-out test sample, it is clear that $Q$ is data-copying in pink region and underfitting in the green region. $Z_U$ will fall near 0, suggesting the model is performing well despite this highly undesirable behavior.

To prevent this misreading, we employ an algorithmic tool seen frequently in non-parametric testing: binning. Break the instance space into a partition $\Pi$ consisting of $k$ 'bins' or 'cells' $\pi \in \Pi$ and collect $Z_U^\pi$ each cell $\pi$. 

{:refdef: style="text-align: center;"}
<img src="/assets/2020-07-13-data-copying/bins_2.png" width="33%">
{:refdef}

The statistic maintains its concentration properties within each cell. The more test and generated samples we have ($n$ and $m$), the more bins we can construct, and the more we can precisely pinpoint a model's data-copying behavior. The 'goodness' of model's fit is an inherently multidimensional entity, and it is informative to explore the range of $Z_U^\pi$ values seen in all cells $\pi \in \Pi$. Our experiments indicate that VAEs and GANs both tend to data-copy in some cells and underfit in others. However, to boil all this down into a single statistic for model comparisons, we simply take an average of the $Z_U^\pi$ values weighted by the number of test samples in the cell: 

<div>
$$
C_T = \sum_{\pi \in \Pi} \frac{\#\{P_n \in \pi\}}{n} Z_U^\pi 
$$
</div>
(In practice, we restrict ourselves to cells with a sufficient number of generated samples. See the [paper](https://arxiv.org/abs/2004.05675).). Intuitively, this statistic tells us whether the model tends to data-copy in the regions most heavily emphasized by the true distribution. It does not tell us whether or not the model $Q$ data-copies *somewhere*. 

### Experiments: data-copying in the wild 
Observing data-copying in VAEs and GANs indicates that the $C_T$ statistic above serves as an instructive tool for model selection. For a more methodical interrogation of the $C_T$ statistic and comparison with baseline tests, be sure to check out the [paper](https://arxiv.org/abs/2004.05675). 

To test how VAE complexity relates to data-copying, we train 20 VAEs on  MNIST with increasing width as indicated by the latent dimension. For each model $Q$, we draw a sample of generated images $Q_m$, and compare with a held out test set $P_n$ to measure $C_T$. Our distance metric is given by the 64d latent space of an autoencoder we trained with a VGG perceptual loss produced by [Zhang et al.](https://arxiv.org/abs/1801.03924). The purpose of this alternative latent space is to provide an embedding that both provides a perceptual distance between images and is independent of the VAE embeddings. For partitioning, we simply take the voronoi cells induced by the $k$ centroids found by $k$-means run on the embedded training dataset.     

{:refdef: style="text-align: center;"}
<img src="/assets/2020-07-13-data-copying/VAE_overfitting.png" width="49%">
<img src="/assets/2020-07-13-data-copying/VAE_gen_gap.png" width="46%">
{:refdef}

Recall that $C_T \ll 0$ indicates data-copying and $C_T \gg 0$ indicates underfitting. We see (above, left) that overly complex models (towards the left of the plot) tend to copy their training set, and simple models (towards the right of the plot) tend to underfit, just as we might expect. Furthermore, $C_T = 0$ approximately coincides with the maximum ELBO, the VAE's likelihood lower bound. For comparison, take the generalization gap of the VAEs' ELBO on the training and test sets (above, right). The gap remains large for both overly complex models ($d > 50$) and simple models ($d < 50$). With the ELBO being a lower bound to the likelihood, it is difficult to interpret precisely why this happens. Regardless, it is clear that the ELBO gap is a compartively imprecise measure of overfitting.      

While the VAEs exhibit increasing data-copying with model complexity *on average*, most of them have cells that are over- and underfit. Poking into the individual cells $\pi \in \Pi$, we can take a look at the difference between a $Z_U^\pi \ll 0$ cell and a $Z_U^\pi \gg 0$ cell:     

{:refdef: style="text-align: center;"}
<img src="/assets/2020-07-13-data-copying/VAE_cells.png" width="90%">
{:refdef}

The two strips exhibit two regions of the same VAE. The bottom row of each shows individual generated samples from the cell, and the top row shows their training nearest neighbors. We immediately see that the data-copied region (left, $Z_U^\pi = -8.54$) practiaclly produces blurry replicas of its training nearest neighbors, while the underfit region (right, $Z_U^\pi = +3.3)$ doesn't appear to produce samples that look like any training image.   

Extending these tests to a more complex and practical domain, we check the ImageNet-trained [BigGAN](https://arxiv.org/abs/1809.11096) model for data-copying. Being a conditional GAN that can output images of any single ImageNet 12 class, we condition on three separate classes and treat them as three separate models: Coffee, Soap Bubble, and Schooner. Here, it is not so simple to re-train GANs of varying degrees of compexity as we did before with VAEs. Instead, we modulate the model's 'trunction threshold': a level beyond which all inputs are resampled. A larger truncation threshold allows for higher variance latent input, and thus higher variance outputs.     

{:refdef: style="text-align: center;"}
<img src="/assets/2020-07-13-data-copying/GAN_overfitting.png" width="60%">
{:refdef}

Low truncation thresholds restrict the model to producing samples near the mode -- those it is most confident in. However it appears that in all image classes, this also leads to significant data copying. Not only are the samples less diverse, but they hang closer to the training set than they should. This contrasts with the BigGAN authors' suggestion that truncation level trades off between 'variety and fidelity'. It appears that it might trade off between 'copying and not copying' the training set. 

Again, even the least copying models with maximized truncation (=2) exihibit data-copying in *some* cells $\pi \in \Pi$:   


{:refdef: style="text-align: center;"}
<img src="/assets/2020-07-13-data-copying/GAN_cells.png" width="95%">
{:refdef}

The left two strips show show data-copied cells of the coffee and bubble instance spaces (low $Z_U^\pi$), and right two strips show underfit cells (high $Z_U^\pi$). The bottom row of each strip shows a subset of generated images from that cell, and the top row training images from the cell. To show the diversity of the cell, these are not necessarily the generated samples' training nearest neighbors as they were in the MNIST example.  

We see that the data-copied cells on the left tend to confidently produce samples of one variety, that linger too closely to some specific examples it caught in the training set. In the coffee case, it is the teacup/saucer combination. In the bubble case, it  is the single large suspended bubble with blurred background. Meanwhile, the slightly underfit cells on the right arguably perform better in a 'generative' sense. The samples, albeit slightly distorted, are more original. According to the inception space distance metric, they hug less closely to the training set.  

### Data-copying is a real failure mode of generative models

The moral of these experiments is that data-copying indeed occurs in contemporary generative models. This failure mode has significant consequences for user privacy and for model generalization. With that said, it is a failure mode not identified by most prominent generative model tests in the literature today.  

- Data-copying is *orthogonal to* over-representation; both should be tested when designing and training generative models. 

- Data-copying is straightforward to test efficiently when equipped with a decent distance metric. 

- Having identified this failure mode, it would be interesting to see modeling techniques that actively try to minimize data-copying in training. 

So be sure to start probing your models for data-copying, and don't be afraid to venture off-recipe every once in a while! 

### More Details

Check out [our AISTATS paper on arxiv](https://arxiv.org/abs/2004.05675), and [our data-copying test code on GitHub](https://github.com/casey-meehan/data-copying). 

