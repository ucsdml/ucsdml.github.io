---
layout: post
mathjax: true
title:  "Capacity Bounded Differential Privacy"
date:   2020-06-02 10:00:00 -0700
categories: jekyll update
tags: adversarial
author: Jacob Imola
excerpt: Differential privacy protects user data from any adversary, no matter how powerful. In cases where the adversary is limited, we can say more. We present capacity-bounded differential privacy, the first notion of differential privacy that takes the computational capacity of the adversary into account. We prove it satisfies many of the desirable properties of differential privacy and that there are algorithms that provide a much stronger privacy guarantee under the capacity-bounded assumption.
---

Differential privacy (DP) is a very popular definition of privacy
that has been adopted on a large scale by Google, Apple, and the U.S. Census Bureau. 
For an introduction to the topic, the first two chapters of the [differential
privacy textbook](https://www.cis.upenn.edu/~aaroth/privacybook.html) are a 
great place to start. The goal
of differential privacy is to release general data patterns, such 
as a data model, without allowing any adversary to infer with $\varepsilon$
"confidence" whether an individual participated in the dataset. We
call this gurantee $\varepsilon$ differential privacy, and the smaller
$\varepsilon$, the stronger the guarantee.
To make things concrete, we will introduce a running example of a sensitive database
called Hospital (this example was adapted from the book [The Ethical Algorithm](https:
//www.amazon.com/Ethical-Algorithm-Science-Socially-Design/dp/0190948205)).

Name | Age | Gender | Zip Code | Smoker | Diagnosis
Richard | 64 | Male | 19146 | Y | Heart Disease
Susan | 61 | Female | 19118 | N | Arthritis
Matthew | 67 | Male | 19104 | Y | Lung Cancer
Alice | 43 | Female | 19146 | N | Chrohn's Disease
Thomas | 69 | Male | 19115 | Y | Lung Cancer

A differentially private algorithm may be able to conclude, after looking at the
dataset, that smoking is correlated to lung cancer. However, a differentially
private algorithm will not allow anyone with a degree of certainty more than
$\varepsilon$
to conclude that Susan has arthritis or that Susan even participated in the
dataset. This is an extremely strong guarantee. Even if someone with unlimited
computational ability knew everything
about Susan except for whether or not she has arthritis, upon seeing the result
of a differentially private algorithm applied to Hospital, their confidence that
Susan participated would not go up or down by more than $\varepsilon$.

While simple, this definition of differential privacy does not take
into account the fact that in the real world, the adversaries who want to find 
out something sensitive about Susan never have unlimited computational ability.
In certain cases, it may be sufficient to defend ourselves against adversaries
whose capacities are bounded somehow, such as in terms of VC dimension or
complexity of a neural net. This could be the case, for example, when our
adversaries are automated and have a certain functional form, or are programmers
who are obligated to use code with functions from a specific library.
Capacity-bounded differential privacy gives us the ability to impose a capacity
bound on our adversaries if there is one. It is a statement about the
maximum harm such adversaries can cause.

### Adversarial interpretation of differential privacy:

The standard definition of differential privacy states that there should be no 
way to distinguish a randomized algorithm $A$ when it is run on dataset $X$ from when it
is run on dataset $X'$, where $X$ and $X'$ differ in the data of just one
individual. There are many ways in which we may define what "distinguishing"
means; a very general way is to bound the
[$f$-divergence](https://en.wikipedia.org/wiki/F-divergence) between $A(X)$ and
$A(X')$. Specifically, 

<div class="definition">
$A$ provides $(\varepsilon, f)$-differential privacy if for all possible
neighboring datasets $X,X'$, $D_f(A(X), A(X')) \leq \varepsilon $.
</div>

Using specific functions $f$, we can express the two most common definitions of
differential privacy, the original [$(\varepsilon, \delta)$ definition]
(https://arxiv.org/abs/1702.07476)
and [Renyi differential privacy](https://arxiv.org/abs/1702.07476).
For the rest of this paper, we assume $f$ is the function that gives the
$\alpha$-divergence which is related to the Renyi divergence of
order $\alpha$; this will result in a Renyi DP guarantee. The exact form of $f$
does not matter for our discussion other than it is convex and satisfies $f(1) =
0$.

The inspiration behind capacity-bounded differential privacy comes from the
variational form of the $f$-divergences, which has become widely used in
the $f$-GAN literature (see, for example, [here](https://arxiv.org/pdf/1809.04542.pdf)). 
Importing that result into the definition of differential privacy, we have 


<div class="theorem">
$A$ provides $(\varepsilon, f)$-differential privacy if for all possible
neighboring datasets $X,X'$ on a domain $\mathcal{X}$,
<div style="text-align:center">
$
\sup_{h:\mathcal{X} \rightarrow \mathbb{R}} \mathbb{E}_{x \sim A(X)}
[h(x)] - \mathbb{E}_{x \sim A(X')}[f^\star(h(x))]_{} \leq \varepsilon
$
</div>
where $f^\star$ is the Fenchel dual of $f$
(it suffices to know $f^\star$ is a positive convex function that passes through the origin.).
</div>

This gives an alternative interpretation for differential privacy. Suppose there
is an adversary $h$ who wants to cause more harm when the database is $X$ than when it is $X'$, 
but all he can see is an output $x$ drawn from either $A(X)$ or $A(X')$. 
He may process the output as much as he likes and ultimately decides to cause $h(x)$ 
harm. If $x$ came from $A(X)$, then he causes $ \mathbb{E}_{x \sim P}[h(x)] $
harm on average. 
If $x$ came from $A(X')$, then he causes $\mathbb{E}_{x \sim Q}[f^\star(h(x))]$ harm on average.
He wants to maximize the difference in harm caused.

Viewing differential privacy through this adversarial interpretation, it is
strightforward how we can make an assumption on the capacity of the adversary. Instead
of letting $h$ be any function, we restrict it to a class $\mathcal{H}$ of
functions from $\mathcal{X}$ to $\mathbb{R}$. This
gives us the following definition:

<div class="definition">
$A$ provides $(\mathcal{H}, \varepsilon, f)$-differential privacy if for all
possible neighboring datasets $X,X'$ on a domain $\mathcal{X}$
$\sup_{h \in \mathcal{H}} \mathbb{E}_{x \sim A(X)}
[h(x)]_{} - \mathbb{E}_{x \sim A(X')}[f^\star(h(x))] \leq \varepsilon$, where 
$f^\star$ is the Fenchel dual of $f$.
</div>


### Does Cap. Bounded DP satisfy the same important properties as differential privacy?

One reason differential privacy is such a great definition of privacy is that it
behaves well in the presence of side information. **Post-processing
invariance** ensures that no matter how smart the adversary is, they will never
be able to worsen the privacy guarantee of $A(X)$, provided they do not have
access to $X$ through any other means. In the formal language of DP, if $A$ provides
$(\varepsilon, f)$-DP, then for any function $M$ which does not have access to
$X$ other than through $A$, $M(A)$ provides $(\varepsilon, f)$-DP.
Capacity-bounded DP provides post-processing invariance with a caveat that
makes sense after some thought. If $A$ provides $(\mathcal{H}, \varepsilon, f)$
capacity-bounded DP and someone applies $M$ to $A$, then we can only guard against those
adversaries $i$ such that $i \circ M \in \mathcal{H}$. Formally,

<div class="theorem">
If $A$ provides $(\mathcal{H}, \varepsilon, f)$ capacity-bounded DP and
$\mathcal{I} \circ M \subseteq \mathcal{H}$, then
$M(A)$ provides $(\mathcal{I}, \varepsilon, f)$ capacity-bounded DP.
</div>

An important special case of this property is when $\mathcal{H}$ is closed under
composition and $M \in \mathcal{H}$. Then, both $A$ and $M(A)$ will satisfy $(\mathcal{H},
\varepsilon, f)$ capacity-bounded DP, a result akin to post-processing
invariance.

The second important property is **adaptive composition**. This ensures that
releasing two differentially private algorithms run on the same database $X$ 
affects the privacy guarantee in a simple, predictable way. Formally, if 
$A$ provides $(\varepsilon_1,
f)$-DP and $B$ provides $(\varepsilon_2, f)$-DP, then the algorithm which
releases $(A(X), B(X))$, even if $B$ depends on $A(X)$, provides
$(\varepsilon_1 + \varepsilon_2, f)$-DP. Capacity-bounded DP provides
composition, but not the adaptive case:

<div class="theorem">
If $A$ provides $(\mathcal{H}, \varepsilon_1, f)$ capacity-bounded DP and 
$B$ provides $(\mathcal{H}, \varepsilon_2, f)$ capacity-bounded DP, and $A$ and
$B$ do not depend on each other, then $(A,B)$ provides $(\mathcal{H},
\varepsilon_1 + \varepsilon_2, f)$ capacity-bounded DP.
</div>

There probably are examples where fully adaptive composition is achievable for
capacity-bounded DP. This is an exciting, open research question!

To see these useful properties in action on our running example, suppose we run a
privacy-preserving algorithm $A$ on Hospital which outputs the average age of
patients for each diagnosis. Here is
a sample from $A(Hospital)$:

Age | Diagnosis
62.1 | Heart Disease
64.0 | Arthritis
70.4 | Lung Cancer
43.8 | Chrohn's Disease

Suppose $A$ provides $(\mathcal{H}^2, f, \varepsilon)$
capacity-bounded DP where $\mathcal{H}^2$ is the class of all
depth 2 decision trees. Post-processing for capacity-bounded DP 
says that if $D$ is a depth 1
decision tree, then $D(A)$ provides $(\mathcal{H}^1, f, \varepsilon)$
capacity-bounded DP where $\mathcal{H}^1$ is the class of all depth 1 decision
trees. Here is a sample from $D(A(Hospital))$:

Age | Diagnosis | Output of D
62.1 | Heart Disease | 0
64.0 | Arthritis | 1
70.4 | Lung Cancer| 1
43.8 | Chrohn's Disease | 0

Now, suppose we want run another private function $A_2$ that computes the most common
ailment by Zip Code. Here is the output of $A_2(Hospital)$.

Zip Code | Diagnosis
19146 | Heart Disease
19118 | Arthritis
19104 | Lung Cancer
19146 | Chrohn's Disease
19115 | Lung Cancer

Composition tells us that if $A,A_2$ provide $(\mathcal{H}^2, f, \varepsilon)$
capacity-bounded DP, then releasing both $A(Hospital),A_2(Hospital)$ provides $
\mathcal{H}^2, f, 2\varepsilon)$ capacity-bounded DP. However, composition for 
capacity-bounded DP is not adaptive, meaning that $A_2$ must not depend on
the output of $A_1$.

### Can bounding the capacity of our adversaries result in meaningful harm reduction?

Because capacity bounded DP assumes that the adversary causing harm is bounded
as opposed to the unlimited adversary assumed by DP, the maximum harm in
capacity-bounded DP must be less than that of DP. Stated another way, if $A$
provides $(\varepsilon, f)$ DP, then it provides $(\mathcal{H}, \varepsilon, f)$
capacity-bounded DP for any $\mathcal{H}$ (this follows immediately from the
definition). In this section, we explore cases when making the capacity-bounded
assumption results in _less_ harm; namely, we find an $A$ that provides $(
\varepsilon, f)$-DP and $(\mathcal{H}, \varepsilon', f)$ capacity-bounded DP for
$\varepsilon' < \varepsilon$.
This is a challenging problem, as it involves maximizing over a general function 
class $\mathcal{H}$ (see the definition). We currently have results when
$\mathcal{H} = poly_d$, the set of degree $d$ polynomials over a single
variable, and $A$ is either of two
simple yet ubiquitous mechanisms in differential privacy, the Laplace and Gaussian 
mechanisms.
For the sake of this blog
post, it suffices to know that the Laplace mechanism results in $A(X),A(X')$ being two
shifted Laplace distributions, and likewise with Gaussian distributions 
for the Gaussian mechanism, for any neighboring databases $X,X'$. 

To frame our above question more concretely, we consider the Laplace
(Gaussian) mechanism that satisfies $(\varepsilon, f)$ DP.
Our goal is to compute
\\[ 
\varepsilon' = \sup_{h \in poly_d} \mathbb{E}_{x \sim P} [h(x)] - 
      \mathbb{E} {x \sim Q}[f^\star(h(x))]
\\]
where $P,Q$ are either shifted Laplace (Gaussian) distributions. Then, the
Laplace (Gaussian) mechanism satisfies $(poly_d, \varepsilon', f)$ capacity-bounded DP.

The following graphs show $\varepsilon'$ for $d = 1,2,3$ and $\varepsilon$,
which corresponds to an unrestricted adversary. The $x$-axis is $\alpha$, which
is the order of the Renyi divergence. For the sake of this post, know that
in practice a higher $\alpha$ means a stronger Renyi DP guarantee, so focus on
$\alpha \approx 10$. This gives one value for $\varepsilon$ and three values for
$\varepsilon'$ corresponding to $d=1,2,3$. Since $\varepsilon'$ is
the maximum harm an adversary in $poly_d$ can cause, we observe that
$\varepsilon'$ increases with $d$, since the complexity of the adversary grows.
As $d \rightarrow \infty$, $\varepsilon' \rightarrow \varepsilon$. For these
low-dimensional adversaries, $\varepsilon'$ appears to be much lower than
$\varepsilon$.


{:refdef: style="text-align: center;"}
<img src="/assets/2020-06-02-capacity/laplace_poly_divs.png" width="40%">
<img src="/assets/2020-06-02-capacity/gauss_poly_divs.png" width="40%">
{:refdef:}

In the paper, we prove $\varepsilon'$ is actually a logarithmic function of
$\varespilon$ when $d = 1$ as long as $\alpha$ is constant bigger than $1$.
This is a promising result that has the
following additional implication: if we have a fixed, maximum tolerance for harm, 
then assuming the adversaries are linearly-bounded allows us to add
logarithmic noise in the Laplace or Gaussian mechanism compared to when the
adversary is unbounded. Since noise reduces the utility of our data analysis, this
can result in a massive utility gain.

### Conclusion

Capacity-bounded differential privacy enhances the original definition of
differential privacy by assuming that the adversaries who have access to
released data are bounded in capacity. Capacity-bounded DP satisfies two
very important properties that have proven to be extremely useful for regular DP
---post-processing and composition.
Finally, there
are examples where making the capacity-bounded assumption on our adversaries
results in a large risk reduction over the unrestricted adversaries assumed in standard DP.
See [our paper on arxiv](https://arxiv.org/abs/1907.02159) for more information!
