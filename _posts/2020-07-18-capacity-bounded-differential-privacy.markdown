---
layout: post
mathjax: true
title:  "Capacity Bounded Differential Privacy"
date:   2020-07-18 10:00:00 -0700
categories: jekyll update
tags: adversarial
author: Jacob Imola
excerpt: Differential privacy protects user data from any adversary, no matter how powerful. In cases where the adversary is limited, we can say more. In this post, we'll cover capacity-bounded differential privacy, the first notion of differential privacy that takes the computational capacity of the adversary into account. We'll see how capacity-bounded differential privacy shares some of the desirable properties of differential privacy and has the potential to provide a much stronger privacy guarantee than standard differential privacy.
---

Differential privacy (DP) is a very popular definition of privacy
that has been adopted on a large scale by many tech companies dealing with sensitive data and, notably, [the U.S. Census Bureau](https://www.census.gov/newsroom/blogs/random-samplings/2019/02/census_bureau_adopts.html).
For an in-depth introduction to the topic, the first two chapters of the [differential
privacy textbook](https://www.cis.upenn.edu/~aaroth/privacybook.html) are a 
great place to start. Briefly, $\varepsilon$-
differential privacy is a method of releasing general queries about data
without allowing any adversary to infer with $\varepsilon$
"confidence" whether an individual participated in the dataset. 
The smaller the $\varepsilon$, the stronger the privacy guarantee. We'll get
into the precise definition later.

To make things concrete, let's introduce an example of a sensitive database
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
private algorithm will not allow anyone with confidence more than
$\varepsilon$ to conclude the value of an individual record in the Hospital dataset, 
for example that Susan has arthritis.
This is an extremely strong guarantee. Even if someone with unlimited
computational ability knew everything
about Susan except for whether or not she has arthritis, upon seeing the result
of a differentially private algorithm applied to Hospital, their confidence
about Susan's record will not go up or down by more than about $\varepsilon$.

The promise of differential privacy applies to any adversary, so it is strong. 
However, in the real world, adversaries never have unlimited ability.
In certain cases, it may be sufficient to assume some sort of bound on the
capacity of an adversary. This could be the case, for example, when our
adversaries are automated and have a certain functional form or are programmers
obligated to use code with functions from a specific library.
Capacity-bounded differential privacy gives us the ability to impose a capacity
bound on our adversaries if there is one. In this way, it enriches the original
notion of differential privacy.

### Adversarial interpretation of differential privacy:

Differential privacy considers two datasets $X$ and
$X'$ which differ in one record---for the Hospital dataset, $X$ might be the
unmodified data and $X'$ might be the data with Susan's ailment set to "Heart
Disease" or even "Healthy". Informally, $A$ satisfies differential privacy if
an adversary cannot distinguish $A(X)$ and $A(X')$ with much confidence. We 
consider a general definition that measures the adversary's distinguishing power 
using the [$f$-divergence](https://en.wikipedia.org/wiki/F-divergence), $D_f$.

<div class="definition">
$A$ provides $(\varepsilon, f)$-differential privacy if for all possible
neighboring datasets $X,X'$, $D_f(A(X), A(X')) \leq \varepsilon $.
</div>

We use the $f$-divergence here because, as mentioned, it is very general. For 
specific $f$, we can recover the two most 
common definitions of differential privacy, the original 
[$(\varepsilon, \delta)$ definition](https://arxiv.org/abs/1702.07476)
and [Renyi differential privacy](https://arxiv.org/abs/1702.07476).
For this post, we don't need to consider a specific form for $f$, just remember
that $D_f$ measures a "distance" between distributions.

The inspiration behind capacity-bounded differential privacy comes from the
variational form of the $f$-divergences, which has become widely used in
the $f$-GAN literature (see, for example, [here](https://arxiv.org/pdf/1809.04542.pdf)). 
Substituting the variational form of $D_f$ into our definition, we get

<div class="theorem">
(Variational Form of Differential Privacy)
$A$ provides $(\varepsilon, f)$-differential privacy if for all possible
neighboring datasets $X,X'$ on a domain $\mathcal{X}$,
<div style="text-align:center">
$
\sup_{h:\mathcal{X} \rightarrow \mathbb{R}} \mathbb{E}_{x \sim A(X)}
[h(x)] - \mathbb{E}_{x \sim A(X')}[f^\star(h(x))]_{} \leq \varepsilon
\hspace{3cm} (1)
$
</div>
where $f^\star$ is the Fenchel dual of $f$
(it suffices to know $f^\star$ is a positive convex function that passes through the origin.).
</div>

This definition looks somewhat scary, but it's really saying what we
mean by the informal phrase that *differential privacy makes it impossible 
for an adversary to distinguish $A(X)$ and $A(X')$ with confidence more than
$\varepsilon$*. Equation (1) captures the adversary with the function
$h$; we can see that it rewards an $h$ which is higher in expectation when $x$ 
comes from $A(X)$ than when $x$ comes from $A(X')$.
For more on this interpretation, please see the
penultimate section about Pinsker's inequality.

Viewing differential privacy through the adversarial lens, we are now able to
formally bound the capacity of the adversary. Instead
of letting $h$ be any function, we restrict it to a class $\mathcal{H}$ of
functions from $\mathcal{X}$ to $\mathbb{R}$. This
gives us the following definition:

<div class="definition">
$A$ provides $(\mathcal{H}, \varepsilon, f)$-differential privacy if for all
possible neighboring datasets $X,X'$ on a domain $\mathcal{X}$,
<div style="text-align:center">
$\sup_{h \in \mathcal{H}} \mathbb{E}_{x \sim A(X)}
[h(x)]_{} - \mathbb{E}_{x \sim A(X')}[f^\star(h(x))] \leq \varepsilon$,
</div>
where $f^\star$ is the Fenchel dual of $f$.
</div>

Let's give a concrete example. Let $\mathcal{T}^i$ denote a decision tree
of depth $i$. Let's assume all adversaries are programs consisting of
conditional statements up to depth $k$. Roughly, this corresponds to the class
$\mathcal{T}^k.$ We would then use $(\mathcal{H}^k, \varepsilon, f)$
capacity-bounded differential privacy for suitable choices of $\varepsilon, f$
as our privacy guarantee.
Note that since decision trees of high-enough depth can express any function, we have
$(\mathcal{H}^\infty, \varepsilon, f)$ capacity-bounded DP is the same as
$(\varepsilon, f)$-DP.

### Does Capacity Bounded DP satisfy the same important properties as differential privacy?

One reason differential privacy is such a great definition of privacy is that it
behaves well in the presence of side information. **Post-processing
invariance** ensures that no matter what the adversary does, they will never
be able to worsen the privacy guarantee of $A(X)$ provided they do not have
access to $X$ through any other means. Formally, if $A(X)$ provides
$(\varepsilon, f)$-DP, then for any function $M$ that takes arguments $y$ that
do not depend on $X$, $M(A(X), y)$ provides $(\varepsilon, f)$-DP.
Capacity-bounded DP provides post-processing invariance with a necessary caveat.
If $A(X)$ provides $(\mathcal{H}, \varepsilon, f)$
capacity-bounded DP but the adversary sees $M(A(X))$, $M$ could possibly
increase the capacity of the adversary. $M$ could even be the
worst-case function $h$ in the variational form of differential privacy. Thus, 
we must make an assumption about $M$. Formally,

<div class="theorem">
If $A(X)$ provides $(\mathcal{H}, \varepsilon, f)$ capacity-bounded DP and
$\mathcal{I} \circ M \subseteq \mathcal{H}$, then
$M(A(X))$ provides $(\mathcal{I}, \varepsilon, f)$ capacity-bounded DP.
</div>

In summary, in order to apply post-processing, you need to
know what functions will be applied to $A$. However, if you are using
capacity-bounded DP, you are assuming a bounded adversary which means you 
know all possible adversaries that will have access to $A$ . Thus, you need to be
careful to not post-process $A$ in a way that will break your bounded adversaries
assumption.

The second important property is **composition**. This ensures that
releasing two differentially private algorithms run on the same database $X$ 
affects the privacy guarantee in a simple, predictable way. There are two types
of composition: in _sequential composition_, if $A(X)$ provides $(\varepsilon_1,
f)$-DP and $B(X)$ provides $(\varepsilon_2, f)$-DP, then the algorithm which
releases $(A(X), B(X))$, even if $B$ depends on $A(X)$, provides
$(\varepsilon_1 + \varepsilon_2, f)$-DP. In _parallel composition_, if $A, B$ have
the same definitions, $X_1,X_2$ are disjoint subsets of $X$, and $A,B$ do not
depend on each other, then
$(A(X_1), B(X_2))$ provides $(\max\{\varepsilon_1, \varepsilon_2\}, f)$-DP.

Capacity-bounded DP provides
_non-adaptive_ sequential composition, where $B$ cannot depend on $A$:

<div class="theorem">
If $A(X)$ provides $(\mathcal{H}_1, \varepsilon_1, f)$ capacity-bounded DP and 
$B(X)$ provides $(\mathcal{H}_2, \varepsilon_2, f)$ capacity-bounded DP and $A$ and
$B$ do not depend on each other, then $(A(X),B(X))$ provides $(\mathcal{H},
\varepsilon_1 + \varepsilon_2, f)$ capacity-bounded DP with respect to the
function class
\[
  \mathcal{H} = \{h(x,y): h(x,y) = h_1(x) + h_2(y), h_1 \in 
  \mathcal{H}_1, h_2 \in \mathcal{H}_2 \}_{}
\]
</div>

There probably are examples where fully adaptive sequential composition is achievable for
capacity-bounded DP. This is an exciting, open research question! Finally,
capacity-bounded DP provides parallel composition:

<div class="theorem">
Let $A(X),B(X)$ have the same definitions as the previous theorem, and recall
they do not depend on each other. Let $X_1,X_2$
be disjoint subsets of a dataset $X$. Then $(A(X_1), B(X_2))$ provides
$(\mathcal{H}, \max\{\varepsilon_1, \varepsilon_2\}, f)$ capacity-bounded DP
with $\mathcal{H}$ having the same definition as the previous theorem.
</div>

### Can bounding the capacity of our adversaries result in meaningful harm reduction?

Because capacity bounded DP assumes that the adversary causing harm is bounded,
the maximum harm caused by these adversaries must be at most the maximum harm 
of an arbitary adversary. Going back to the adversarial interpretation of DP, we
may say equivalently that if $A$
provides $(\varepsilon, f)$ DP, then it provides $(\mathcal{H}, \varepsilon, f)$
capacity-bounded DP for any $\mathcal{H}$. In this section, we explore cases when making the capacity-bounded
assumption results in _less_ harm; namely, we find an $A$ that provides $(
\varepsilon, f)$-DP and $(\mathcal{H}, \varepsilon', f)$ capacity-bounded DP for
$\varepsilon' < \varepsilon$.
This is a challenging problem, as it involves maximizing over a general function 
class $\mathcal{H}$ (see the definition of capacity-bounded DP). We currently have results when
$\mathcal{H} = poly_d$, the set of degree $d$ polynomials over a single
variable, and $A$ is either of two
simple yet ubiquitous mechanisms in differential privacy, the Laplace and Gaussian 
mechanisms.
For the sake of this blog
post, it suffices to know that the Laplace mechanism results in $A(X),A(X')$ being two
shifted Laplace distributions, and likewise with Gaussian distributions 
for the Gaussian mechanism, for any neighboring databases $X,X'$. 

To frame our above question more concretely, we consider the Laplace
(Gaussian) mechanism that satisfies $(\varepsilon, f)$-DP, meaning that
<div style="text-align:center">
$
\varepsilon = \sup_{h : \mathbb{R} \rightarrow \mathbb{R}} \mathbb{E}_{x \sim P} [h(x)] - 
      \mathbb{E}_{x \sim Q}[f^\star(h(x))]
$
</div>
where $P,Q$ are shifted Laplace (Gaussian) distributions.
Our goal is to compute
<div style="text-align:center">
$
\varepsilon' = \sup_{h \in poly_d} \mathbb{E}_{x \sim P} [h(x)] - 
      \mathbb{E}_{x \sim Q}[f^\star(h(x))]
$
</div>
Then, the Laplace (Gaussian) mechanism satisfies $(poly_d, \varepsilon', f)$ capacity-bounded DP.
The following graphs show $\varepsilon'$ for degree $d = 1,2,3$ and $\varepsilon$
(under the Unrestricted label). The $x$-axis is $\alpha$, which
is the order of the Renyi divergence. For the sake of this post, know that
in practice a higher $\alpha$ means a stronger Renyi DP guarantee, so focus on
$\alpha \approx 5$. This gives one value for $\varepsilon$ and three values for
$\varepsilon'$ corresponding to $d=1,2,3$. Since $\varepsilon'$ is
the maximum harm an adversary in $poly_d$ can cause, we observe that
$\varepsilon'$ increases with $d$, since the complexity of the adversary grows.
As $d \rightarrow \infty$, $\varepsilon' \rightarrow \varepsilon$. For these
low-dimensional adversaries, $\varepsilon'$ appears to be much lower than
$\varepsilon$.


{:refdef: style="text-align: center;"}
<img src="/assets/2020-07-18-capacity/laplace_poly_divs.png" width="40%">
<img src="/assets/2020-07-18-capacity/gauss_poly_divs.png" width="40%">
{:refdef:}

In the paper, we prove $\varepsilon'$ is actually a logarithmic function of
$\varepsilon$ when $d = 1$ as long as $\alpha$ is a constant bigger than $1$.
This is a promising result that has the
following additional implication: if we have a fixed, maximum tolerance for harm, 
then assuming the adversaries are linearly-bounded allows us to add
logarithmically as much noise in the Laplace or Gaussian mechanism compared to when the
adversary is unbounded. Since noise reduces the utility of our data analysis, this
can result in a massive utility gain.

### A Pinsker-like Inequality for Capacity-bounded Adversaries

We claimed that Equation (1) in the variational form of differential privacy
measures an adversary's ability to tell apart $A(X)$ and $A(X')$. However, the
analogy is rather weak---what is that $f^* $ doing? A much more natural quantity
would be to get rid of the $f^*$:

<div style="text-align:center">
$
\sup_{h:\mathcal{X} \rightarrow \mathbb{R}} \mathbb{E}_{x \sim A(X)}
[h(x)] - \mathbb{E}_{x \sim A(X')}[h(x)]_{} \leq \varepsilon
\hspace{3cm} (2)
$
</div>

Of course, now we must bound $h$ to prevent (2) from being overwhelmingly large.
The following is the variational form of the _total variation distance_

<div style="text-align:center">
$
\sup_{h:\mathcal{X} \rightarrow [-1,1]} \mathbb{E}_{x \sim A(X)}
[h(x)] - \mathbb{E}_{x \sim A(X')}[h(x)]_{} \leq \varepsilon
\hspace{3cm} (3)
$
</div>
The capacity-bounded version of (3) is the following:

<div style="text-align:center">
$
\sup_{h\in \mathcal{H}} \mathbb{E}_{x \sim A(X)}
[h(x)] - \mathbb{E}_{x \sim A(X')}[h(x)]_{} \leq \varepsilon
\hspace{3cm} (4)
$
</div>

We must assume that functions in $\mathcal{H}$ have range on $[-1,1]$.
Equations (3) and (4) have a very nice interpretation: any adversary (or an
adversary restricted to $\mathcal{H}$) is incentivized to be close to 1 when he
believes $A(X)$ was used and be close to $-1$ when he believes $A(X')$ is used.
(3) and (4) measure the expected advantage he has when he is not bounded and when he is.

Our paper gives a connection between (4) and (1)---namely, an upper bound like
(1) implies an upper bound like (4).

<div class="theorem"> (A Pinsker-like inequality)
Let $IPM^{\mathcal{H}}(A(X), A(X'))$ be the LHS of (4) and let
$D_f^{\mathcal{H}}(A(X), A(X'))$ be the LHS of (1). For any distributions $P,Q$,
\[
  IPM^{\mathcal{H}}(P,Q) \leq 8 \sqrt{D_f^{\mathcal{H}}(P,Q)}
\]
</div>

This means that capacity-bounded DP, which says $D_f^{\mathcal{H}}(A(X), A(X'))
\leq \varepsilon$ implies $IPM^{\mathcal{H}}(A(X),A(X')) \leq
8\sqrt{\varepsilon}$ which, for small $\varepsilon$, has a nice privacy interpretation.

### Conclusion

Capacity-bounded differential privacy enhances the original definition of
differential privacy by assuming that the adversaries who have access to
released data are bounded in capacity. Capacity-bounded DP satisfies two
very important properties that have proven to be extremely useful for regular DP
---post-processing invariance and composition. Finally, there
are examples where making the capacity-bounded assumption on our adversaries
results in a large risk reduction over unrestricted adversaries.
See [our paper on arxiv](https://arxiv.org/abs/1907.02159) for more information!
