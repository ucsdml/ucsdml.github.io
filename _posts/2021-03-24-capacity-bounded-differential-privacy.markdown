---
layout: post
mathjax: true
title:  "Capacity Bounded Differential Privacy"
date:   2021-03-24 10:00:00 -0700
categories: jekyll update
tags: adversarial, differential privacy, machine learning
author: Jacob Imola
excerpt: Differential privacy protects user data from any adversary, no matter how powerful. In cases where the adversary is limited, we can say more. In this post, we'll cover capacity-bounded differential privacy, the first notion of differential privacy that takes the computational capacity of the adversary into account. We'll see how capacity-bounded differential privacy shares some of the desirable properties of differential privacy and has the potential to provide a much stronger privacy guarantee than standard differential privacy.
---

Differential privacy (DP) is a very popular definition of privacy
that has been adopted on a large scale by many entities who deal with sensitive data, including notably [the U.S. Census Bureau](https://www.census.gov/newsroom/blogs/random-samplings/2019/02/census_bureau_adopts.html).
For an in-depth introduction to the topic, the first two chapters of the [differential
privacy textbook](https://www.cis.upenn.edu/~aaroth/privacybook.html) are a 
great place to start. Briefly, 
differential privacy is a method of releasing general queries about data
without allowing any adversary to infer with much
"confidence" whether an individual participated in the dataset. 
We'll get into the precise definition later.

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
dataset, that smoking causes lung cancer. However, a differentially
private algorithm will not allow anyone to conclude with much confidence 
the value of an individual 
record in the Hospital dataset, for example that Susan has arthritis.
This is an extremely strong promise---this adversary could have unlimited
computational ability and could know everything except for the secret. In the Susan example, 
even a medical expert who knows everything but Susan's diagnosis cannot infer her
diagnosis. This assumes that Susan's public data other than her diagnosis are not 
too correlated with her diagnosis, which is not always true.
We won't cover differential privacy in the presence of 
strongly correlated data in this post, but we will soon have a post about it on
the blog; stay tuned!

As we stated, differential privacy applies to any adversary---
this has the benefit that we
don't need to consider the adversary when making a privacy guarantee.
However, in the real world, adversaries never have unlimited ability.
It may be sufficient to assume some sort of bound on the
capacity of an adversary. This could be the case, for example, when our
adversaries are automated and have a certain programmatic form or are data scientists
obligated to use code from a specific library.
Capacity-bounded differential privacy gives us the ability to impose a capacity
bound on our adversaries if there is one. It can allow for a stronger privacy
guarantee.

### Defining Differential Privacy and Capacity-Bounded DP

Differential privacy considers two datasets $X$ and
$X'$ which differ in one record---for the Hospital dataset, $X$ might be the
unmodified data and $X'$ might be the data with Susan's ailment set to "Heart
Disease" or even "Healthy". Informally, $A$ satisfies differential privacy if
an adversary cannot distinguish $A(X)$ and $A(X')$ with much confidence. To
formalize this, we consider privacy definitions that measure the adversary's 
distinguishing power using the [Rényi divergence of order $\alpha$](https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy#R%C3%A9nyi_divergence), $R_\alpha$.
This gives us [Rényi differential privacy](https://arxiv.org/abs/1702.07476),
which on this post we will shorten to just "differential privacy" or "DP":

<div class="definition">
$A$ provides $(\varepsilon, \alpha)$-differential privacy if for all possible
neighboring datasets $X,X'$, $R_\alpha(A(X), A(X')) \leq \varepsilon $.
</div>

For our purposes, we won't need to know exactly what $R_\alpha$ is. Just remember
that it measures a "distance" between distributions.

The inspiration behind capacity-bounded differential privacy comes from the
variational form of $f$-divergences, which has become widely used in
the $f$-GAN literature (see, for example, [here](https://arxiv.org/pdf/1809.04542.pdf)). 
The variational form of $R_\alpha$ gives an alternative way of viewing Rényi divergence. 
Plugging it into our definition of DP, we get

<div class="theorem">
(Variational Form of Differential Privacy)
$A$ provides $(\varepsilon, \alpha)$-differential privacy if for all possible
neighboring datasets $X,X'$ on a domain $\mathcal{X}$,
<div style="text-align:center">
$
\sup_{h:\mathcal{X} \rightarrow \mathbb{R}} \mathbb{E}_{x \sim A(X)}
[h(x)] - \mathbb{E}_{x \sim A(X')}[f_\alpha^\star(h(x))] \leq e^{(\alpha-1)\varepsilon}
\hspace{3cm} (1)
$
</div>
where $f_\alpha^\star$ is a positive convex function that passes through the 
origin (we don't need to know its specific definition here).
</div>

This definition offers a different way of viewing differential privacy.
It closely captures the
the informal phrase that *differential privacy makes it impossible 
for an adversary to distinguish $A(X)$ and $A(X')$ with confidence more than
$\varepsilon$*. Equation (1) represents the adversary with the function
$h$; we can see that it rewards an $h$ which is higher in expectation when $x$ 
comes from $A(X)$ than when $x$ comes from $A(X')$. It upper bounds the
distinguishing power of any $h$, meaning it applies to all possible adversaries.
For more on how $h$ represents the adversary, see the
penultimate section on Pinsker's inequality.

Viewing differential privacy through the adversarial lens, we are now able to
formally bound the capacity of the adversary. Instead
of letting $h$ be any function, we restrict it to a class $\mathcal{H}$ of
functions from $\mathcal{X}$ to $\mathbb{R}$. This
gives us the following definition:

<div class="definition">
$A$ provides $(\mathcal{H}, \varepsilon, \alpha)$-differential privacy if for all
possible neighboring datasets $X,X'$ on a domain $\mathcal{X}$,
<div style="text-align:center">
$\sup_{h \in \mathcal{H}} \mathbb{E}_{x \sim A(X)}
[h(x)] - \mathbb{E}_{x \sim A(X')}[f_\alpha^\star(h(x))] \leq e^{(\alpha-1)\varepsilon}
\hspace{3cm} (2)$
</div>
</div>

By restricting adversaries to
the class $\mathcal{H}$, Equation (2) is now just a privacy guarantee
against adversaries in the class $\mathcal{H}$. The LHS of (2) is upper 
bounded by, and hopefully much smaller than, the LHS of
Equation (1). The bigger the difference between the two quantities, the
stronger the privacy guarantee is for adversaries in $\mathcal{H}$ compared to
the worst-case guarantee.

Let's give a concrete example. Let $\mathcal{T}^i$ denote a decision tree
of depth $i$. Another way of thinking about $\mathcal{T}^i$ is as the set of
programs with just conditional statements 
up to depth $i$ where each conditional is allowed to ask a question about one
attribute of $X$. For some private algorithm $A$ and $\alpha$, we can say 
that it satisfies $(\mathcal{T}^i, \epsilon_i, \alpha)$-capacity bounded DP 
for all positive integers $i$. We also know that $\epsilon_1 \leq \epsilon_2
\leq \cdots$ since $\mathcal{T}^1 \subset \mathcal{T}^2 \subset \cdots$. Because
decision trees of unbounded depth can express any function, we have that $A$
satisfies $(\epsilon, \alpha)$-DP for $\epsilon = \lim_{i \rightarrow \infty}
\epsilon_i$.

### Capacity Bounded DP Satisfies the Same Important Properties as Differential Privacy

One reason differential privacy is such a great notion of privacy is that it
behaves well when the outputs of a private mechanism are manipulated, which in the
real world, they often are. One simple example of this would be rounding the
output of an algorithm so it is easier to read. **Post-processing
invariance** ensures that no matter what manipulation $M$ we do, including
adversarial manipulations, $M(A(X))$ will have the same privacy guarantee as
$A(X)$ provided $M$ does not have access to $X$ through any other means. 
Capacity-bounded DP provides post-processing invariance with a necessary caveat.
Notice that the adversary, who lies in $\mathcal{H}$, sees
$M(A(X))$. It is possible for $M$ to increase the capacity of the adversary 
beyond $\mathcal{H}$. $M$ could even be the
worst-case function $h$ in Equation (1), and then even the identity function
would be
able to distinguish $A(X)$ and $A(X')$ in the worst way possible just because it
is given access to
$M(A(X))$. Thus, we must make an assumption about $M$. Formally,

<div class="theorem">
If $A(X)$ provides $(\mathcal{H}, \varepsilon, \alpha)$ capacity-bounded DP and
$\mathcal{I} \circ M \subseteq \mathcal{H}$, then
$M(A(X))$ provides $(\mathcal{I}, \varepsilon, \alpha)$ capacity-bounded DP.
</div>

In summary, in order to apply post-processing for capacity-bounded DP, be aware
that all post-processings add to the power of the adversary. This is by design,
since capacity-bounded DP makes an assumption about how anyone, including adversaries, 
is able to post-process data.

The second important property is **composition**. This ensures that
releasing two differentially private algorithms run on the same database $X$ 
affects the privacy parameters in a simple, predictable way. There are two types
of composition: in _sequential composition_, if $A(X)$ provides $(\varepsilon_1,
\alpha)$-DP and $B(X)$ provides $(\varepsilon_2, \alpha)$-DP, then the algorithm which
releases $(A(X), B(X))$, even if $B$ depends on $A(X)$, provides
$(\varepsilon_1 + \varepsilon_2, \alpha)$-DP. In _parallel composition_, if $A, B$ have
the same definitions, $X_1,X_2$ are disjoint subsets of $X$, then
$(A(X_1), B(X_2))$ provides $(\max\{\varepsilon_1, \varepsilon_2\}, \alpha)$-DP.

Capacity-bounded DP provides
_non-adaptive_ sequential composition, where $B$ cannot depend on $A$:

<div class="theorem">
If $A(X)$ provides $(\mathcal{H}_1, \varepsilon_1, \alpha)$ capacity-bounded DP and 
$B(X)$ provides $(\mathcal{H}_2, \varepsilon_2, \alpha)$ capacity-bounded DP and $A$ and
$B$ do not depend on each other, then $(A(X),B(X))$ provides $(\mathcal{H},
\varepsilon_1 + \varepsilon_2, \alpha)$ capacity-bounded DP with respect to the
function class
\[
  \mathcal{H} = \{h(x,y): h(x,y) = h_1(x) + h_2(y), h_1 \in 
  \mathcal{H}_1, h_2 \in \mathcal{H}_2 \}
\]
</div>

There probably are cases where fully adaptive sequential composition is achievable for
capacity-bounded DP. This is an exciting, open research question! Finally,
capacity-bounded DP provides parallel composition:

<div class="theorem">
Let $A(X),B(X)$ have the same definitions as the previous theorem, but this time
allow $B$ to potentially depend on $A(X)$. Let $X_1,X_2$
be disjoint subsets of a dataset $X$. Then $(A(X_1), B(X_2))$ provides
$(\mathcal{H}, \max\{\varepsilon_1, \varepsilon_2\}, \alpha)$ capacity-bounded DP
with $\mathcal{H}$ having the same definition as the previous theorem.
</div>

The object $\mathcal{H}$ in the composition results is known as the Minkowski sum of
sets $\mathcal{H}_1$ and $\mathcal{H}_2$. It makes sense intuitively because
the adversarial power in Equation (2) is summed up when the two algorithms
are released.

### Capacity Bounded DP Can Limit Adversarial Power

Recall that Capacity Bounded DP can give a stronger privacy
guarantee for adversaries restricted to $\mathcal{H}$ compared to general
adversaries. This is measured by the difference between the LHS of Equation (1)
and Equation (2). In general, computing Equation (2) is a challenging problem, 
as it involves maximizing over the general function 
class $\mathcal{H}$. In the paper, we compute $\mathcal{H}$-bounded 
privacy guarantees for two simple mechanisms, the Laplace and Gaussian 
mechanisms, when 
$\mathcal{H} = poly_d$, the set of degree $d$ polynomials over a
single variable.
 
As a quick summary, the Gaussian mechanism $M_{Gauss}(X)$ simply adds noise
drawn from the standard Gaussian to $\sum_{x \in X} x$. We are assuming that $X$
consists of $n$ real numbers in the domain $[0,1]$. One can show that
$M_{Gauss}$ satisfies $(\frac{\alpha}{2}, \alpha)$-DP. The Laplace mechanism is
similar, but it adds noise drawn from the standard [Laplace
distribution](https://en.wikipedia.org/wiki/Laplace_distribution) (with parameter
$b=1$). The privacy guarantee of $M_{Laplace}$ is more complex, but has an
analytic form.

In our paper, we show that, for $\alpha$ not too close to $1$, that there 
is an exponential
increase in the privacy guarantee for both $M_{Gauss}$ and $M_{Laplace}$ 
when $\mathcal{H} = poly_1$ compared to unrestricted adversaries. That is, if Equation
(1) when $A = M_{Gauss}$ or $M_{Laplace}$ is tightly equal to $\varepsilon$,
then Equation (2) can be upper bounded by $\log(\varepsilon)$, provided $\alpha$
is not too close to $1$.

The following graphs contain the smallest privacy parameters $\epsilon$ 
of $M_{Gauss}$ and $M_{Laplace}$ when $\mathcal{H} = poly_d$ for $d=1,2,3$, 
and contains the smallest privacy parameter for an unrestricted privacy 
guarantee. The privacy guarantee for the 
restricted adversaries is much better than the unrestricted adversary.

{:refdef: style="text-align: center;"}
<img src="/assets/2020-07-18-capacity/laplace_poly_divs.png" width="40%">
<img src="/assets/2020-07-18-capacity/gauss_poly_divs.png" width="40%">
{:refdef:}

This plot and our accompanying theorem are promising first results that
restricting adversaries can massively improve privacy guarantees.

### A Pinsker-like Inequality for Capacity-bounded Adversaries

In this section, we explain our final result, a Pinsker-like inequality
for restricted divergences. Besides being of interest to information theorists,
the result also gives more intuition to the meaning of capacity-bounded DP.
Let's go back to the variational definition of DP.
We claimed that Equation (1)
measures an adversary's ability to tell apart $A(X)$ and $A(X')$. However, the
analogy is somewhat unclear---what does that $f^\star_\alpha$ do? 
A more natural quantity
to consider would be the same equation but with no $f^\star_\alpha$:

<div style="text-align:center">
$
\sup_{h:\mathcal{X} \rightarrow \mathbb{R}} \mathbb{E}_{x \sim A(X)}
[h(x)] - \mathbb{E}_{x \sim A(X')}[h(x)]_{} \leq \varepsilon
\hspace{3cm} (3)
$
</div>

Of course, now we must bound $h$ to prevent (3) from being overwhelmingly large.
The following is the variational form of the _total variation distance_

<div style="text-align:center">
$
\sup_{h:\mathcal{X} \rightarrow [-1,1]} \mathbb{E}_{x \sim A(X)}
[h(x)] - \mathbb{E}_{x \sim A(X')}[h(x)]_{} \leq \varepsilon
\hspace{3cm} (4)
$
</div>
The capacity-bounded version of (4) is the following:

<div style="text-align:center">
$
\sup_{h\in \mathcal{H}} \mathbb{E}_{x \sim A(X)}
[h(x)] - \mathbb{E}_{x \sim A(X')}[h(x)]_{} \leq \varepsilon
\hspace{3cm} (5)
$
</div>

We assume that functions in $\mathcal{H}$ have range on $[-1,1]$.
Equations (4) and (5) have a very nice interpretation: any adversary (or an
adversary restricted to $\mathcal{H}$) is incentivized to be close to 1 when he
believes $A(X)$ was used and be close to $-1$ when he believes $A(X')$ is used,
and the loss he incurs when he is off is linear.

Our paper gives a connection between (5) and (1)---namely, an upper bound like
(1) implies an upper bound like (5).

<div class="theorem"> (A Pinsker-like inequality)
Let $IPM^{\mathcal{H}}(A(X), A(X'))$ be the LHS of (5) and let
$R_\alpha^{\mathcal{H}}(A(X), A(X'))$ be the LHS of (1). For any distributions $P,Q$,
\[
  IPM^{\mathcal{H}}(P,Q) \leq 8 \sqrt{R_1^{\mathcal{H}}(P,Q)}
\]
</div>

This means that capacity-bounded DP, which says $R_\alpha^{\mathcal{H}}(A(X), A(X'))
\leq \varepsilon$ implies $IPM^{\mathcal{H}}(A(X),A(X')) \leq
8\sqrt{\varepsilon}$. Thus, Equation (1), which has a somewhat challenging
interpretation, implies Equation (5) which has a much simpler linear loss
interpretation. 

Equation (5) measures privacy guarantees using Total Variation
Distance, and there are problems with doing this (see Section 7.1.6. of [this
survey](https://salil.seas.harvard.edu/files/salil/files/the_complexity_of_differential_privacy.pdf)).
Thus, the analogy is not yet perfect. For future work, we are investigating how to
generalize our Pinsker-like inequality to $R_\alpha^{\mathcal{H}}$ for general $\alpha$
and instead use a stronger divergence than IPM^{\mathcal{H}}(P,Q)$. Stay
tuned!

### Conclusion

Capacity-bounded differential privacy enhances the original definition of
DP by assuming that the adversaries who have access to
private mechanisms are bounded. Like DP, Capacity-bounded DP satisfies two
extremely useful properties:
post-processing invariance and composition. Finally, we have
examples where making the capacity-bounded assumption on our adversaries
results in a much better privacy guarantee than unrestricted adversaries.
See [our paper on arxiv](https://arxiv.org/abs/1907.02159) for more information!

