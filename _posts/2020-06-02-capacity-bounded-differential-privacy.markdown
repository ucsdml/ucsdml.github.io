---
layout: post
mathjax: true
title:  "Capacity Bounded Differential Privacy"
date:   2020-06-02 10:00:00 -0700
categories: jekyll update
tags: adversarial
author: Jacob Imola
excerpt: Capacity Bounded draft
---

This post is about differential privacy (DP), a very popular definition of privacy
that has been adopted on a large scale by Google, Apple, and the U.S. Census Bureau. 
For a brief introduction to the topic, click [here](Google.com). The goal
of differential privacy is to release general data patterns, such 
as a data model, without allowing any adversary to infer with more
than $e^{\varepsilon}$ confidence whether an individual participated in the dataset. We
call this gurantee $\varepsilon$ differential privacy, and the smaller
$\varepsilon$, the stronger the guarantee.
To make things concrete, we will introduce a running example of a sensitive database
called Hospital.

Name | Age | Gender | Zip Code | Smoker | Diagnosis
Richard | 64 | Male | 19146 | Y | Heart Disease
Susan | 61 | Female | 19118 | N | Arthritis
Matthew | 67 | Male | 19104 | Y | Lung Cancer
Alice | 43 | Female | 19146 | N | Chrohn's Disease
Thomas | 69 | Male | 19115 | Y | Lung Cancer

A differentially private algorithm may be able to conclude, after looking at the
dataset, that smoking is correlated to lung cancer. However, a differentially
private algorithm will not allow anyone with a degree of certainty more than
$e^{\varepsilon}$
to conclude that Susan has arthritis or that Susan even participated in the
dataset. This is an extremely strong guarantee. Even if someone with unlimited
computational power knew everything
about Susan except for whether or not she has arthritis, upon seeing the result
of a differentially private algorithm applied to Hospital, their confidence that
Susan participated would not go up or down by more than a factor 
of $e^{\varepsilon}$.

While a powerful tool, differential privacy is not the best definition of
privacy in all situations. There are many reasons for this, but one big
situation in which it is not adequate is
when there is a disparity between the attacker model described above and what
happens in practice. Capacity-bounded differential privacy seeks to address one shortcoming of the
attacker model: the fact that the attacker has
has unlimited computational power and knows everything except for the
sensitive attributes. In real life, adversaries are never this powerful.
Capacity bounded DP makes an assumption that the computational power
of the adversary is limited. Perhaps the only entites who
see the released model are automated and have a certain functional form. Perhaps
they are programmers who are obligated to write code using functions from a specific library.
Capacity-bounded differential privacy gives us the ability to choose which
computational adversaries we want to defend against. It is a statement about the
maximum harm such adversaries can cause.

### Adversarial interpretation of differential privacy:

The standard definition of differential privacy states that there should be no 
way to distinguish a randomized algorithm $A$ when it is run on dataset $X$ from when it
is run on dataset $X'$, where $X$ and $X'$ differ in the data of just one
individual. There are many ways in which we may define what "distinguishing"
means; a very general way is to bound the $f$-divergence between $A(X)$ and
$A(X')$. Specifically, 

<div class="definition">
$A$ provides $(\varepsilon, f)$-differential privacy if for all possible
neighboring datasets $X,X'$, $D_f(A(X), A(X')) \leq \varepsilon $.
</div>

Using specific functions $f$, we can express the two most common definitions of
differential privacy, the original $(\varepsilon, \delta)$ definition~\cite{} 
and Renyi differential privacy~\cite{}.

The inspiration behind capacity-bounded differential privacy comes from the
variational form of the $f$-divergences, which has become widely used in
the $f$-GAN literature~\cite{}. Plugging this representation into the definition
of differential privacy, we have 

<div class="theorem">
$A$ provides $(\varepsilon, f)$-differential privacy if for all possible
neighboring datasets $X,X'$ on a domain $\mathcal{X}$,
$\sup_{h:\mathcal{X} \rightarrow \mathbb{R}} \mathbb{E}_{x \sim A(X)}
[h(x)] - \mathbb{E}_{x \sim A(X')}[f^\star(h(x))]_{} \leq \varepsilon$, where 
$f^\star$ is the Fenchel dual of $f$.
</div>

This gives an alternative interpretation for differential privacy. Suppose there
is an adversary $h$ who wants to cause more harm when the database is $X'$ than when it is $X$, 
but all he can see is an output $x$ drawn from either $A(X)$ or $A(X')$. 
He may process the output as much as he likes and ultimately decides to cause $h(x)$ 
harm. If $x$ came from $A(X)$, then he causes $h(x)$ harm. If $x$ came from
$A(X')$, then he loses $f^\star(h(x))$ harm. 

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
invariance** ensures that no matter how smart the adversary is, if they do not
have access the the private database $X$, they will never be able to worsen the
privacy guarantee. In the formal language of DP, if $A$ provides
$(\varepsilon, f)$-DP, then for any function $M$ which does not have access to
$X$ other than through $A$, $M(A)$ provides $(\varepsilon, f)$-DP.
Capacity-bounded DP provides post-processing invariance with a caveat that
makes sense with a little thought. If $A$ protects against adversaries in
$\mathcal{H}$, and we apply $M$ to it, then we can only guard against those
adversaries $i$ such that $i \circ M \in \mathcal{H}$, as that is what our
privacy guarantee applies to. Formally,

<div class="theorem">
If $A$ provides $(\mathcal{H}, f, \varepsilon)$ capacity-bounded DP and
$\mathcal{I} \circ M \subseteq \mathcal{H}$, then
$M(A)$ provides $(\mathcal{I}, f, \varepsilon)$ capacity-bounded DP.
</div>

An important special case of this property is when $\mathcal{H}$ is closed under
composition and $M \in \mathcal{H}$. Then, $A$ will satisfy $(\mathcal{H}, f,
\varepsilon)$ capacity-bounded DP before and after applying $M$.

The second important property is **adaptive composition**. This ensures that
releasing two differentially private algorithms run on the same database $X$ 
worsens the privacy in a predictable way. The definitions of privacy preceeding 
differential privacy would completely break in this scenario! Formally, if 
$A$ provides $(\varepsilon_1,
f)$-DP and $B$ provides $(\varepsilon_2, f)$-DP, then the algorithm which
releases $(A(X), B(X))$, even if $B$ depends on $A(X)$, provides
$(\varepsilon_1 + \varepsilon_2, f)$-DP. Capacity-bounded DP provides
composition, but not the adaptive case:

<div class="theorem">
If $A$ provides $(\mathcal{H}, f, \varepsilon_1)$ capacity-bounded DP and 
$B$ provides $(\mathcal{H}, f, \varepsilon_2)$ capacity-bounded DP, and $A$ and
$B$ do not depend on each other, then $(A,B)$ provides $(\mathcal{H}, f,
\varepsilon_1 + \varepsilon_2)$ capacity-bounded DP.
</div>

There probably are cases where fully adaptive composition is achievable for
capacity-bounded DP. This is an exciting, open research question!

To see these useful properties in action on our running example, suppose we run a
privacy-preserving algorithm $A$ on Hospital which outputs the average age of
patients for each diagnosis. Suppose we run the Laplace mechanism on ages and
randomized response on patient diagnoses, and then compute the averages. Here is
a sample from $A(Hospital)$:

Age | Diagnosis
62.1 | Heart Disease
64.0 | Arthritis
70.4 | Lung Cancer
43.8 | Chrohn's Disease

The function $A$, being differentially private, also provides 
$(\mathcal{H}^2, f, \varepsilon)$
capacity-bounded DP where $\mathcal{H}^2$ is the class of all
depth 2 decision trees and $f$ is an appropriate function (such as the $f$ that
gives us Renyi divergence). Post-processing for capacity-bounded DP 
says that if $D$ is a depth 1
decision tree, then $D(A)$ satisfies $(\mathcal{H}^1, f, \varepsilon)$
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
capacity-bounded DP is not adaptive, meaning that $A_2$ cannot look at $A_1(Hospital)$ 
during its computation.

### Can bounding the capacity of our adversaries result in meaningful harm reduction?

Because capacity bounded DP assumes that the adversary causing harm is bounded
as opposed to the unlimited adversary assumed by DP, the maximum harm in
capacity-bounded DP must be less than that of DP. Stated another way, if $A$
provides $(f, \varepsilon)$ DP, then it provides $(f, \mathcal{H}, \varepsilon)$
capacity-bounded DP for any $\mathcal{H}$. This follows immediately from the
definition. In this section, we explore cases when making the capacity-bounded
assumption results in provably less harm; namely, we find $A$ that provide $(f,
\varepsilon)$-DP and $(f, \mathcal{H}, \varepsilon')$ capacity-bounded DP for
$\varepsilon' < \varepsilon$.
This is a challenging problem, as it involves analyzing
Definition~\ref{} for general function classes $\mathcal{H}$. Our results apply
when $\mathcal{H} = lin$, the set of linear functions, and $A$ is either of the two
simplest mechanism in differential privacy, the Laplace or Gaussian mechanisms.
For more information on these mechanism, click here. For the sake of this blog
post, it suffices to know that the Laplace mechanism results in $A(X),A(X')$ being two
shifted Laplace distributions in Definition~\ref{} and likewise with Gaussian
distributions for the Gaussian mechanism. Namely, we compute
\\[ \sup_{h \in \mathcal{H}} \mathbb{E}_{x \sim P} [h(x)] - 
      \mathbb{E} {x \sim Q}[f^\star(h(x))]
\\]
where $P,Q$ are either shifted Laplace or Gaussian distributions.

We show the guarantee of $(\varepsilon, f, poly)$ capacity-bounded privacy 
for the Laplace and Gaussian
distributions below, along with the guarantee of standard, or unrestricted, 
$(\varepsilon, f)$ differential privacy. 
Here, $f_{}$ is the function $r_\alpha$ that gives us Renyi divergence.
As we would expect, the potential harm increases as we consider
adversaries with greater polynomial complexity. The poly-bounded
adversaries provide much less potential harm than standard DP for both
mechanisms, particularly for the Gaussian mechanism. One counterintuitive
observation is that for Renyi divergence, higher $\alpha$
means higher divergence. For poly-bounded divergence, this is not the case, as
all the polynomial plots start to decrease with $\alpha$.

{:refdef: style="text-align: center;"}
<img src="/assets/2020-06-02-capacity/laplace_poly_divs.png" width="40%">
<img src="/assets/2020-06-02-capacity/gauss_poly_divs.png" width="40%">
{:refdef:}

In the paper, we prove that the maximum harm a
linear-bounded adversary can cause is a logarithmic function of the harm that
the worst-case adversary can cause. This is a promising result that has the
following additional implication: if we have a fixed, maximum tolerance for harm, 
then assuming the adversaries are linearly-bounded allows us to add
logarithmic noise in the Laplace or Gaussian mechanism compared to standard
differential privacy. Since noise reduces the utility of our data analysis, this
can result in a massive utility gain.

### Conclusion

Capacity-bounded differential privacy enhances the original definition of
differential privacy by assuming that the adversaries who have access to
released data are computationally bounded. Capacity-bounded DP satisfies two
very important properties that have led differential privacy to become a gold
standard for privacy---post-processing and function composition. Finally, there
are examples where making this capacity-bounded assumption on our adversaries
results in a large risk reduction over the unrestricted scenario of standard DP.
See [our paper on arxiv](https://arxiv.org/abs/1907.02159) for more information!
