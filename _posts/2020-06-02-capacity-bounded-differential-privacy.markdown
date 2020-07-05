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

### Introduction
This post is about differential privacy, the gold standard privacy definition
that has been adopted on a large scale by Google, Apple, and the U.S. Census Bureau. 
For a brief introduction to the topic, click [here](Google.com). The goal
of $\varepsilon$ differential privacy is to release general data patterns, such 
as a data model, without allowing any adversary to infer with certaintly more
than $e^{\varepsilon}$ whether an individual participated in the dataset.
To make things concrete, we will introduce a running example of a sensitive database
called Hospital.

Name | Age | Gender | Zip Code | Smoker | Diagnosis
Richard | 64 | Male | 19146 | Y | Heart Disease
Susan | 61 | Female | 19118 | N | Arthritis
Matthew | 67 | Male | 19104 | Y | Lung Cancer
Alice | 63 | Female | 19146 | N | Chrohn's disease
Thomas | 69 | Male | 19115 | Y | Lung Cancer

A differentially private algorithm may be able to conclude, after looking at the
dataset, that smoking is correlated to lung cancer. However, a differentially
private algorithm will not allow anyone with a degree of certainty more than
$e^{\varepsilon}$
to conclude that Susan has lung cancer or that Susan even participated in the
dataset.

Differential privacy satisfies important properties which make it amenable to
the design of private algorithms. Post-processing invariance allows a data
practitioner to transform the result of an $\varepsilon$-DP algorithm however he pleases
with the assurance that releasing the transformation is still $\varepsilon$-DP.
Sequential composition allows the data practitioner to run two $\varepsilon$-DP
algorithms, one possibly depending on the output of the other, with the
assurance that releasing both outputs is $2\varepsilon$-DP.

While a powerful tool, differential privacy is not the best definition of
privacy in all situations. There are many reasons for this, but most of the time
its inadequacy results from a disparity between the privacy model and what
happens in practice. Capacity-bounded differential privacy seeks to address one shortcoming of the
privacy model: differential privacy is equivalent to bounding the risk that
befalls an individual by participating in a dataset against the worst-case
adversary, meaning one which has unlimited access to all side informaion and
unlimited computational power. In practice,
adversaries are never unbounded like this. Perhaps the only entites who
see the released model are automated and have a certain functional form. Perhaps
they are programmers who write code using functions from a specific library.
Capacity-bounded differential privacy gives us the ability to choose which
adversaries we want to defend against, and bound the maximum damage that these
adversaries can cause.

### Adversarial interpretation of differential privacy:

The standard definition of differential privacy states that there should be no 
way to distinguish a randomized algorithm $A$ when it is run on dataset $X$ from when it
is run on dataset $X'$, where $X$ and $X'$ differ in the data of just one
individual. There are many ways in which we may define what "distinguishing"
means; a very general way is to bound the $f$-divergence between $A(X)$ and
$A(X')$. Specifically, $A$ satisfies $(\varepsilon, f)$-differential privacy if
$\sup_{X,X'} D_f(A(X), A(X')) \leq \varepsilon $.
Using specific functions $f$, we can express the two most common definitions of
differential privacy, the original $(\varepsilon, \delta)$ definition~\cite{} 
and Renyi differential privacy~\cite{}.

The inspiration behind capacity-bounded differential privacy comes from an
alternative representation for $f$-divergences which has become widespread in
the $f$-GAN literature~\cite{}. Plugging this representation into our definition
of differential privacy, we have 

<div class="definition">
$A$ satisfies $(\varepsilon, f)$-differential privacy if 
$\sup_{X,X'} \sup_{h:\mathcal{X} \rightarrow \mathbb{R}} \mathbb{E}_{x \sim A(X)}
[h(x)] - \mathbb{E}_{x \sim A(X')}[f^*(h(x))] \leq \varepsilon$, where $f^*$ is the
Fenchel dual of $f$.
</div>

This gives an alternative interpretation for differential privacy. Suppose there
is an adversary $h$ who wants to cause more harm when the database is $X'$ than when it is $X$, 
but all he can see is an output $x$ which is either drawns from $A(X)$ or $A(X')$. 
He may process the output as much as he likes and ultimately decides to cause $h(x)$ 
harm. If $x$ came from $A(X)$, then he causes $h(x)$ harm. If $x$ came from
$A(X')$, then he loses $f^*(h(x))$ harm. 

Viewing differential privacy through this adversarial interpretation, it is
strightforward to make an assumption on the capacity of the adversary. Instead
of letting $h$ be any function, we restrict it to a class $\mathcal{H}$. This
gives us the following definition:

<div class="definition">
$A$ satisfies $(\mathcal{H}, \varepsilon, f)$-differential privacy if 
$\sup_{X,X'} \sup_{h \in \mathcal{H}} \mathbb{E}_{x \sim A(X)}
[h(x)] - \mathbb{E}_{x \sim A(X')}[f^*(h(x))] \leq \varepsilon$, where $f^*$ is the
Fenchel dual of $f$.
</div>


### Does Cap. Bounded DP satisfy the same important properties as differential privacy?

The reason differential privacy is such a great definition of privacy is that it
behaves well in the presence of side information. Post-processing
invariance ensures that no matter how smart the adversary is, if they do not
have access the the private database $X$, they will never be able to worsen the
privacy guarantee. In the language we used above, if $A$ satisfies
$(\varepsilon, f)$-DP, then for any function $M$ which does not have access to
$X$ other than through $A$, $M(A)$ satisfies $(\varepsilon, f)$-DP.
Capacity-bounded DP satisfies this as well, with a caveat. If $M$ is a more
complicated function than the functions in $\mathcal{H}$, then the adversaries
who then analyze $M(A)$ becomes stronger. Formally,

<div class="definition">
If $A$ satisfies $(\mathcal{H}, f, \varepsilon)$ capacity-bounded DP, then
$M(A)$ satisfies $(\mathcal{H} \circ M, f, \varepsilon)$ capacity-bounded DP.
</div>

Our second important property is adaptive composition. This ensures that
releasing two differentially private algorithms run on the same database $X$ 
worsens the privacy in a predictable way. Previous definitions of privacy would
completely break in this scenario! Formally, if $A$ satisfies $(\varepsilon_1,
f)$-DP and $B$ satisfies $(\varepsilon_2, f)$-DP, then the algorithm which
releases $(A(X), B(X))$, even if $B$ depends on $A(X)$, satisfies
$(\varepsilon_1 + \varepsilon_2, f)$-DP. Capacity-bounded DP satisfies
composition, but not the adaptive case.

<div class="definition">
If $A$ satisfies $(\mathcal{H}, f, \varepsilon_1)$ capacity-bounded DP and 
$B$ satisfies $(\mathcal{H}, f, \varepsilon_2)$ capacity-bounded DP, and $A$ and
$B$ do not depend on each other, then $(A,B)$ satisfies $(\mathcal{H}, f,
\varepsilon_1 + \varepsilon_2)$ capacity-bounded DP.
</div>

There probably are cases where fully adaptive composition is achievable for
capacity-bounded DP. This is an exciting, open research question!

### Can bounding the capacity of our adversaries result in meaningful risk reduction?

### More Details

See [our paper on arxiv](https://arxiv.org/abs/1907.02159).
