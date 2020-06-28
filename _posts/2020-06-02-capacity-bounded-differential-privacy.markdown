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
privacy model: that differential privacy prevents all adversaries from
inferring the participation of an individual in the dataset. In practice,
adversaries are never computationally unbounded. Perhaps the only entites who
see the released model are automated and have a certain functional form. Perhaps
they are programmers who write code using functions from a specific library.
Capacity-bounded differential privacy gives us the ability to choose which
adversaries we want to defend against.

### Adversarial interpretation of differential privacy:

### More Details

See [our paper on arxiv](https://arxiv.org/abs/1907.02159).
