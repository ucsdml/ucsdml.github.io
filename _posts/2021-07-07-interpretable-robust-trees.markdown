---
layout: post
mathjax: true
title: "Connecting Interpretability and Robustness in Decision Trees through Separation"
date: 2021-07-07 10:00:00 -0700
categories: jekyll update
tags: adversarial non-parametric robustness
paper_url: https://arxiv.org/abs/2102.07048
code_url: https://github.com/yangarbiter/interpretable-robust-trees
excerpt: We are constructing a tree-based model that isguaranteedto be adversarially robust, inter-pretable, and accurate.
author: <a href='https://sites.google.com/view/michal-moshkovitz'>Michal Moshkovitz</a> and <a href='http://yyyang.me'>Yao-Yuan Yang</a>
---


**TL;DR** We are constructing a tree-based model that is <inc>guaranteed</inc>
to be adversarially robust, interpretable, and accurate.


Imagine a world where computers are making decisions and are integrated into our lives.
No need to worry about overly executed doctors making life-changing decisions or
driving your car after a long day at the office.
Sounds great, right? Well, what if those computers weren’t reliable? What if a
computer decided you need to go through surgery without telling you why?
What if a car confused a child with a green light? It doesn’t sound so great
after all.

Machine learning models have to be reliable before we allow them to fully enter
our lives. Two cornerstones for reliable machine learning are explainability,
where the model’s decisions are transparent, and robustness, where small changes
to the input don’t change the model’s answer. Unfortunately, these properties
are generally studied (i) in isolation or (ii) only empirically.
In our [recent paper](https://arxiv.org/abs/2102.07048), we explore
interpretability and robustness <ins>simultaneously</ins>, and we examine it
<ins>both theoretically and empirically</ins>.

## What do we mean by interpretability and robustness?

When different people say an "explainable model" or "robust model" they can
mean different things. In this section, we explain what we mean by
explainability and robustness.

### Interpretability

A model is __interpretable__ if the model is simple and self-explainable.
There are several forms of
[self-explanatory models](https://christophm.github.io/interpretable-ml-book/simple.html),
e.g., decision sets, logistic regression, and decision rules.
One of the most fundamental interpretable models, which we focus on here, is
**small** decision trees.
In decision trees, each inner node corresponds to a threshold and a
feature and each leaf correspond to a label.
The label of an example is the leaf’s label of the corresponding path.
We focus on binary classifications, where the label can get one of two options.

In [our paper](https://arxiv.org/abs/2102.07048), we construct a specific
kind of decision tree ---
[risk scores](https://jmlr.org/papers/volume20/18-615/18-615.pdf), see Table 1
and check Table 2 to see how to reduce a risk score to a decision tree.
A risk score is composed of several conditions (e.g., $age \geq 75$) and each
matched with a weight, i.e., a small integer.
A score $s(x)$ of an example $x$ is the weighted sum of all the satisfied
conditions. The label is then $sign(s(x))$.
The number of parameters required to represent a risk score is much smaller than
their corresponding decision trees, hence they might be considered
[more interpretable than decision trees](https://jmlr.org/papers/volume20/18-615/18-615.pdf).

<div style="width: 100%; overflow-x: auto; margin-bottom: 40pt">
<table style="width: 80%; font-size: 80%; margin: auto;" class="concise-table">
  <caption>
	Two risk score models:
	<a href='https://jmlr.org/papers/volume20/18-615/18-615.pdf'>LCPA</a> and our new BBM-RS algorithm on the
	<a href='https://core.ac.uk/download/pdf/55631291.pdf'>bank dataset</a>.
	Each satisfied condition is multiplied by its weight and summed. Bias term is always satisfied. 
	If the total score $>0$, the risk model predicts "1" (i.e., the client will open a bank account after a marketing call).
    All features are binary (either $0$ or $1$).
  </caption>
  <tr>
    <th colspan="1">features</th>
    <th colspan="3" style="text-align: center;">weights</th>
  </tr>
  <tr>
    <th colspan="1"></th>
    <th colspan="1" style="text-align: center;">LCPA</th>
    <th colspan="1" style="text-align: center;">BBM-RS</th>
    <th colspan="1"></th>
  </tr>
  <tr>
   <th>Bias term</th> <th style="text-align: center;">-6</th> <th style="text-align: center;">-7</th> <th> + ... </th>
  </tr>
  <tr>
   <th>Age $\geq$ 75</th> <th style="text-align: center;">-</th> <th style="text-align: center;">2</th> <th> + ... </th>
  </tr>
  <tr>
   <th>Called in Q1</th> <th style="text-align: center;">1</th> <th style="text-align: center;">2</th> <th> + ... </th>
  </tr>
  <tr>
   <th>Called in Q2</th> <th style="text-align: center;">-1</th> <th style="text-align: center;">-</th> <th> + ... </th>
  </tr>
  <tr>
   <th>Called before</th> <th style="text-align: center;">1</th> <th style="text-align: center;">4</th> <th> + ... </th>
  </tr>
  <tr>
   <th>Previous call was successful</th> <th style="text-align: center;">1</th> <th style="text-align: center;">2</th> <th> + ... </th>
  </tr>
  <tr>
   <th>Employment variation rate $<-1$</th> <th style="text-align: center;">5</th> <th style="text-align: center;">4</th> <th> + ... </th>
  </tr>
  <tr>
   <th>Consumer price index $\geq 93.5$</th> <th style="text-align: center;">1</th> <th style="text-align: center;">-</th> <th> + ... </th>
  </tr>
  <tr>
   <th>3 month euribor rate $\geq 200$</th> <th style="text-align: center;">-2</th> <th style="text-align: center;">-</th> <th> + ... </th>
  </tr>
  <tr>
   <th>3 month euribor rate $\geq 400$</th> <th style="text-align: center;">5</th> <th style="text-align: center;">-</th> <th> + ... </th>
  </tr>
  <tr>
   <th>3 month euribor rate $\geq 400$</th> <th style="text-align: center;">2</th> <th style="text-align: center;">-</th> <th> + ... </th>
  </tr>
  <tr>
   <th></th> <th> </th> <th>total scores=</th> <th> </th>
  </tr>
</table>
</div>

Here is an example of how to convert a risk score to a decision tree.
The table on the left is a risk score with three conditions and the the figure on the right is the corresponding decision tree.
For each node in the tree, the branch towards right represents the path to go if the condition is true.
The leaves represents the final risk score of the given condition.

<div style="width: 100%; overflow-x: auto;">
  <span style="width: 50%; overflow-x: auto; display: inline-block; margin: auto; float: left;">
    <table style="width: 100%; font-size: 80%; margin: 0;" class="concise-table">
      <tr>
        <th colspan="2" style="text-align: center;">features</th>
        <th colspan="1"></th>
      </tr>
      <tr><th>Bias term</th> <th style="text-align: center;">-3</th> <th> + ... </th></tr>
      <tr><th>Condition 1</th> <th style="text-align: center;">3</th> <th> + ... </th></tr>
      <tr><th>Condition 2</th> <th style="text-align: center;">1</th> <th> + ... </th></tr>
      <tr><th>Condition 3</th> <th style="text-align: center;">2</th> <th> + ... </th></tr>
      <tr><th></th> <th>total scores=</th> <th></th></tr>
    </table>
  </span>
  <span style="width: 48%; overflow-x: auto; display: inline-block; margin: auto; float: right;">
    <img src='/assets/2021-07-07-interpretable-robust-trees/risk_score_tree.png' style="width: 100%">
  </span>
</div>

### Robustness
We want our model to be robust to adversarial perturbation.
This means that if an example $x$ is changed, by a bit, to $x'$, the model's
answer remains the same.
By "a bit", we mean that $x'=x+\delta$ where $\|\delta\|\_\infty\leq r$ is
small. A model $h:\mathbf{X} \rightarrow \\{-1, +1\\}$ is <inc>robust</inc> at $x$ with radius
$r$ if for all $x'$ we have that $h(x)=h(x')$. The notion of <inc>astuteness</inc>
[was previously introduced](http://proceedings.mlr.press/v80/wang18c.html)
to jointly measure the robustness and the accuracy of a model.
The astuteness of a model $h$ at radius $r > 0$ under a distribution $\mu$
is \\[\Pr_{(x,y)\sim\mu}[\forall x'. \|x-x'\|\_\infty\leq r.\; h(x')=y].\\]

## Data properties

### $r$-Separation

We want models that are robust,  interpretable, and have high-accuracy. Without
any assumptions on the data we cannot guarantee all three to hold
simultaneously. For example, if the true labels of the examples are different
for close examples, a model cannot have high accuracy and be robust. To
circumvent this problem,
[a prior work](http://proceedings.mlr.press/v80/wang18c.html) suggested
to focus on datasets that satisfy $r$-separation.

A binary labeled data is <inc>$r$-separated </inc> if every two differently labeled examples,
$(x^1,+1)$,$(x^2,-1)$, are far apart,
$\|x^1-x^2\|\_\infty\geq 2r.$
[Yang et. al.](https://arxiv.org/abs/2003.02460) showed that
$r$-separation is sufficient for robust learning. We examine whether it is
also sufficient for interpretability.
We discovered that the answer is YES and NO.
YES - because there is a high-accuracy decision tree with size independent
of the number of examples, and the depth of the tree is linear to the number
of features and $r$.
So we can efficiently explain each leaf by going through the path from root
the that leaf.
NO - because the size of the tree can be exponential in the number of
features.
We showed some $r$-separated data and proved that the tree-size must be
exponential in $d$ to achieve accuracy better than random.

### Linear separation

Next, we investigate a stronger assumption --- linear separation with a
$\gamma$-margin, where there exists a vector $w$ with $\|w\|\_1=1$ such that
$ywx\geq \gamma.$ Linear separation is a popular assumption in the research of
machine learning models, e.g., for
[support vector machines](https://en.wikipedia.org/wiki/Support-vector_machine),
[neural networks](https://arxiv.org/abs/1705.08292),
and [decision trees](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.129.6343&rep=rep1&type=pdf).


Using a generalization of
[previous work](https://www.cs.huji.ac.il/~shais/papers/ShalevSi08.pdf)
we know that under the linear separation assumption, there has to be a feature
that gives nontrivial information. Or in different words, decision stumps are a
weak learner.
More formally, for each sample $S=((x^1,y^1),\ldots,(x^m,y^m))$, we focus on
the best decision stump for this sample $h_S(x)=sign(x_i-\theta)$ where $i$ is
a  feature  and  $\theta$ is a threshold that minimize the training error
$\sum_{j=1}^m sign(x^j_i\geq\theta) y^j.$ We are able to prove that $h_S$ has
accuracy better than $0.5$, i.e., better than a random guess:

<div class="theorem">
  Fix $\alpha>0$.
  For any distribution $\mu$ over  $[-1,+1]^d\times\{-1,+1\}$ that satisfies
  linear separability with a $\gamma$-margin, and for any $\delta\in(0,1)$ there
  is $m=O\left(\frac{d+\log\frac1\delta}{\gamma^2}\right)$, such that with
  probability at least $1-\delta$ over the sample $S$ of size $m$, it holds that
  $$\Pr_{(x,y)\sim\mu}(h_S(x)=y)\geq \frac12+\frac{\gamma}{4}-\alpha.$$
</div>

We showed that decision stumps are a weak learner under the linear separability
assumption.
So why not use a boosting method to learn the entire interpretable decision
tree? Kearns and Mansour showed that if at each node we have a weak learner then
the famous algorithm for learning decision tree,
[ID3](https://link.springer.com/content/pdf/10.1007/BF00116251.pdf), works.
As a side benefit, this is the <inc>first</inc> time that a distributional
assumption that does not include feature independence is used.
[Previously](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.129.6343&rep=rep1&type=pdf),
[many](https://arxiv.org/abs/1911.07375)
[papers](http://proceedings.mlr.press/v125/brutzkus20a/brutzkus20a.pdf)
assumed uniformity or feature independence.

Are we done? Is this model also robust?

## New algorithm: BBM-RS

Designing robust decision trees is inherently a difficult task.
The reason is that, generally, the model defined by the right and left subtrees
can be completely different. The feature $i$ in the root determines if the model
uses the right or left subtrees.
Thus, a small change in the $i$-th feature completely changes the model.
To overcome this difficulty we focus on a specific type of decision trees, risk
scores, which were mentioned at the beginning of the post.
**Note that** in the decision tree that corresponds to the risk score, the right
and left subtrees are the same.

We design a new algorithm for learning risk scores by utilizing the known
boosting method
\href{https://link.springer.com/content/pdf/10.1023/A:1010852229904.pdf}{boost-by-majority}
(BBM). The different conditions are added to the risk score one by one, using
the weak learner. BBM has the benefit of ensuring the weights in the risk score
are small integers.  This will lead to an interpretable model with size only
$O(\gamma^{-2}\log1/\epsilon)$ where the model has accuracy $1-\epsilon$. See
[our paper](https://arxiv.org/abs/2102.07048) for the pseudocode.

Now we want to make sure that the risk model is also robust. The idea is to add
noise. We take each point in the sample and just make sure that it's a little
bit closer to the decision boundary, see the figure below.
{:refdef: style="text-align: center;"}
  <img src='/assets/2021-07-07-interpretable-robust-trees/BBM_RS_add_noise.png'>
{:refdef}
The idea is that if the model is correct for the noisy point, then it's
defiantly correct for the point without the noise. To formally prove it, we
show that choosing the risk-score conditions in a specific way ensures that they
are monotone models.
In such models, adding noise in the way we described is
sufficient for robustness (more details are in [our
paper](https://arxiv.org/abs/2102.07048)).

To summarize, we designed a new algorithm that is robust, interpretable, and
have high-accuracy see the pseudocode below and the formal theorem next.

<div class="theorem">
Suppose data is $\gamma$-linearly separable and fix $\epsilon,\delta\in(0,1)$.
Then, with probability $1-\delta$ the output of BBM-RS, after receiving
$(d+\log(1/\delta))\log(1/\epsilon)\gamma^{-O(1)}$ samples, has astuteness
$1-\epsilon$ at radius $\gamma/2$ and has  $O(\gamma^{-2}\log(1/\epsilon))$
feature-threshold pairs.
</div>

### Performance on real data
Previously, we showed that BBM-RS is robust and interpretable on linearly
separable data.
Now, let's see how it performs on real datasets, which may not be perfectly
linearly separable (here we present the result for four datasets, for other
datasets, please refer to [our paper](https://arxiv.org/abs/2102.07048)).

First, we look at how  non-linearly separable these real datasets are.
We measure the linear separateness by training a linear SVM with different
regularization parameters and record the best accuracy. From the next table, we
see that there are datasets that are
very or moderately close to being perfectly linearly separated.
This shows that the assumption of our theorem may still be useful
for many datasets.

<div style="width: 100%; overflow-x: auto;">
<table style="width: 50%; font-size: 80%; margin: auto" class="concise-table">
  <tr>
    <th colspan="1"></th> <th colspan="1" style="text-align: center;">linear separateness</th>
  </tr>
  <tr>
    <th colspan="1">adult</th> <th colspan="1" style="text-align: center;">0.84</th>
  </tr>
  <tr>
    <th colspan="1">breastcancer</th> <th colspan="1" style="text-align: center;">0.97</th>
  </tr>
  <tr>
    <th colspan="1">diabetes</th> <th colspan="1" style="text-align: center;">0.77</th>
  </tr>
  <tr>
    <th colspan="1">heart</th> <th colspan="1" style="text-align: center;">0.89</th>
  </tr>
</table>
</div>

To see the performance of BBM-RS, we compare it to three baselines,
[LCPA](https://arxiv.org/abs/1610.00168),
[decision tree (DT)](https://books.google.co.il/books?hl=en&lr=&id=MGlQDwAAQBAJ&oi=fnd&pg=PP1&ots=gBmdjTJVdK&sig=\_jUBiPW4cTS7JYUKpzKcJLYipl4&redir_esc=y#v=onepage&q&f=false), and
[robust decision tree (RobDT)](http://proceedings.mlr.press/v97/chen19m/chen19m.pdf).
We measure a model's robustness by evaluating its
[__Empirical robustness (ER)__](https://arxiv.org/abs/2003.02460), which is the
$\ell_\infty$
distance to the closest adversarial example.
The larger ER is, the more robust the classifier is.
We measure a model's interpretability by evaluating its
__Interpretation Complexity (IC)__.
We measure IC with the number of unique feature-thresholds pairs in the
model (this corresponds to the number of conditions in the risk score).
The smaller IC is, the more interpretable the classifier is.

From the tables, we see that BBM-RS have a test accuracy comparable to other
methods.
In terms of robustness, it performs slightly better than others (performing the
best on three datasets among a total of four).
In terms of interpretability, BBM-RS
performs the best among three out of four datasets.
All in all, we see that BBM-RS can bring better robustness and interpretability
(depending on the measurement) while perform competitively on test accuracy.
This shows that BBM-RS not only performs well theoretically, it also performs
well empirically.

<div style="width: 100%; overflow-x: auto;">
<table style="width: 60%; font-size: 80%; margin: auto" class="concise-table">
  <tr>
    <th colspan="1"></th>
    <th colspan="4" style="text-align: center;">test accuracy (higher=better)</th>
  </tr>
  <tr>
    <th colspan="1"></th>
    <th colspan="1" style="text-align: center;">DT</th>
    <th colspan="1" style="text-align: center;">RobDT</th>
    <th colspan="1" style="text-align: center;">LCPA</th>
    <th colspan="1" style="text-align: center;">BBM-RS</th>
  </tr>
  <tr>
    <th colspan="1">adult</th>
    <th colspan="1" style="text-align: center; color: green;"><b>0.83</b></th>
    <th colspan="1" style="text-align: center; color: green;"><b>0.83</b></th>
    <th colspan="1" style="text-align: center;">0.82</th>
    <th colspan="1" style="text-align: center;">0.81</th>
  </tr>
  <tr>
    <th colspan="1">breastcancer</th>
    <th colspan="1" style="text-align: center;">0.94</th>
    <th colspan="1" style="text-align: center;">0.94</th>
    <th colspan="1" style="text-align: center; color: green;"><b>0.96</b></th>
    <th colspan="1" style="text-align: center; color: green;"><b>0.96</b></th>
  </tr>
  <tr>
    <th colspan="1">diabetes</th>
    <th colspan="1" style="text-align: center;">0.74</th>
    <th colspan="1" style="text-align: center;">0.73</th>
    <th colspan="1" style="text-align: center; color: green;"><b>0.76</b></th>
    <th colspan="1" style="text-align: center;">0.65</th>
  </tr>
  <tr>
    <th colspan="1">heart</th>
    <th colspan="1" style="text-align: center;">0.76</th>
    <th colspan="1" style="text-align: center;">0.79</th>
    <th colspan="1" style="text-align: center; color: green;"><b>0.82</b></th>
    <th colspan="1" style="text-align: center; color: green;"><b>0.82</b></th>
  </tr>
</table>
</div>

<div style="width: 100%; overflow-x: auto;">
<table style="width: 60%; font-size: 80%; margin: auto" class="concise-table">
  <tr>
    <th colspan="1"></th>
    <th colspan="4" style="text-align: center;">ER (higher=better)</th>
  </tr>
  <tr>
    <th colspan="1"></th>
    <th colspan="1" style="text-align: center;">DT</th>
    <th colspan="1" style="text-align: center;">RobDT</th>
    <th colspan="1" style="text-align: center;">LCPA</th>
    <th colspan="1" style="text-align: center;">BBM-RS</th>
  </tr>
  <tr>
    <th colspan="1">adult</th>
    <th colspan="1" style="text-align: center; color: green;"><b>0.50</b></th>
    <th colspan="1" style="text-align: center; color: green;"><b>0.50</b></th>
    <th colspan="1" style="text-align: center;">0.12</th>
    <th colspan="1" style="text-align: center; color: green;"><b>0.50</b></th>
  </tr>
  <tr>
    <th colspan="1">breastcancer</th>
    <th colspan="1" style="text-align: center;">0.23</th>
    <th colspan="1" style="text-align: center; color: green;"><b>0.29</b></th>
    <th colspan="1" style="text-align: center;">0.28</th>
    <th colspan="1" style="text-align: center;">0.27</th>
  </tr>
  <tr>
    <th colspan="1">diabetes</th>
    <th colspan="1" style="text-align: center;">0.08</th>
    <th colspan="1" style="text-align: center;">0.08</th>
    <th colspan="1" style="text-align: center;">0.09</th>
    <th colspan="1" style="text-align: center; color: green;"><b>0.15</b></th>
  </tr>
  <tr>
    <th colspan="1">heart</th>
    <th colspan="1" style="text-align: center;">0.23</th>
    <th colspan="1" style="text-align: center;">0.31</th>
    <th colspan="1" style="text-align: center;">0.14</th>
    <th colspan="1" style="text-align: center; color: green;"><b>0.32</b></th>
  </tr>
</table>
</div>

<div style="width: 100%; overflow-x: auto;">
<table style="width: 60%; font-size: 80%; margin: auto" class="concise-table">
  <tr>
    <th colspan="1"></th>
    <th colspan="4" style="text-align: center;">IC feature threshold pairs (lower=better)</th>
  </tr>
  <tr>
    <th colspan="1"></th>
    <th colspan="1" style="text-align: center;">DT</th>
    <th colspan="1" style="text-align: center;">RobDT</th>
    <th colspan="1" style="text-align: center;">LCPA</th>
    <th colspan="1" style="text-align: center;">BBM-RS</th>
  </tr>
  <tr>
    <th colspan="1">adult</th>
    <th colspan="1" style="text-align: center;">414.20</th>
    <th colspan="1" style="text-align: center;">287.90</th>
    <th colspan="1" style="text-align: center;">14.90</th>
    <th colspan="1" style="text-align: center; color: green;"><b>6.00</b></th>
  </tr>
  <tr>
    <th colspan="1">breastcancer</th>
    <th colspan="1" style="text-align: center;">15.20</th>
    <th colspan="1" style="text-align: center;">7.40</th>
    <th colspan="1" style="text-align: center; color: green;"><b>6.00</b></th>
    <th colspan="1" style="text-align: center;">11.00</th>
  </tr>
  <tr>
    <th colspan="1">diabetes</th>
    <th colspan="1" style="text-align: center;">31.20</th>
    <th colspan="1" style="text-align: center;">27.90</th>
    <th colspan="1" style="text-align: center;">6.00</th>
    <th colspan="1" style="text-align: center; color: green;"><b>2.10</b></th>
  </tr>
  <tr>
    <th colspan="1">heart</th>
    <th colspan="1" style="text-align: center;">20.30</th>
    <th colspan="1" style="text-align: center;">13.60</th>
    <th colspan="1" style="text-align: center;">11.90</th>
    <th colspan="1" style="text-align: center; color: green;"><b>9.50</b></th>
  </tr>
</table>
</div>

## Conclusion

In conclusion, we investigated three important properties of a classifier,
accuracy, robustness, and interpretability. We designed and analyzed an
<ins>efficient</ins> tree-based algorithm that provably achieves all these
properties, under linear separation with a margin assumption. Our research is a
step towards building trustworthy models that provably achieve many desired
properties.

Our research raises many open problems.
What is the optimal dependence between accuracy, IC, ER, and sample complexity?
Can we have guarantees using different notions of interpretability?
We showed how to construct an interpretable, robust, and accurate model. But,
for reliable machine learning models, many more properties are required,
such as privacy and fairness.
Can we build a model with guarantees on all these properties simultaneously?

#### More Details

See [our paper on arxiv](https://arxiv.org/abs/2102.07048) or [our repository](https://github.com/yangarbiter/interpretable-robust-trees).
