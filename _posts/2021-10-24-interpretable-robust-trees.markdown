---
layout: post
mathjax: true
title: "Connecting Interpretability and Robustness in Decision Trees through Separation"
date: 2021-10-24 10:00:00 -0700
categories: jekyll update
tags: adversarial non-parametric robustness
paper_url: https://arxiv.org/abs/2102.07048
code_url: https://github.com/yangarbiter/interpretable-robust-trees
excerpt: Trustworthy machine learning (ML) has emerged as a crucial topic for the success of ML models. This post focuses on three fundamental properties of trustworthy ML models -- high accuracy, interpretability, and robustness. Building on ideas from ensemble learning, we construct a tree-based model that is guaranteed to be adversely robust, interpretable, and accurate on linearly separable data. Experiments confirm that our algorithm yields classifiers that are both interpretable, robust, and have high accuracy. 
author: <a href='https://sites.google.com/view/michal-moshkovitz'>Michal Moshkovitz</a> and <a href='http://yyyang.me'>Yao-Yuan Yang</a>
---


**TL;DR** We construct a tree-based model that is <inc>guaranteed</inc>
to be adversarially robust, interpretable, and accurate.

Imagine a world where computers are fully integrated into
our everyday lives. Making decisions independently, without
human intervention. No need to worry about overly exhausted
doctors making life-changing decisions or driving your car
after a long day at the office. Sounds great, right? Well,
what if those computers weren’t reliable? What if a
computer decided you need to go through surgery without
telling you why? What if a car confused a child with a
green light? It doesn’t sound so great after all.

Before we fully embrace machine learning, it needs to be reliable.
The cornerstones for reliable machine learning are (i) interpretability,
where the model’s decisions are transparent, and (ii) robustness, where small changes
to the input do not change the model’s prediction.
Unfortunately, these properties are generally studied in isolation or only empirically.
Here, we explore interpretability and robustness <ins>simultaneously</ins>,
and examine it <ins>both theoretically and empirically</ins>.

<!--In this post, our objective is to build a decision tree with guarantees
on its accuracy, robustness, and interpretability.-->
We start this post by explaining what we mean by interpretability and robustness.
Next, to derive guarantees, we need some assumptions on the data.
We start with the known [$r$-separated data](http://proceedings.mlr.press/v80/wang18c.html).
We show that although there exists a tree that is accurate and robust,
such tree can be exponentially large, which makes it not interpretable.
To improve the guarantees, we make a stronger assumption on the data
and focus on linearly separable data.
We design an algorithm called BBM-RS and prove that it is accurate, robust, and interpretable on
linearly separable data.
Lastly, real datasets may not be linearly separable, so to understand how BBM-RS performs in practice,
we conduct an empirical study on $13$ datasets.
We find out that BBM-RS brings better robustness and interpretability while performing competitively
on test accuracy.

## What do we mean by interpretability and robustness?

### Interpretability

A model is __interpretable__ if the model is simple and self-explanatory.
There are several forms of
[self-explanatory models](https://christophm.github.io/interpretable-ml-book/simple.html),
e.g., [decision sets](https://www-cs-faculty.stanford.edu/people/jure/pubs/interpretable-kdd16.pdf),
[logistic regression](https://en.wikipedia.org/wiki/Logistic_regression), and
[decision rules](https://christophm.github.io/interpretable-ml-book/rules.html).
One of the most fundamental interpretable models, which we focus on here, are
**small** decision trees.
We use the size of a tree to determine whether it is interpretable or not.

### Robustness
We also want our model to be robust to adversarial perturbations.
This means that if example $x$ is changed, by a bit, to $x'$, the model's
answer remains the same.
By "a bit", we mean that $x'=x+\delta$ where $\\|\delta\\|\_\infty\leq r$ is
small. A model $h:\mathbf{X} \rightarrow \\{-1, +1\\}$ is <inc>robust</inc> at $x$ with radius
$r$ if for all such $x'$ we have that $h(x)=h(x')$. The notion of <inc>astuteness</inc>
[was previously introduced](http://proceedings.mlr.press/v80/wang18c.html)
to jointly measure the robustness and the accuracy of a model.
The astuteness of a model $h$ at radius $r > 0$ under a distribution $\mu$
is \\[\Pr_{(x,y)\sim\mu}[h(x')=y \ |\  \forall x' \text{ with } \\|x-x'\\|\_\infty\leq r].\\]

## Guarantees under different data assumptions

Without any assumptions on the data, we cannot guarantee
accuracy, interpretability, and robustness to hold simultaneously.
For example, if the true labels of the examples are different for close
examples, a model cannot be astute (accurate and robust).
In this section, we explore which data properties are sufficient for astuteness and interpretability.

### $r$-Separation

[A prior work](http://proceedings.mlr.press/v80/wang18c.html) suggested
focusing on datasets that satisfy $r$-separation.
A binary labeled data distribution is <inc>$r$-separated </inc> if every two differently labeled examples, $(x^1,+1)$,$(x^2,-1)$, are far apart,
$\\|x^1-x^2\\|\_\infty\geq 2r.$
[Yang et al.](https://arxiv.org/abs/2003.02460) showed that
$r$-separation is sufficient for robust learning.
Therefore, we examine whether it is also sufficient for accuracy and
interpretability.
We have two main findings.
First, we found that there is a accurate decision tree with size
independent of the number of examples.
Second, we discovered that the size of the accurate tree can be exponential
in the number of features.
Combining these two findings, it appears we need to find a stronger assumption on the data to
be able to have guarantees on both accuracy and interpretability.

### Linear separation

Next, we investigate a stronger assumption --- linear separation with a
$\gamma$-margin.
Intuitively, it means that a hyperplane separates the two labels in the data,
and the margin (distance of the closest point to the hyperplane) is at
least $\gamma$ (larger $\gamma$ means larger margin for the classifier).
More formally, there exists a vector $w$ with $\\|w\\|\_1=1$ such that
for each training example and its label $(x, y)$, we have
$ywx\geq \gamma$.
Linear separation is a popular assumption in the research of
machine learning models, e.g., for
[support vector machines](https://en.wikipedia.org/wiki/Support-vector_machine),
[neural networks](https://arxiv.org/abs/1705.08292),
and [decision trees](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.129.6343&rep=rep1&type=pdf).

Using a generalization of
[previous work](https://www.cs.huji.ac.il/~shais/papers/ShalevSi08.pdf),
we know that under the linear separation assumption, there has to be a feature
that gives nontrivial information.
To formalize it, we use the notion of
[decision stumps](https://en.wikipedia.org/wiki/Decision_stump) and
[weak learners](https://en.wikipedia.org/wiki/Boosting_(machine_learning)).
A decision stump is a (simple) hypothesis of the form $sign(x_i-\theta)$ defined
by a feature $i$ and a threshold $\theta$.
A hypothesis class is a $\gamma$-weak learner if one can learn it with accuracy
$\gamma$ (slightly) better than random, i.e., if there is always a
hypothesis in the class with accuracy of at least $1/2+\gamma$.

Now, we look at the hypothesis class of all possible decision stumps, and we want
to show that this class is a weak learner.
For each dataset $S=((x^1,y^1),\ldots,(x^m,y^m))$, we denote
the best decision stump for this dataset by $h_S(x)=sign(x_i-\theta)$, where $i$
is a feature and $\theta$ is a threshold that minimize the error
$\sum_{j=1}^m sign(x^j_i < \theta) y^j.$
We can show that $h_S$ has accuracy better than $0.5$, i.e., better than a
random guess:

<div class="theorem">
  Fix $\alpha>0$.
  For any distribution $\mu$ over $[-1,+1]^d\times\{-1,+1\}$ that satisfies
  linear separability with a $\gamma$-margin, and for any $\delta\in(0,1)$ there
  is $m=O\left(\frac{d+\log\frac1\delta}{\gamma^2}\right)$, such that with
  probability at least $1-\delta$ over the sample $S$ of size $m$, it holds that
  $$\Pr_{(x,y)\sim\mu}(h_S(x)=y)\geq \frac12+\frac{\gamma}{4}-\alpha.$$
</div>

This result proves that there exists a classifier $h_S$ in the hypothesis class of
all possible decision stumps that produces a non-trivial
solution under the linear separability assumption.
Using this theorem along with the result from
[Kearns and Mansour](https://www.sciencedirect.com/science/article/pii/S0022000097915439),
we can show that
[CART](https://onlinelibrary.wiley.com/doi/full/10.1002/widm.8?casa_token=O2ehHd8cYlwAAAAA%3AplOtiUnZ41vnEXcvBTZiQxwPJfl1DTFB4ROZX8fX7VP0uXhyxJoqXmRKAIdUyaXRHe7EP1Y860w38A)-type
algorithms can deliver a **small** tree with high accuracy.
As a side benefit, this is the <inc>first</inc> time that a distributional
assumption that does not include feature independence is used.
Many papers on theoretical guarantees of decision trees assumed either uniformity or feature independence
(papers [1](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.129.6343&rep=rep1&type=pdf),
[2](https://arxiv.org/abs/1911.07375), and
[3](http://proceedings.mlr.press/v125/brutzkus20a/brutzkus20a.pdf)).

Are we done? Is this model also robust?

## New algorithm: BBM-RS

Designing robust decision trees is inherently a difficult task.
One reason is that, generally, the models defined by the right and left subtrees
can be completely different.
The feature $i$ in the root determines if the model
uses the right or left subtree.
Thus, a small change in the $i$-th feature completely changes the model.
To overcome this difficulty, we focus on a specific class of decision trees.
<!--**Note that** in the decision tree that corresponds to the risk score, the right
and left subtrees are the same.-->

### Risk score
<!--For decision trees, each inner node corresponds to a threshold and a
feature and each leaf corresponds to a label.
The label of an example is the leaf’s label in the corresponding path.
We focus on binary classification problems in this post.
In [our paper](https://arxiv.org/abs/2102.07048), we construct
a specific kind of decision tree ---
[risk scores](https://jmlr.org/papers/volume20/18-615/18-615.pdf).-->
We design our algorithm to learn a specific kind of decision tree ---
[risk score](https://jmlr.org/papers/volume20/18-615/18-615.pdf).
A risk score is composed of several conditions (e.g., $age \geq 75$), and each
is matched with an integer weight.
A score $s(x)$ of example $x$ is the weighted sum of all the satisfied
conditions.
The label is then $sign(s(x))$.


<div style="width: 100%; overflow-x: auto; margin-bottom: 35pt">
<table style="width: 90%; font-size: 80%; margin: auto;" class="concise-table">
  <caption>
    <!--Two risk score models:
    <a href='https://jmlr.org/papers/volume20/18-615/18-615.pdf'>LCPA</a> and our new BBM-RS algorithm on the
    <a href='https://core.ac.uk/download/pdf/55631291.pdf'>bank dataset</a>.-->
    An example of the risk score model on the <a href='https://core.ac.uk/download/pdf/55631291.pdf'>bank dataset</a>.
    Each satisfied condition is multiplied by its weight and summed. Bias term is always satisfied.
    If the total score $>0$, the risk model predicts "1" (i.e., the client will open a bank account after a marketing call).
    All features are binary (either $0$ or $1$).
    For a concrete example, a person with age greater than 75, called before but the previous call
    was not successful, and the consumer price index is greater than 93.5, the total score would be
    $1$ and the prediction would be "1".
  </caption>
  <tr>
    <th colspan="1">features</th>
    <th colspan="2" style="text-align: center;">weights</th>
  </tr>
  <tr>
   <th>Bias term</th> <th style="text-align: center;">-5</th> <th> + ... </th>
  </tr>
  <tr>
   <th>Age $\geq 75$</th> <th style="text-align: center;">2</th> <th> + ... </th>
  </tr>
  <tr>
   <th>Called before</th> <th style="text-align: center;">4</th> <th> + ... </th>
  </tr>
  <tr>
   <th>Previous call was successful</th> <th style="text-align: center;">2</th> <th> + ... </th>
  </tr>
  <tr>
   <th></th> <th>total scores=</th> <th> </th>
  </tr>
</table>
</div>

A risk score can be viewed as a decision tree with the same feature-threshold pair at
each level (see example below).
A risk score has simpler structure than a standard decision tree,
and it generally has fewer number of *unique* nodes.
Hence, they are considered
[more interpretable than decision trees](https://jmlr.org/papers/volume20/18-615/18-615.pdf).
The following table shows an example of a risk score.

<div style="width: 100%; overflow-x: auto; margin-bottom: 20pt">
  <table style="width: 100%; font-size: 80%; margin-bottom: 2pt;" class="concise-table">
  <caption>
    Here is an example of how to convert a risk score into a decision tree.
    The table on the left is an example of a risk score that may be used by a doctor to determine
    whether a patient caught a cold or not.
    It has three conditions and the figure on the right is the corresponding decision tree.
    For each node in the tree, the branch towards the right represents the path to take if the condition is true.
    The leaves represent the final risk score of the given condition.
    For a concrete example, if a patient has a fever, coughs, but does not sneeze, we would follow the green
    path in the decision tree and result in a score of $2$.
  </caption>
  </table>
  <span style="width: 49%; overflow-x: auto; display: inline-block; margin: 10pt; float: left;">
    <table style="width: 100%; font-size: 80%; margin: auto;" class="concise-table">
      <tr>
        <th colspan="1" style="text-align: left;">features</th>
        <th colspan="1" style="text-align: center;">weights</th>
        <th colspan="1"></th>
      </tr>
      <tr><th>Bias term</th> <th style="text-align: center;">-3</th> <th> + ... </th></tr>
      <tr><th>Fever</th> <th style="text-align: center;">3</th> <th> + ... </th></tr>
      <tr><th>Sneeze</th> <th style="text-align: center;">1</th> <th> + ... </th></tr>
      <tr><th>Cough</th> <th style="text-align: center;">2</th> <th> + ... </th></tr>
      <tr><th></th> <th>total scores=</th> <th></th></tr>
    </table>
  </span>
  <span style="width: 47%; overflow-x: auto; display: inline-block; margin: auto; float: right;">
    <img src='/assets/2021-10-24-interpretable-robust-trees/risk_score_tree.png' style="width: 100%">
  </span>
</div>

### BBM-RS

We design a new algorithm for learning risk scores by utilizing the known
boosting method
[boost-by-majority](https://link.springer.com/content/pdf/10.1023/A:1010852229904.pdf)
(BBM).
The different conditions are added to the risk score one by one, using
the weak learner.
BBM has the benefit of ensuring the weights in the risk score
are small integers.
This will lead to an interpretable model with size only
$O(\gamma^{-2}\log1/\epsilon)$ where the model has accuracy $1-\epsilon$.

Now we want to make sure that the risk model is also robust.
The idea is to add
noise.
We take each point in the sample and just make sure that it's a little
bit closer to the decision boundary, see the figure below.
{:refdef: style="text-align: center;"}
  <img src='/assets/2021-10-24-interpretable-robust-trees/BBM_RS_add_noise.png'>
{:refdef}
The idea is that if the model is correct for the noisy point, then it
should be correct for the point without the noise.
To formally prove it, we show that choosing the risk-score conditions in a specific 
way ensures that they are monotone models.
In such models, adding noise in the way we described is
sufficient for robustness.

Before we examine this algorithm on real datasets, let’s check its running time.
We focus on the case the margin and desired accuracy are constants.
In this case, the number of steps BBM-RS will take is also constant.
In each step, we run the weak learner and find the best $(i,\theta)$.
So the overall time is linear (up to logarithmic factors) in the input size and the time to run the
weak learner.

To summarize, we designed a new efficient algorithm, BBM-RS, that is robust, interpretable, and
has high accuracy. The following theorem shows this. Please refer to
[our paper](https://arxiv.org/abs/2102.07048) for the pseudocode of BBM-RS
and more details for the theorem.

<div class="theorem">
Suppose data is $\gamma$-linearly separable and fix $\epsilon,\delta\in(0,1)$.
Then, with probability $1-\delta$ the output of BBM-RS, after receiving
$(d+\log(1/\delta))\log(1/\epsilon)\gamma^{-O(1)}$ samples, has astuteness
$1-\epsilon$ at radius $\gamma/2$ and has  $O(\gamma^{-2}\log(1/\epsilon))$
feature-threshold pairs.
</div>

### Performance on real data

For BBM-RS, our theorem is restricted to linearly separable data.
However, real datasets may not perfectly linearly separable.
A straightforward question: is linear separability a reasonable
assumption in practice?

To answer this question, we consider $13$ real datasets (here we present the
results for four datasets; for more datasets, please refer to [our
paper](https://arxiv.org/abs/2102.07048)).
We measure how linearly separable each of these datasets is.
We define the __linear separateness__ as one minus the minimal fraction
of points that needed to be removed for the data to be linearly separable.
Since finding the optimal linear separateness on arbitrary data
[is NP-hard](https://www.sciencedirect.com/science/article/pii/S0022000003000382),
we approximate linear separateness with the training accuracy of the best linear classifier we can
find (since removing the incorrect examples for a linear classifier would make the dataset linearly
separable).
We train linear SVMs with different regularization parameters and record the best training accuracy.
After removing the misclassified points by an SVM, we are left with accuracy 
fraction of linearly separable examples.
The higher this accuracy is, the more linearly separable the data is.
The following table shows the results and it reveals that most datasets
are very or moderately close to being linearly separated.
This indicates that the linear assumption in our theorem may not be too
restrictive in practice.



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

Even though these datasets are not perfectly linearly separable, BBM-RS can
still be applied (but the theorem may not hold).
We are interested to see how BBM-RS performed against others on these
non-linearly separable datasets.
We compare BBM-RS to three baselines,
[LCPA](https://arxiv.org/abs/1610.00168),
[decision tree (DT)](https://books.google.co.il/books?hl=en&lr=&id=MGlQDwAAQBAJ&oi=fnd&pg=PP1&ots=gBmdjTJVdK&sig=\_jUBiPW4cTS7JYUKpzKcJLYipl4&redir_esc=y#v=onepage&q&f=false), and
[robust decision tree (RobDT)](http://proceedings.mlr.press/v97/chen19m/chen19m.pdf).
We measure a model's robustness by evaluating its
[__Empirical robustness (ER)__](https://arxiv.org/abs/2003.02460), which is the
average $\ell_\infty$
distance to the closest adversarial example on correctly predicted test examples.
The larger ER is, the more robust the classifier is.
We measure a model's interpretability by evaluating its
__interpretation complexity (IC)__.
We measure IC with the number of unique feature-threshold pairs in the
model (this corresponds to the number of conditions in the risk score).
The smaller IC is, the more interpretable the classifier is.
The following tables show the experimental results.

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

From the tables, we see that BBM-RS has a test accuracy comparable to other
methods.
In terms of robustness, it performs slightly better than others (performing the
best on three datasets among a total of four).
In terms of interpretability, BBM-RS
performs the best in three out of four datasets.
All in all, we see that BBM-RS can bring better robustness and interpretability
while performing competitively on test accuracy.
This shows that BBM-RS not only performs well theoretically, it also performs
well empirically.

## Conclusion

We investigated three important properties of a classifier: accuracy, robustness, and
interpretability.
We designed and analyzed a tree-based algorithm that provably achieves all these properties, under
linear separation with a margin assumption.
Our research is a step towards building trustworthy models that provably achieve many desired
properties.

Our research raises many open problems.
What is the optimal dependence between accuracy, interpretation complexity,
empirical robustness, and sample complexity?
Can we have guarantees using different notions of interpretability?
We showed how to construct an interpretable, robust, and accurate model. But,
for reliable machine learning models, many more properties are required,
such as privacy and fairness.
Can we build a model with guarantees on all these properties simultaneously?

#### More Details

See [our paper on arxiv](https://arxiv.org/abs/2102.07048) or [our repository](https://github.com/yangarbiter/interpretable-robust-trees).
