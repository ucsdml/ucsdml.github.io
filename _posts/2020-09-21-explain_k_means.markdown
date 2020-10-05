---
layout: post
mathjax: true
title:  "Explainable k-means Clustering"
date:   2020-09-21 10:00:00 -0700
categories: jekyll update
tags: explainable
author:  <a href='mailto:navefrost@mail.tau.ac.il'>Nave Frost</a>, <a href='mailto:mmoshkovitz@eng.ucsd.edu'>Michal Moshkovitz</a>, <a href='mailto:crashtchian@eng.ucsd.edu'>Cyrus Rashtchian</a>

paper_url: https://proceedings.icml.cc/paper/2020/file/8e489b4966fe8f703b5be647f1cbae63-Paper.pdf
code_url: https://github.com/navefr/ExKMC
excerpt: Popular algorithms for learning decision trees can be arbitrarily bad for clustering. We present a new algorithm for explainable clustering that has provable guarantees --- the Iterative Mistake Minimization (IMM) algorithm. This algorithm exhibits good results in practice. It's running time is comparable to KMeans implemented in sklearn. So our method gives you explanations basically for free. Our code is available on github. 
---

**TL;DR:** 
Explainable AI has gained a lot of interest in the last few years, but effective methods for unsupervised learning are scarce. And the rare methods that do exist do not have provable guarantees. We present a new algorithm for explainable clustering that is provably good for $k$-means clustering --- the Iterative Mistake Minimization (IMM) algorithm. Specifically, we want to build a clustering defined by a small decision tree. Overall, this post summarizes our new paper: [Explainable $k$-Means and $k$-Medians clustering](https://arxiv.org/pdf/2002.12538.pdf).

### Explainability: why?
Machine learning models are mostly "black box". They give good results, but their reasoning is unclear. These days, machine learning is entering fields like healthcare (e.g., for a better understanding of [Alzheimer's Disease](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6543980/#:~:text=In%20the%20medical%20field%2C%20clustering,in%20labeled%20and%20unlabeled%20datasets.&text=The%20aim%20is%20to%20provide,AD%20based%20on%20their%20similarity.) and [Breast Cancer](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118453#sec013)), transportation, or law. In these fields, quality is not the only objective. No matter how well a computer is making its predictions, we can't even imagine blindly following computer's suggestion. Can you imagine blindly medicating or performing a surgery on a patient just because a computer said so? Instead, it would be much better to provide insight into what parts of the data the algorithm used to make its prediction.


### Tree-based explainable clustering
<!--Despite the popularity of explainability, there is limited work in unsupervised learning. To remedy it, --> 
We study a prominent problem in unsupervised learning, $k$-means clustering. We are given a dataset, and the goal is to partition it to $k$ clusters such that the [$k$-means cost](https://en.wikipedia.org/wiki/K-means_clustering) is minimal. The cost of a clustering $C=(C^1,\ldots,C^k)$ is the sum of all points from their optimal centers, $mean(C^i)$:

\\[cost(C)=\sum_{i=1}^k\sum_{x\in C^i} \lVert x-mean(C^i)\rVert ^2.\\]


For any cluster, $C^i$, one possible explanation of this cluster is $mean(C^i)$. In a low-cost clustering, the center is close to its points, and they are close to each other. For example, see the next figure. 


{:refdef: style="text-align: center; float: left"}
<figure class="image" style="float: left">
 <img src="/assets/2020-09-21-explain_k_means/intro_IMM_blog_pic_1.png" width="40%" style="margin: 0 auto">
 <figcaption>
  Near optimal 5-means clustering
 </figcaption>
</figure>
{:refdef}

Unfortunately, this explanation is not as useful as it could be. The centers themselves may depend on all the data points and all the features in a complicated way. We instead aim to develop a clustering method that is explainable by design. To explain why a point is in a cluster, we will only need to look at small number of features, and we will just evaluate a threshold for each feature one by one. This allows us to extract information about which features cause a point to go to one cluster compared to another. This method also means that we can derive an explanation that does not depend on the centers.

More formally, at each step we test if $x_i\leq \theta$ or not, for some feature $i$ and threshold $\theta$. We call this test a **split**. According to the test's result, we decide on the next step. In the end, the algorithm returns the cluster identity. This procedure is exactly a decision tree where the leaves correspond to clusters. 

Importantly, for the tree to be explainable it should be **small**. The smallest decision tree has $k$ leaves since each cluster must appear in at least one leaf. We call a clustering defined by a decision tree with $k$ leaves a **tree-based explainable clustering**. See the next tree for an illustration.


<p align="center">
<tr>
    <td> <img src="/assets/2020-09-21-explain_k_means/intro_IMM_blog_pic_2.png" width="40%" style="margin: 0 auto"/>  </td>
    <td> <img src="/assets/2020-09-21-explain_k_means/intro_IMM_blog_pic_3.png" width="40%" style="margin: 0 auto"/> </td>
  </tr>
</p>

<!--
{:refdef: style="text-align: center;"}
<figure class="image">
 <img src="/assets/2020-06-06/intro_IMM_blog_pic_2.png" width="40%" style="margin: 0 auto">
 <figcaption>
  Decision tree
 </figcaption>
</figure>
{:refdef}


{:refdef: style="text-align: center;"}
<figure class="image">
 <img src="/assets/2020-06-06/intro_IMM_blog_pic_3.png" width="40%" style="margin: 0 auto">
 <figcaption>
  Geometric representation of the decision tree
 </figcaption>
</figure>
{:refdef}
-->

On the left, we see a decision tree that defines a clustering with $5$ clusters. On the right, we see the geometric representation of this decision tree. We see that the decision tree imposes a partition to $5$ clusters aligned to the axis. The clustering looks close to the optimal clustering that we started with. Which is great. But can we do it for all datasets? How?

Several algorithms are trying to find a tree-based explainable clustering like [CLTree](https://link.springer.com/chapter/10.1007/11362197_5) and [CUBT](https://d1wqtxts1xzle7.cloudfront.net/52949489/09e41508aeaf39a453000000.pdf?1493812476=&response-content-disposition=inline%3B+filename%3DClustering_using_Unsupervised_Binary_Tre.pdf&Expires=1596413380&Signature=WaxRD8LssFz4XMuD2C~m1oB62igf7B5Iea~lCDhv7VcU68ZpkbeMXuHop~qZnKEbuqPMyc6sWwFHFQulHJ1XSRnhjNHix93EhB~LS-dVIlwtB9aB6qKHgefuszHTj-igogeWfocU~VHCOI5VfeozOfDJf-S4mWZBc7~En2rdcTDqz~c2y8ykT9oyeYpRzwnfSd5phmE3VHWln9rSFAJYB4PhxlcuP8sD7MkJgkJ7rx666LKxQY5MoR3qBqiwUwkYZbLN3GZtDLqeetcKGO94j2hW8K6mIlFk625-1QrP49ZIlmNJzlylaKNyqJ1ebQHBp9EVmohCB50joYMtIU2aQQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA). But we are the first to give formal guarantees. We first need to define the quality of an algorithm. It's common that unsupervised learning problems are [NP-hard](http://cseweb.ucsd.edu/~dasgupta/papers/kmeans.pdf). Clustering is no exception.  So it is common to settle for an approximated solution. A bit more formal, an algorithm that returns a tree-based clustering $T$ is an *$a$-approximation* if $cost(T)\leq a\cdot cost(opt),$ where $opt$ is the clustering that minimizes the $k$-means cost.


### General scheme
Many supervised learning algorithms learn a decision tree, can we use one of them here? Yes, after we transform the problem into a supervised learning problem! How might you ask? We can use any clustering algorithm that will return a good, but not explainable clustering. These will form the labeling. Next, we can use a supervised algorithm that learns a decision tree. Let's summarize these three steps:
1. Find a clustering using some clustering algorithm
2. Label each example according to its cluster
3. Call a supervised algorithm that learns a decision tree


Which algorithm can we use in step 3? Maybe the popular ID3 algorithm?

### Can we use the ID3 algorithm?
Short answer: no.

One might hope that in step 3, in the previous scheme, the known [ID3](https://link.springer.com/content/pdf/10.1007/BF00116251.pdf) algorithm can be used (or one of its variants like [C4.5](https://link.springer.com/article/10.1007/BF00993309)). We will show that this does not work. There are datasets where ID3 will perform poorly. Here is an example:

{:refdef: style="text-align: center;"}
<figure class="image">
 <img src="/assets/2020-09-21-explain_k_means/intro_IMM_blog_pic_4.png" width="40%" style="margin: 0 auto">
 <figcaption>
  ID3 perform poorly on this dataset
 </figcaption>
</figure>
{:refdef}

The dataset is composed of three clusters, as you can see in the figure above. Two large clusters (0 and 1 in the figure) have centers (-2, 0) and (2, 0) accordingly and small noise. The third cluster (2 in the figure) is composed of only two points that are very, very (very) far away from clusters 0 and 1. Given these data, ID3 will prefer to maximize the information gain and split between clusters 0 and 1. Recall that the final tree has only three leaves. This means that in the final tree, one point in cluster 2 must be with cluster 0 or cluster 1. Thus the cost is enormous.
To solve this problem, we design a new algorithm called [*Iterative Mistake Minimization (IMM)*](https://proceedings.icml.cc/paper/2020/file/8e489b4966fe8f703b5be647f1cbae63-Paper.pdf).

### IMM algorithm for explainable clustering
We learned that the ID3 algorithm cannot be used in step 3 at the general scheme. Before we give up on this scheme, can we use a different decision-tree algorithm? Well, since we wrote this post, you probably know the answer: there is such an algorithm, the IMM algorithm.

We build the tree greedily from top to bottom. Each step we take the split (i.e., feature and threshold) that minimizes a new parameter called a **mistake**. A point $x$ is a mistake for node $u$ if $x$ and its center $c(x)$ reached $u$ and then separated by $u$'s split. See the next figure for an example of a split with one mistake.
{:refdef: style="text-align: center;"}
<figure class="image">
 <img src="/assets/2020-09-21-explain_k_means/mistakes_example.png" width="40%" style="margin: 0 auto">
 <figcaption>
  Split (in yellow) with one mistake. Two optimal clusters are in red and blue. Centers are the stars.
 </figcaption>
</figure>
{:refdef}

<!--For another example of the mistakes concept, let's go back to the previous dataset where ID3 failed. Focus on the first split again. The ID3 split has one mistake since one of the points in cluster $2$ will be separated from its center. On the other hand, the horizontal split has $0$ mistakes: the two large clusters will go with their centers to one side of the tree, and the small cluster will go with its center to the other side of the tree. -->

To summarize, the high-level description of the IMM algorithm: &nbsp;
<!--<center>
<span style="font-family:Papyrus; font-size:2em;align-self: center;">As long as there is more than one center
 <br> find the split with minimal number of mistakes</span>
</center>
-->
<center>
<span style="font-size:larger;">
As long as there is more than one center
 <br> find the split with minimal number of mistakes
</span>
</center>
&nbsp;
 


<!--What if there are no mistakes. 
The main definition that we need is a mistake:
Creare a different figure that explains a mistake with small number of points 
-->

<!--
<center>
<span style="font-family:Papyrus; font-size:2em;align-self: center;">If a point and its center diverge,
 <br> then it counts as a mistake</span>
</center>


<div class="definition"> [mistake at node $u$]. 
If a point and its center end up at different leafs, then it counts as a mistake.
</div>
... Explain what is a split early on ... 
-->




<!---
{% highlight python %}
def IMM(points, centers):
 node = new Node()
 if |centers| > 1:
  i, theta = find_split(points, centers)
  node.condition = 'x_i <= theta'

  points_left_mask = points[:,i] <= theta
  centers_left_mask = centers[:,i] <= theta

  node.left = IMM(points[points_left_mask], centers[centers_left_mask])
  node.right = IMM(points[~points_left_mask], centers[~centers_left_mask])
 else:
  node.label = centers

 return node

def find_split(points, centers):
 for i in range(d):
  l = min(centers[:,i])
  r = max(centers[:,i])
 i,theta = argmin_{i,l <= theta < r} mistakes(i, theta)

 return i,theta
{% endhighlight %}

-->


Here is an illustration of the IMM algorithm. We use $k$-means++ with $k=5$ to find a clustering for our dataset. Each point is colored with its cluster label. At each node in the tree, we choose a split with a minimal number of mistakes. We stop if a node contains only one center, we call it *homogeneous*. In the end, we stop where each of the $k=5$ centers is in its own leaf. This defines the explainable clustering on the left.
<center>
<img src="/assets/2020-09-21-explain_k_means/imm_example_slow.gif" width="600" height="320" />
</center>

The algorithm is guaranteed to perform well. For any dataset. See the next theorem.
<div class="theorem">
IMM is an $O(k^2)$-approximation to the optimal $k$-means clustering.
</div>

This theorem shows that we can always find a small tree, with $k$ leaves, such that the tree-based clustering is only $O(k^2)$ times worse in terms of the cost.  IMM efficiently find this explainable clustering. Importantly, this approximation is independent of the dimension and the number of points. A proof for the case $k=2$ will appear in a [follow-up post](explain_2_means.html), and you can read the proof for general $k$ in the paper. Intuitively, we discovered that the number of mistakes is a good indicator for the $k$-means cost, and so, minimizing the number of mistakes is an effective way to find a low-cost clustering. <!-- Surprisingly, we can also use a tree with $k$ leaves, which means that IMM produces an explainable clustering.-->

#### Running Time

What is the running time of the IMM algorithm? With an efficient implementation, using dynamic programming, the running time is $O(kdn\log(n)).$ Why? For each of the $k-1$ inner nodes and each of the $d$ features, we can find the split that minimizes the number of mistakes for this node and feature, in time $O(n\log(n)).$

For $2$-means one can do better than running IMM: going over all possible $(n-1)d$ cuts and find the best one. The running time is $O(nd^2+nd\log(n))$.

### Results Summary
In each cell in the following table, we write the approximation factor. We want this value to be small for the upper bounds and large for the lower bounds.  In $2$-medians, the upper and lower bounds are pretty tight, about $2$. But, there is a large gap for $k$-means and $k$-median: the lower bound is $\log(k)$, while the upper bound is $\mathsf{poly}(k)$. 

<center>
<table style="text-align: center">
<thead>
 <tr>
  <th></th>
  <th colspan="2" style="text-align: center">$k$-medians</th>
  <th colspan="2" style="text-align: center">$k$-means</th>
 </tr>
  <tr>
  <th></th>
  <th> $k=2$ </th>
  <th> $k>2$ </th>
  <th> $k=2$ </th>
  <th> $k>2$ </th>
 </tr>
</thead>
<tbody>
 <tr>
  <td> <strong>Lower</strong> </td>
  <td> $2-\frac1d$ </td>
  <td> $\Omega(\log k)$ </td>
  <td> $3\left(1-\frac1d\right)^2$ </td>
  <td> $\Omega(\log k)$ </td>
 </tr>
 <tr>
  <td> <strong>Upper</strong> </td>
  <td> $2$ </td>
  <td> $O(k)$ </td> 
  <td> $4$ </td>
  <td> $O(k^2)$ </td>
 </tr>
</tbody>
</table>
</center>


### What's next
1. IMM exhibits excellent results in practice on many datasets, see [this](https://arxiv.org/abs/2006.02399). It's running time is comparable to KMeans implemented in sklearn. We implemented the IMM algorithm, it's [here](https://github.com/navefr/ExKMC). Try it yourself.
2. We plan to have several posts on explainable clusterings, here is the [second](explain_2_means.html) in the series, stay tuned for more!
3. In a follow-up work, we explore the tradeoff between explainability and accuracy. If we allow a slightly larger tree, can we get a lower cost? We introduce the [ExKMC](https://arxiv.org/abs/2006.02399), "Expanding Explainable $k$-Means Clustering", algorithm that builds on IMM.
4. Found cool applications of IMM? Let us know!