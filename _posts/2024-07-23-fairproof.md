---
layout: distill
title:  "FairProof : Confidential and Certifiable Fairness for Neural Networks"
date:   2024-07-23
authors: 
    - name: Chhavi Yadav
      url: https://www.chhaviyadav.org/
      affiliations:
        name: UC San Diego
bibliography: blog_fairproof.bib
paper_url: https://arxiv.org/pdf/2402.12572
code_url: https://github.com/infinite-pursuits/FairProof/tree/main
description: Machine learning models are increasingly used in societal applications, yet legal and privacy concerns demand that they very often be kept confidential. Consequently, there is a growing distrust about the fairness properties of these models in the minds of consumers, who are often at the receiving end of model predictions. To this end, we propose FairProof -- a system that uses Zero-Knowledge Proofs (a cryptographic primitive) to publicly verify the fairness of a model, while maintaining confidentiality. We also propose a fairness certification algorithm for fully-connected neural networks which is befitting to ZKPs and is used in this system. We implement FairProof in Gnark and demonstrate empirically that our system is practically feasible. Code is available at [https://github.com/infinite-pursuits/FairProof](https://github.com/infinite-pursuits/FairProof).


---

<div class='l-body' align="center">
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/2024-07-fairproof/Fairproof_diag_nomath.png">
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
Setting of FairProof. Customer (which is also the verifier) submits a loan application to a bank (which is the prover). The bank uses a machine learning model to make decisions and responds with a decision, fairness certificate and a proof of the correct computation of this certificate. This proof is verified by customer.</figcaption>
</div>


## Introduction

As machine learning models proliferate in societal applications , it is important to verify that they possess desirable properties such as accuracy, fairness and privacy. Additionally due to IP reasons, models are kept confidential and therefore verification techniques should respect this confidentiality. This brings us to the question of how do we verify properties of a model while maintaining confidentiality.

The canonical approach to verification is ‘third-party auditing’ <d-cite key="yadav2022learningtheoretic,yan2022active,pentyala2022privfair,soares2023keeping"></d-cite>, wherein an external auditor uses API queries to estimate the value of the property . However, this approach has certain problems including 1) use of auditing datasets which can be manipulated 2) model swapping wherein the model is changed post-audit 3) leaking the model in the process, which loses confidentiality <d-cite key="casper2024black,hamman2023can,fukuchi2019faking, confidant"></d-cite>. 

#### Our Key Idea

Motivated by these issues, we propose an alternate framework for verification which is based on Zero-Knowledge Proofs (ZKPs) <d-cite key="GMR,GMW"></d-cite>, a cryptographic primitive. To put it simply, a ZKP system involves two parties, a Prover and a Verifier. Prover proves the correct computation of an algorithm which uses model weights while Verifier verifies the proof given by the prover without looking at the model weights.

## Guarantees of FairProof

In this work, we propose a ZKP system called FairProof to verify the fairness of a model. We want FairProof to guarantee the following three requirements: 1) ensure that the same model is used for all customers, 2) maintain the confidentiality of the model and 3) the fairness score is correctly computed without manipulation using the fixed model weights only. The first requirement is guaranteed through model commitments -- a cryptographic commitment to the model weights binds the organization to those weights publicly while maintaining confidentiality of the weights and has been widely studied in the ML security literature <d-cite key="gupta2023sigma, boemer2020mp2ml, juvekar2018gazelle, liu2017oblivious, srinivasan2019delphi, mohassel2017secureml, mohassel2018aby3"></d-cite>. The other two requirements are guaranteed with ZKPs.

#### Problem Setting

Consider the following problem setting: on one hand, we have a customer which applies for a loan (i.e. query) and corresponds to the verifier while on the other, we have a bank which uses a machine learning model to make loan decisions and corresponds to the prover. Along with the loan decision, the bank also gives a fairness certificate and a cryptographic proof proving the correct computation of this certificate. The verifier verifies this proof without looking at the model weights.

## Two parts of FairProof

There are two important parts to FairProof : 1) How to calculate the fairness certificate (in-the-clear)? and 2) How to verify this certificate with ZKPs?

#### Fairness Certification in-the-clear

The fairness metric we use is Local Individual Fairness (IF) and give a simple algorithm to calculate this certificate by using a connection between adversarial robustness and IF.  Experimentally, we see that the resulting certification algorithm is able to differentiate between less and more fair models.

<div class='l-body' align="center">
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/2024-07-fairproof/fair-unfair">
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;"> Histogram of fairness parameter for fair and unfair models for 100 randomly sampled data points. Fairness parameter values are higher for more fair models.</figcaption>
</div>

#### ZKP for Fairness Certification

Next we must code this certification algorithm in a ZKP library. However, ZKPs are infamous for adding a big computational overhead and can be notoriously hard to code due to only using arithmetic operations. To overcome these challenges, we strategically choose some sub-functionalities which are enough to verify the certificate and also propose to do some computations offline to save time.

Empirically we find that the maximum proof generation time is on ~4 min while the maximum verification time is ~3 seconds (note the change from minutes to seconds). Maximum time is consumed by the VerifyNeighbor functionality. Also the proof size is a meagre 200 KB.

<div class='l-body' align="center">
<img class="img-fluid rounded z-depth-1" src="{{ site.baseurl }}/assets/img/2024-07-fairproof/fairproof-results">
<figcaption style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
Results are over 100 randomly sampled points from the test sete. (a) Average Proof Generation (in mins) and Verification times (in secs) for different models. Offline computations are done in the initial setup phase while Online computations are done for every new query. Verification is only done online, for every query. (b) Breakdown of the proof generation time (in mins) for the data point with the median time. VerifyNeighbor sub-functionality takes the maximum time. (c) Average Total Proof Size (in KB) for various models. This includes the proof generated during both online and offline phases.</figcaption>
</div>

## Conclusion

In conclusion, we propose FairProof – a protocol enabling model owners to issue publicly verifiable certificates while ensuring model confidentiality. While our work is grounded in fairness and societal applications, we believe that ZKPs are a general-purpose tool and can be a promising solution for overcoming problems arising out of the need for model confidentiality in other areas/applications as well.

For code check this link : [https://github.com/infinite-pursuits/FairProof](https://github.com/infinite-pursuits/FairProof)

For the paper check this link : [https://arxiv.org/pdf/2402.12572](https://arxiv.org/pdf/2402.12572)

For any enquiries, write to : [cyadav@ucsd.edu](cyadav@ucsd.edu)
