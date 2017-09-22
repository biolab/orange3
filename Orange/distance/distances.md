# Computation of distances in Orange 3

This document describes and justifies how Orange 3 computes distances between data rows or columns from the data that can include discrete (nominal) and numeric features with missing values.

The aim of normalization is to bring all numeric features onto the same scale and on the same scale as discrete features. The meaning of *the same scale* is rather arbitrary. We gauge the normalization so that missing values have the same effect for all features and their types.

For missing values, we compute the expected difference given the probability distribution of the feature that is estimated from the data.

Two nominal values are treated as same or different, that is, the difference between them is 0 and 1.

Difference between values of two distinct nominal features does not make sense, so Orange reports an error when the user tries to compute column-wise distance in data with some non-numeric features.

## Euclidean distance

#### Normalization of numeric features

Orange 2 used to normalize by subtracting the minimum and dividing by the span (the difference between the maximum and the minimum) to bring the range of differences into interval $[0, 1]$. This however did not work since the data on which the distances were computed could include more extreme values than the training data.

Normalization in Orange 3 is based on mean and variance due to other desired effects described below. A value $x$ is normalized as

$$ x' = \frac{x - \mu}{\sqrt{2\sigma^2}},$$

where $\mu$ and $\sigma^2$ are the mean and the variance of that feature (e.g. estimated accross the column).

Normalized values thus have a mean of $\mu'=0$ and a variance of $\sigma'^2 = 1/2$.

#### Missing values of numeric features

If one value (denoted by $v$) is known and one missing, the expected difference along this dimension is

$$\int_{-\infty}^{\infty}(v - x)^2p(x)dx = \\
v^2\int_{-\infty}^{\infty}p(x)dx- 2v\int_{-\infty}^{\infty}xp(x) + \int_{-\infty}^{\infty}x^2p(x) = \\
v^2 - 2v\mu + (\sigma^2 + \mu^2) = \\
(v - \mu)^2 + \sigma^2.$$

If both values are unknown and we compute the difference between rows so that both values come from the same distirbutions, we have

$$\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}(x - y)^2p(x)p(y)dxdy = \\
 \int_{-\infty}^{\infty}\int_{-\infty}^{\infty}x^2p(x)p(y)dxdy
 + \int_{-\infty}^{\infty}\int_{-\infty}^{\infty}y^2p(x)p(y)dxdy
 - 2\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}xyp(x)p(y)dxdy = \\
 (\sigma^2 + \mu^2) + (\sigma^2 + \mu^2) - 2\int_{-\infty}^{\infty}xp(x)dx\int_{-\infty}^{\infty}yp(y)dxdy = \\
 (\sigma^2 + \mu^2) + (\sigma^2 + \mu^2) - 2\mu\mu = \\
 2\sigma^2.$$

When computing the difference between columns, the derivation is similar except that the two distributions are not the same. For one missing value we get

$$(v - \mu_x)^2 + \sigma_x^2$$

where $\mu_x$ and $\sigma_x$ correspond to the distribution of the unknown value. For two missing values, we get

$$\sigma_x^2 + \sigma_y^2 + \mu_x^2 + \mu_y^2 - 2\mu_x\mu_y = \\
\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^ 2.$$

For normalized data, the difference between $v$ and unknown value is $(v' - \mu)^2 + \sigma^2 = v'^2 + 1/2$. The difference between two missing values is $2\sigma^2 = 1$ (or $\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2 = 1$).

#### Missing values of discrete features

The difference between a known and a missing value is

$$\sum_x \mbox{I}_{v\ne x}^2p(x) = 1 - p(x).$$

The difference between two missing values is

$$\sum_x\sum_y \mbox{I}_{y\ne x}^2p(x)p(y) = 1 - \sum_x p(x)^2.$$

This is the Gini index. Also, if the number of values goes to infinity and the distribution towards the uniform, the difference goes towards 1, which brings it, in some sense, to the same scale as continuous features.

This case assumes that $x$ and $y$ come from the same distribution. The case when these are missing values of two distinct discrete features is not covered since Orange does not support such distances (see the introduction).

## Manhattan distance

The derivation here may be more difficult, so we do it by analogy with Euclidean distances and hope for the best.

We use the median ($m$) and median of absolute distances to the median (mad) ($a$) instead of the mean and the variance.

Normalization of numeric features: $x' = (x - m)\,/\,(2a)$

#### Missing values of numeric features

**Between a known and a missing**: $|v - m| + a.$

**Between two unknowns**: $2a$ (same features), $a_x + a_y$ (different features).

**For normalized data**: $|v'| + 1/2$ (one unknown), $1$ (both unknown).

#### Missing values of discrete features

Same as for Euclidean because $I_{v\ne x}^2 = I_{v\ne x}$.


## Cosine distance

The cosine similarity is normalized, so we do not normalize the data.

Cosine distance treats discrete features differently from the Euclidean and Manhattan distance. The latter needs to consider only whether two values of a discrete feature is different or not, while the cosine distance normalizes by dividing by vector lengths. For this, we need the notion of absolute magnitude of a (single) discrete value -- as compared to some "base value".

For this reason, cosine distance treats discrete attributes as boolean, that is, all non-zero values are treated as 1. This may be incorrect in some scenarios, especially those in which cosine distance is inappropriate anyway. How to (and whether to) use cosine distances on discrete data is up to the user.

#### Distances between rows

For a continuous variable $x$, product of $v$ and a missing value is computed as

$$\int_{-\infty}^{\infty}vxp(x)dx = v\mu_x$$

The product of two unknown values is

$$\int_{-\infty}^{\infty}xp(x)yp(y)dxdy=\mu_x^2$$

For discrete values, we compute the probabilities $p(x=0)$ and $p_x(x\ne 0)$. The product of known value $v$ and a missing value is 0 if v=0 and $p(x \ne 1)$ otherwise. The product of two missing values is $p(x\ne 1)^2$.

When computing the absolute value of a row, a missing value of continuous variable theoretically contributes

$$\int_{-infty}^{\infty}x^2p(x)dx = \mu_x^2 + \sigma_x^2$$

However, since we essentially impute the mean in the dot product, the actual contribution of the missing value is $\mu_x^2$. We therefore use this value, which also simplifies the computation which is reduced to simple imputation of means.

A missing value of discrete variable contributes

$$1\cdot 1\;p(x\ne 0) = p(x\ne 0)$$


#### Distances between columns

The product of a known value $v$ and a missing value of $x$ is $v\mu_x$. The contribution of a missing value to the absolute value of the column is $\mu_x^2+\sigma_x^2$. All derivations are same as above.

## Jaccard distance

Jaccard index, whose computation is described below, is a measure of similarity. The distance is computed by subtracting the similarity from 1.

Let $p(A_i)$ be the probability (computed from the training data) that a random data instance belongs to set $A_i$, i.e., have a non-zero value for feature $A_i$.

### Similarity between rows (instances)

Let $M$ and $N$ be two data instances. $M$ and $N$ can belong to $A_i$ (or not). The Jaccard similarity between $M$ and $N$ is the number of the common sets to which $M$ and $N$ belong, divided by the number of sets with either $M$ or $N$ (or both).

Let $\mbox{I}_M$ be 1 if $M\in A_i$ and 0 otherwise. Similarly, $\mbox{I}_{M'}$ will indicate that $M\not\in A_i$ and $\mbox{I}_{M?}$ will indicate that it is unknown whether $M$ belongs to $A_i$ or not. We will also use conjuctions and disjunctions by adding more indices to $\mbox{I}$; e.g. $\mbox{I}_{M\wedge N'}$ indicates that $M$ belongs to $A_i$ and $N$ does not. $A_i$ is omitted for clarity as it can be deduced from the context.

$$Jaccard(M, N) = \frac{\sum_i\mbox{I}_{M\wedge N}}{\sum_i\mbox{I}_{M\vee N}}$$

Consider that $\mbox{I}_{M\wedge N} = \mbox{I}_M \mbox{I}_N$ and $\mbox{I}_{M\vee N} = \max(\mbox{I}_M, \mbox{I}_N)$. If the data whether $M\in A_i$ or $N\in A_i$ is missing, we replace indicator function with the probability. In the denominator we add a few terms to $\mbox{I}_{M\wedge N}$

$$\mbox{I}_{M\wedge N}
  + p(A_i)\mbox{I}_{M\wedge N?}
  + p(A_i)\mbox{I}_{M?\wedge N}
  + p(A_i)^2\mbox{I}_{M?\wedge N?},$$

and in the nominator we get

$$\mbox{I}_{M\vee N} + p(A_i)\mbox{I}_{M'\wedge N?} + p(A_i)\mbox{I}_{M?\wedge N'} + \left(1 - (1 - p(A_i)^2\right)\mbox{I}_{I?\wedge J?}$$

Note that the denominator counts cases $\mbox{I}_{M'\wedge N?}$ and not $\mbox{I}_{N?}$, since those for which $M\in A_i$ are already covered in $\mbox{I}_{M\vee N}$. The last term refers to the probability that at least one (that is, not none) of the two instances is in $A_i$.

### Similarity between columns

$\mbox{I}_{i}$ will now denote that a data instance $M$ belongs to $A_i$, $\mbox{I}_{i'}$ will denote it does not, and $\mbox{I}_{i?}$ will denote that it is unknown whether $M$ belongs to $A_i$.

Without considering missing data, Jaccard index between two columns is

$$Jaccard(A_i, A_j) = \frac{\sum_M\mbox{I}_{i\wedge j}}{\sum_M\mbox{I}_{i\vee j}}$$

By the same reasoning as above, the denominator becomes

$$\mbox{I}_{i\wedge j}
  + p(A_j)\mbox{I}_{i\wedge j?}
  + p(A_i)\mbox{I}_{i?\wedge j}
  + p(A_i)p(A_j)\mbox{I}_{i?\wedge j?},$$

and the nominator is

$$\mbox{I}_{i\vee j} + p(A_j)\mbox{I}_{i'\wedge j?} + p(A_i)\mbox{I}_{i?\wedge j'} + \left(1 - [1 - p(A_i)][1 - p(A_j)]\right)\mbox{I}_{I?\wedge J?}.$$

The sums runs over instances, $M$, so the actual implementation can work by counting the cases and multipying by probabilities at the end. Let $N_c$ represent the number of cases that match condition $c$, i.e. $N_c = \sum_M\mbox{I}_c$. Then

$$Jaccard(A_i, A_j) = \frac{
      N_{i\wedge j}
    + p(A_j)N_{i\wedge j?}
    + p(A_i)N_{i?\wedge j}
    + p(A_i) p(A_j)N_{i?\wedge j?}
    }{
      N_{i\vee j}
    + p(A_j)N_{i'\wedge j?}
    + p(A_i)N_{i?\wedge j'}
    + \left(1 - [1 - p(A_i)][1 - p(A_j]\right)N_{i?\wedge j?}
    }$$
