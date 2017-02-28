# Chapter 3. Statistics

Tags: quant_interview

---

## Part 1. Applied Statistics

### 1. One-Sample Proportion 

Sample proportion distribution approximates a normal distribution under two conditions $$n\pi > k$$ and $$n(1-\pi) > k$$, where $$k=5,10$$, depends on who you ask. 

Then sample proportion to a z-score $$Z = \frac{\hat{\pi} - \pi}{\sqrt{\frac{\pi(1-\pi)}{n}}}$$

**Inference for binomial parameter $$H_0: \pi = \pi_0$$**

1. Wald test: use full model / alternative hypo $$z_w = \frac{\hat{\pi} - \pi_0}{\sqrt{\hat{\pi}(1 - \hat{\pi})/n}}$$

2. Score test: use null hypo $$z_s = \frac{\hat{\pi} - \pi_0}{\sqrt{\pi_0(1 - \pi_0)/n}}$$

3. Likelihood ratio test: $$z_l = 2(L_1 - L_0)$$, following $$\chi^2$$ under $$H_0$$

**Sample size for hypo testing**

To distinguish two populations $$Ber(\pi_0)$$ and $$Ber(\pi_1)$$, we have 

$$N \ge \frac{(z_\alpha\sqrt{p_0(1-p_0)} + z_\beta\sqrt{p_1(1-p_1)})^2}{(p_0 - p_1)^2}$$

> [http://www.itl.nist.gov/div898/handbook/prc/section2/prc242.htm](http://www.itl.nist.gov/div898/handbook/prc/section2/prc242.htm)

```r
# one-sample proportion test
prop.test(650, 1118, 0.5)
```

### 2. One-Sample Population Mean

Poulation mean has a t-distribution if the population standard deviation is unknown, with $$n - 1$$ degree of freedom.

$$t = \frac{\bar{X} - \mu}{s/\sqrt{n}}$$

Confidence interval $$\bar{x} \pm t_{\alpha/2}\cdot \frac{s}{\sqrt{n}}$$. 

**Conditions when $$t$$-Procedure is proper:**

1. When sample size is less than 15, use t-interval procedure only when population is very close to normal.
2. When sample size is between 15 and 30, it can be used if the variable is not far from normal.
3. When sample size is large, we can always use t-interval if there are no extreme outliers that cannot be removed.

If one cannot use the $$t$$ procedure, may look for a more robust procedure such as one-sample Wilcoxon procedure.

$$\bar{X}$$ is normal when $$n \ge 30$$. 

**Sample size for hypo testing**

For $$H_0: \mu = \mu_0$$ vs $$H_1: \mu = \mu_a$$, the sample size that satisfies Type I error rate $$\alpha$$ and power $$1-\beta$$ is $$N=\sigma^2 \frac{(z_{1 - \alpha/2} + z_\beta)^2}{(\mu_0-\mu_a)^2}$$, when the two boundaries are identical. 

1. Reject $$H_0$$ with significant level $$1-\alpha$$ if $$\bar{x} \ge \mu_0 + z_{1 - \alpha/2}\frac{\sigma}{\sqrt{n}}$$
2. Favor $$H_0$$ with power $$\beta$$ if $$\bar{x} \ge \mu_A - z_{\beta}\frac{\sigma}{\sqrt{n}}$$

> [https://onlinecourses.science.psu.edu/stat414/node/306](https://onlinecourses.science.psu.edu/stat414/node/306)

### 3. Two Sample Tests

#### Difference Between Two Population Propotions

Point estimate between the two proportions $$\hat{p}_1-\hat{p}_2=\frac{x_1}{n_1}-\frac{x_2}{n_2}$$
    
Standard deviation is given by  $$se(\hat{p}_1-\hat{p}_2)=\sqrt{\frac{\hat{p_1}(1-\hat{p_1})}{n_1}+\frac{\hat{p_2}(1-\hat{p_2})}{n_2}}$$
    
When the observed number of successes and the observed number of failures are greater than or equal to 5 for both populations, then the sampling distribution is approximately normal.
    
The confidence interval is given by $$\hat{p}_1-\hat{p}_2 \pm z_{\alpha/2} \cdot s.e.(\hat{p}_1-\hat{p}_2)$$
    
The hypothesis $$p_1 = p_2$$ can be tested using $$z^{*} = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1 - \hat{p})(\frac{1}{n_1}+\frac{1}{n_2})}}$$, where $$\hat{p} = \frac{x_1 + x_2}{n_1 + n_2}$$ (pooled variance)

> Equivalence to statistical independence test. The statistics $$z^2 = \chi^2$$.  

#### Difference Between Two Population Means

**Independent sampling: Two samples mean test**
    
Point estimate: $$\bar{x}_1 - \bar{x}_2$$
    
Standard deviation: $$se(\bar{x}_1 - \bar{x}_2) = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}(\frac{1}{n_1} + \frac{1}{n_2})}$$ with $$df = n_1 + n_2 - 2$$
    
> When the sample sizes are nearly equal, then a good Rule of Thumb to use is to see if this ratio falls from 0.5 to 2 (that is neither sample standard deviation is more than twice the other).
    
**Paired sampling**

$$\bar{d}\pm t_{\alpha/2} \cdot \frac{s_d}{\sqrt{n}}$$
    
> Assumption: the differences of the pairs follow a normal distribution or the number of pairs is large (note here that if the number of pairs is < 30, we need to check whether the differences are normal, but we do not need to check for the normality of each population)
    
#### Two Population Variances
    
Under $$H_0: \sigma_1^2 = \sigma_2^2$$

1. The F-test: assumes the two samples come from populations that are normally distributed. $$F^{*} = \frac{s_1^2}{s_2^2} \sim F(n_1 - 1, n_2 - 1)$$
2. Bonett's test: this assumes only that the two samples are quantitative. 
3. Levene's test: similar to Bonett's in that the only assumption is that the data is quantitative. Best to use if one or both samples are heavily skewed and your two sample sizes are both under 20.

#### Ratio of Two Proportions

> Relative Risk. It is a better measure of association than the difference in proportions when cell probabilities are close to 0 and 1, i.e., when they are in the tails of the probability distribution.

The standard way to do this is to first log-transform the ratio, calculate a confidence interval on the log scale using the delta method and assuming a normal distribution, then transform back. This works better in moderate sample sizes than using the delta method on the untransformed scale, though it will still behave poorly if the number of events in either group is very small, and fails completely if there are no events in either group.

If there are $x_1$ and $x_2$ successes in the two groups out of totals $n_1$ and $n_2$, then the obvious estimate for the ratio of proportions is $\hat{\theta} = \frac{x_1/n_1}{x_2/n_2}$. 
    
Using the delta method and assuming the two groups are independent and the successes are binomially distributed, you can show that

$$\hat{Var}(\log \theta) = \frac{1}{x_1} - \frac{1}{n_1} + \frac{1}{x_2} - \frac{1}{n_2}$$

Taking the square-root of this gives the standard error $se(\log\hat{\theta})$. Assuming that $\log\hat{\theta}$ is normally distributed, a $95\%$ confidence interval for $\log\theta$

$$\log\hat{\theta} \pm 1.96 se(\log\hat{\theta})$$

Exponentiating this gives a 95% confidence interval for the ratio of proportions $\theta$ as $\hat{\theta}\cdot \exp[\pm 1.96 se(\log\hat{\theta})]$.

> [http://stats.stackexchange.com/questions/21298/confidence-interval-around-the-ratio-of-two-proportions](http://stats.stackexchange.com/questions/21298/confidence-interval-around-the-ratio-of-two-proportions)

#### Odds ratio

Odds ratio is $\theta = \frac{\pi_{11}/\pi_{21}}{\pi_{12}/\pi_{22}} = \frac{\pi_{11}\pi_{22}}{\pi_{12}\pi_{22}}$. The natural estimation is the cross-product ratio $\frac{n_{11}n_{22}}{n_{21}n_{12}}$. 

Take the log transformation, the approximated variance of $\log\hat\theta $ is $\hat{Var}(\log \hat\theta) = \frac{1}{n_{11}}+\frac{1}{n_{11}}+\frac{1}{n_{11}}+\frac{1}{n_{11}}$
    
> Odds ratio is useful in retrospective study, since odds ratio is invariant to exchanging $Y$ (cause, e.g. smoking) and $Z$ (effect, e.g. disease). In retrospective study, however, the relative risk cannot be estimated. 

#### Two way tables Dependent samples (paired data)

In dependent samples, each observation in one sample can be paired with an observation in the other sample. The interesting question is to compare the margin of the table. (e.g., whether the population have the same opinion over time). 

> **McNemar Test**, **Loglinear model**, **Logit model**

Test of marginal homogenity, in a $2\times 2$ table, is to test the off-diagonal probabilities are equal. 

In McNemar test, suppose the total number of observations in teh off-diagonal as fixed, $n^* = n_{12} + n_{21}$. Then the null hypothesis is $p_0 = 0.5$. Thus, $z$ statistic is 

$$z = \frac{n_{12}/n^* - 0.5}{\sqrt{0.5(1-0.5)/n^*}} = \frac{n_{12} - n_{21}}{\sqrt{n_{12} + n_{21}}}$$

$z$ is approximately from standard normal distribution. 

> Longitudinal sampling is more powerful for the same reason as to pair the data whenever possible. 

> For larger table, use Cohen's Kappa test instead of McNemar test. [https://onlinecourses.science.psu.edu/stat504/node/99](https://onlinecourses.science.psu.edu/stat504/node/99)

### 4. Goodness of Fit

#### Person GOF statistics (Chi-Square test)

$\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$. 

- For a $2 \times 2$ table, the Chi-Square test is same as conducting difference-in-two-proportions test. $ E=\frac{row\ total \times column\ total}{sample\ size}$ or $E = \text{expected outcomes}$. The critical value for the Chi-Square test is $\chi^2_{\alpha}$ with degree of freedom $1$. [https://onlinecourses.science.psu.edu/stat500/sites/onlinecourses.science.psu.edu.stat500/files/lesson14/summary_table.pdf](https://onlinecourses.science.psu.edu/stat500/sites/onlinecourses.science.psu.edu.stat500/files/lesson14/summary_table.pdf)
- For a one-sample proportion, the Chi-Square test can be used in testing multinomial distributions. If $x$ is a realization of $X \sim Mult(n, \pi)$, then as $n$ becomes large, the sampling distributions of $\chi^2(x, \pi)$ approaches chi-squared distribution with $df = k-1$, where $k$ = number of cells. [https://onlinecourses.science.psu.edu/stat504/node/60](https://onlinecourses.science.psu.edu/stat504/node/60)
- For $I\times J$ table. Test is on $H_0$: independent model vs $H_1$: saturated model. $E = \frac{row\ total \times column \ total}{ grand \ total}$. Degree of freedom. Under $H_0$, there are $IJ - 1$ free parameters. Under $H_1$, there are $(I-1) + (J-1)$ parameters. The difference are $(I-1)(J-1)$
    
#### Deviance Statistics (LR test)

$G^2 = 2\sum O_i \log \frac{O_i}{E_i}$. 

- In some texts, G2 is also called the likelihood-ratio test statistic, for comparing the likelihoods ($L_0$ and $L_1$) of two models, that is comparing the loglikelihoods under $H_0$ (i.e., loglikelihood of the fitted model, $l_0$) and loglikelihood under $H_A$ (i.e., loglikelihood of the larger, less restricted, or saturated model $l_1$): $G_2 = -2\log(L_0/L_1) = -2(l_0 - l_1)$.

```r
#### one-sample proportion test

x <- 650
n <- 1118
p0 <- 0.5

prop.test(x, n, p0)

#### confidence interval 
p_bar = x / n 
ci = c(p_bar - 1.96 * sqrt(p_bar * (1 - p_bar) / n), 
       p_bar + 1.96 * sqrt(p_bar * (1 - p_bar) / n))

pval <- 2 * (1 - pt(abs(p_bar - p0) / sqrt(p0 * (1 - p0) / n), df = n - 1))

#### multinomial, chisq
x <- c(3, 7, 5, 10, 2, 3)
x_exp <- mean(x)

chi_stat <- sum((x - x_exp)^2 / x_exp)
dev_stat <- 2 * sum(x * log(x / x_exp))
chi_pval <- 1 - pchisq(chi_stat, 5)
dev_pval <- 1 - pchisq(dev_stat, 5)

x <- c(926, 288, 293, 104)
x_exp <- c(9/16, 3/16, 3/16, 1/16) * sum(x)

chi_stat <- sum((x - x_exp)^2 / x_exp)
dev_stat <- 2 * sum(x * log(x / x_exp))

chi_pval <- 1 - pchisq(chi_stat, 3)
dev_pval <- 1 - pchisq(dev_stat, 3)
```

#### Goodness of fit for probability functions of unknown parameters 
[https://onlinecourses.science.psu.edu/stat504/node/63](https://onlinecourses.science.psu.edu/stat504/node/63)

Example 1. Hardy-Weinberg problem. 

> genotype | proportion 
> --- | --- 
AA | $\pi^2$
Aa | $2\pi ( 1- \pi)$
aa | $(1-\pi)^2$

Example 2. Poisson distribution with $\lambda$

> Steps to conduct goodness of fit test. 
1. Estimate $\hat\theta$
2. Calculate cell probability $\hat \pi = g(\hat\theta)$
3. Calculate the goodness of fit statistics $\chi^2$ and $G^2$. $\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$ and $G^2 = 2\sum O_i \log \frac{O_i}{E_i}$, where $E_i = n \pi_i$.

```r
cnt <- c(19, 26, 29, 13, 13)

lambda_bar <- sum((0:4) * cnt) / sum(cnt)

p_bar <- dpois(0:4, lambda_bar)
p_bar[5] <- 1 - sum(p_bar[1:4])

cnt_exp <- p_bar * sum(cnt)

# chisq 
chi_stat <- sum((cnt - cnt_exp)^2 / cnt_exp)
dev_stat <- 2 * sum(cnt * log(cnt / cnt_exp))

1 - pchisq(chi_stat, length(cnt) - 1)
1 - pchisq(dev_stat, length(cnt) - 1)
```

> 1. Note that if any element of vector is zero, then $\chi^2$ and $G^2$ both break down. One simple possible way to avoid such problems is to put a very small mass in all the cells, including the zero-cell.
> 2. The $\chi^2$ and $G^2$ test statistics are not appropriate measures of association between two variables. They are sufficient to test the null hypothesis, but not to describe the direction and magnitude of association.

### 5. P-value 

The p-value is defined as the probability of obtaining a result equal to or "more extreme" than what was actually observed, when the null hypothesis is true.  If the p-value is less than 0.05 or 0.01, corresponding respectively to a 5% or 1% chance of rejecting the null hypothesis when it is true (Type I error).

### 6. Confidence interval 

A $95\%$ confidence interval is defined as a range of values such that with $95\%$ probability, the range will contain the true unknown value of the parameter. Or We are 95% confident that the true value is within this interval. 

### 7. Non-parametric Tests 

#### Wilcoxon rank-sum test (for two sample)

Assume that no two observations have the same value so that the ranks are distinct. Also assume that treatment 1 has m observations and treatment 2 has n observations.

1. Combine the m + n observations into one group and rank the observations from smallest to largest. Find the observed rank sum, W, of treatment 1.
2. Find all the possible permutation of the ranks into which m ranks are assigned to treatment 1 and n ranks are assigned into treatment 2.
3. For each permutation of the ranks, find the sum of the ranks for treatment 1.
4. Determine the p-value: $P_{upper}=\frac{\text{#of rank sums ≤observed rank sum}}{\text{m + n choose n}}$

> For ties, group all the tied observations and assign the average rank to tied values in that group. Call these ranks adjusted ranks.

```
## wilcoxon test
sample1 <- rnorm(10) + 0.5
sample2 <- rnorm(12)
wilcox.test(sample1, sample2)
```

#### Mann-Whitney Test

Suppose we have the following set-up:
Treatment 1:  $x_1, x_2, ... , x_m$
Treatment 2:  $y_1, y_2, ... , y_n$

Assume that there are no ties in the data. This means that any given observation is either strictly less than or strictly greater than any other observation. Consider the statistic: $U = \text{# of pairs of (Xi, Yj)}, that \ X_i < Y_j$

The null hypothesis for this test is the distributions are the same. Large values of U indicate that the values with treatment 2 tend to larger than those from treatment 1 and vise versa if the value of U is small. 

> Mann-Whitney and Wilcoxon are equivalent. 

Confidence interval. $P(k_a\le U\le k_{b-1})\approx P\left(\frac{k_a-E(U)}{\sqrt{Var(U)}}\le Z\le \frac{k_{b-1}-E(U)}{\sqrt{Var(U)}}\right)$, where $Var(U) = \frac{mn(M + 1)}{12}$

#### Komogorov test (on two distributions)

Test if two distributions are the same. Suppose two samples, $X_1, \dots X_m$ and $Y_1, ..., Y_n$. Denote $F_1(x)$ and $F_2(y)$ the cdf. The Kolmogorov-Smirnov statistic, is $KS = \max_w |F_1(w) - F_2(w)|$. Steps are 
1 Calculate the observed test statistic, $K_{obs}$.
2. Find all the possible permutation of the data and calculate KS for each permutation.
3. The p-value is found by $p-value=\frac{\text{# of KS≥KSobs}}{\text{total # of permutations}}$

### 8. What if the data is not normal?

1. if your data are not normal, you should do a nonparametric version of the test, which does not assume normality. 
2. some tests are robust even if the normal assumption is invalid. t-tests (1-sample, 2-sample, and paired t-tests), Analysis of Variance (ANOVA), Regression, and Design of Experiments (DOE). 

> [http://blog.minitab.com/blog/understanding-statistics-and-its-application/what-should-i-do-if-my-data-is-not-normal-v2](http://blog.minitab.com/blog/understanding-statistics-and-its-application/what-should-i-do-if-my-data-is-not-normal-v2)

## Part 2. Linear Regression 

### 1. Assumption on linear model: LINE

1. Linearity. The mean of response, $\mathbb E[Y]$ is a Linear function of $x_i$.
2. The errors, $\epsilon_i$, are Independent
3. The errors, $\epsilon_i$, at each value of the predictor, $x_i$, are of Normal distribution. 
4. The errors, $\epsilon$ at each value fo the predictor, $x_i$, have equal variance. $\epsilon_i \sim N(0, \sigma^2)$

> What if $X$ is random?
> Suppose $X_i$ is sampled from $g(X)$, we can still regress $X$ on $Y$ if 
> 1) Conditional distribution of $Y_i$ given $X_i$ are normal and independent with conditional means $\beta_0 + \beta_1 X_i$ and conditional variance $\sigma^2$
> 2) The $X_i$ are independent and $g(X_i)$ does not involve the parameters $\beta_0$, $\beta_1$, and $\sigma^2$.

### 2. Properties on residuals $e_i = Y_i - \hat{Y}_i$.

1. $\sum e_i = 0$
2. $\sum e_i^2$ is minimized
3. $\sum X_i e_i = 0$
4. $\sum Y_i = \sum \hat{Y}_i$
5. $\sum \hat{Y}_i e_i = 0$

### 3. Inference of $e$, $b$, $\hat{Y}$

**Residuals**
$e = (I - H)Y$
$\mathbb E[e] = 0$ and $Var(e) = (I - H)\sigma^2$

**Studentized residuals** (or internally studentized residuals) 
For each observation, an ordinary residual divided by an estimate of its standard deviation.
$r_{i}=\frac{e_{i}}{s(e_{i})}=\frac{e_{i}}{\sqrt{MSE(1-h_{ii})}}\sim t(n-p)$ distribution.

> It's useful in identifying outliers.

**Estimation $b$**
$b = (X'X)^{-1}X'Y$
$\mathbb E[b] = \beta$
$Var(b) = (X'X)^{-1}\sigma^2$
Test $H_0: \beta_i = 0$ use $\frac{b_i}{s(b_i)} \sim t(n - k)$

**Mean response $\hat{Y}_h$**
$\hat{Y}_h = X'_hb$
$\mathbb E[\hat{Y}_h] = X'_h\beta$
$Var(\hat{Y}_h)=X'_h(X'X)^{-1}X_h\sigma^2$
Working Hotelling Band $W^2 = 2F(1 - \alpha, 2, n - 2)$ covers all $X_h$, $Pr(|\hat{Y}_h - Y_h| \le W \cdot s(\hat{Y}_h), \forall X_h) \ge 1 - \alpha$.

**Prediction $\hat{Y}_p$**
$\mathbb E[\hat{Y}_p] = X'_p\beta$
$Var(\hat{Y}_h)= (1 + X'_p(X'X)^{-1}X_p)\sigma^2$
    
### 4. Sum of square

- SSTO: Total sum of square: $SSTO = \sum (Y_i - \bar{Y})^2$
- SSE: Sum of square error: $SSE = \sum (Y_i - \hat{Y}_i)^2$, $MSE = \frac{\sum (Y_i - \hat{Y}_i)^2}{n - k}$
- SSM: Model sum of square: $SSR = \sum (\hat{Y}_i - \bar{Y})^2$, $MSR = \frac{\sum (\hat{Y}_i - \bar{Y})^2}{k - 1}$

> If $\beta_1 = 0$, $\mathbb E(MSR) = \sigma^2 + \beta_1^2 \sum(X_i - \bar{X})^2 = \sigma^2$, thus, $MSR / MSE$ can be used to test $\beta_1 = 0$ and $\frac{MSR}{MSE} \sim F_{1, n - 2}$.

- Type 1 SS: Seq Sum of Square. Can be used in test on significance of a single $\beta$ or several $\beta$s together. 
- Type 3 SS: Adj Sum of Square

Partial $R^2$: $R^2_{B|A} = \frac{SSR(B|A)}{SSE(A)} = \frac{SSE(A) - SSE(A,B)}{SSE(A)}$

### 5. Lack of fit test

There are two ways to think of lack of fit test. One is to decompose the residual errors into two terms, lack of fit and pure error. The larger the lack of fit, the unlikely the linear model is a good fit. The second way is to compare a reduced model (linear model) with a full model (cell mean model). If the linear relationship is valid, then the difference in sum of square is merely a random error.

Source of Variation | DF | SS | MS | $F$ statistics 
------------------- | -- | -- | -- | ----
Regression     | $p-1$  | $SSM = \sum_i \sum_j (\hat{y}_{ij} - \bar{y})^2$ | $\frac{SSM}{p - 1}$ | $F = \frac{MSM}{MSR}$
Residual Error | $n - p$ | $SSE = \sum_i \sum_j (\hat{y}_{ij} - y_{ij})^2$ | $\frac{SSE}{n - p}$ |
Lack of fit | $m - p$ | $SSLF = \sum_i \sum_j (\hat{y}_{ij} - \bar{y_i})^2$ | $\frac{SSLF}{m - p}$ | $F = \frac{MSLF}{MSPL}$
Pure Error | $n - m$ | $SSPL = \sum_i \sum_j (y_{ij} - \bar{y}_i)^2$ | $\frac{SSPL}{n - m}$ | 
Total | $n - 1$ | $SSTO = \sum_i \sum_j (y_{ij} - \bar{y})^2$


Full model: $Y_{ij}=\mu_i + \epsilon_{ij}$
Reduced model: $Y_{ij} = X\beta + \epsilon_{ij}$
$$F^{\star} = \frac{(SSE(R) - SSE(F))/(c - p)}{SSE(F)/(n - c)}$$

### 6. Assumption Diagnostic

#### 1. Summary
    
Plots
1. Scatter plot $Y$ vs $X$
2. Predictor $X$
3. Residual $e$ vs predictor $X$
4. Residual $e$ vs fitted value $\hat{Y}$
    
Tests
1. Shapiro-Wilk for normality
2. Durbin-Watson for correlation / autocorrelation 
3. Modified Levene for constant variance 
4. VIF for multicollinearity: VIF is the kth diagonal element of $r^{-1}_{XX}$
5. Cook's distance, DFFITS, DFBETAS for Influential points
    
Remedies
1. Nonlinear relationship: transform $X$, or nonlinear regression
2. Nonconstant variance: transform $Y$, weighted least square
3. Nonnormal errors: transform $Y$, generalized linear model
4. Nonindependent: difference
    
> [http://www.stat.purdue.edu/~wsharaba/stat512/topic5.pdf](http://www.stat.purdue.edu/~wsharaba/stat512/topic5.pdf)

#### 2. Normality

Plot eye-test: histogram, quantile plot, scatter plot

Formal test: Shapiro-Wilk. `shapiro.test` tests the Null hypothesis that "the samples come from a Normal distribution" against the alternative hypothesis "the samples do not come from a Normal distribution". Significance tests results are very dependent on the sample size; with sufficiently large samples we can reject slight deviations from null hypothesis that would not invalidate results if ignored. Plots are more likely to suggest a remedy if one is needed.

> [http://stackoverflow.com/questions/7781798/seeing-if-data-is-normally-distributed-in-r/7788452#7788452](http://stackoverflow.com/questions/7781798/seeing-if-data-is-normally-distributed-in-r/7788452#7788452)

#### 3. Constant variance

Test: **Modified Levene test**. This test does not require the error terms to be drawn from a normal distribution and hence it is a nonparametric test. The test is constructed by grouping the residuals into g groups according to the values of the quantity on the horizontal axis of the residual plot. It is typically recommended that each group has at least 25 observations and usually $g=2$ groups are used.

Begin by letting group 1 consist of the residuals associated with the $n_1$ lowest values of the predictor. Then, let group 2 consists of the residuals associated with the $n_2$ highest values of the predictor (so $n_1+n_2=n$). The objective is to perform the following hypothesis test: $H_0$: the variance is constant.

```r
library(car)
sample1 <- rnorm(100)
sample2 <- rnorm(100)
res <- data.frame(
    GroupID = as.factor(c(rep(1, length(sample1)), rep(2, length(sample2)))), 
    DV = c(sample1, sample2)
)
leveneTest(DV ~ GroupID, data = res)
```

Remedies:
1. Box-cox for non-normal or non-constant variance. ML to estimate the optimal transformation of $Y$
```
boxcox(Volume ~ log(Height) + log(Girth), data = trees,
     lambda = seq(-0.25, 0.25, length = 10))
```
2. Weighted least square. Solve weighted least square can use IRWLS (iteratively reweighted least square) method, that regress x on $\epsilon^2$ iteratively. IRWLS may be useful when the errors are uncorrelated, but have unequal variance where the form of inequality is unknown 
3. Linear mixed model

#### 4. Multicollinearity

A little bit of multicollinearity isn't necessarily a huge problem. Severe multicollinearity increases the variance of the regression coefficients, making them unstable. The more variance they have, the more difficult it is to interpret the coefficients. Signals:
    1. A regression coefficient is not significant even though, theoretically, that variable should be highly correlated with Y.
    2. When you add or delete an X variable, the regression coefficients change dramatically.
    3. You see a negative regression coefficient when your response should increase along with X. You see a positive regression coefficient when the response should decrease as X increases.
    4. Your X variables have high pairwise correlations. 
    
Test: variance inflation factor (VIF), $VIF_k = \frac{1}{1-R_k^2}$. 

VIF assesses how much the variance of an estimated regression coefficient increases if your predictors are correlated. If no factors are correlated, the VIFs will all be 1. A VIF between 5 and 10 indicates high correlation that may be problematic. And if the VIF goes above 10, you can assume that the regression coefficients are poorly estimated due to multicollinearity.

Remedy:
1. Remove highly correlated predictors from the model. If you have two or more factors with a high VIF, remove one from the model. Because they supply redundant information, removing one of the correlated factors usually doesn't drastically reduce the R-squared.  Consider using stepwise regression, best subsets regression, or specialized knowledge of the data set to remove these variables. Select the model that has the highest R-squared value. 
2. Use Partial Least Squares Regression (PLS) or Principal Components Analysis, regression methods that cut the number of predictors to a smaller set of uncorrelated components.

> [http://stats.stackexchange.com/questions/86269/what-is-the-effect-of-having-correlated-predictors-in-a-multiple-regression-mode](http://stats.stackexchange.com/questions/86269/what-is-the-effect-of-having-correlated-predictors-in-a-multiple-regression-mode)

#### 5. Outliers vs High Leverage points

An outlier is a data point whose response y does not follow the general trend of the rest of the data. To identify outliers, we can look into the studentized residuals or studentized deleted residuals.

> Treatment towards outliers include: winsorize (truncate data at some threshold), transform (box-cox), remove outliers (be certain they are anomalies and not worth predicting)

A data point has high leverage if it has "extreme" predictor x values. With a single predictor, an extreme x value is simply one that is particularly high or low.

**Outliers are detected by Studentized error.** 

**Difference in fits (DFFITS)** The basic idea is to delete the observations one at a time, each time refitting the regression model on the remaining $n–1$ observations. Then, we compare the results using all $n$ observations to the results with the ith observation deleted to see how much influence the observation has on the analysis. Analyzed as such, we are able to assess the potential impact each data point has on the regression analysis.
$$DFFITS_i=\frac{\hat{y}_i - \hat{y}_{(i)}}{\sqrt{MSE_{(i)}h_{ii}}}$$
    
The numerator measures the difference in the predicted responses obtained when the ith data point is included and excluded from the analysis. The denominator is the estimated standard deviation of the difference in the predicted responses. Therefore, the difference in fits quantifies the number of standard deviations that the fitted value changes when the ith data point is omitted.

**Cook's distance** $$D_i=\frac{(y_i-\hat{y}_i)^2}{p \times MSE}\left[ \frac{h_{ii}}{(1-h_{ii})^2}\right]$$ 

$D_i$ directly summarizes how much all of the fitted values change when the ith observation is deleted. A data point having a large $D_i$ indicates that the data point strongly influences the fitted values. If $D_i$ is greater than 1, then the $i^{th}$ data point is quite likely to be influential.

In R, `cooks.distance(fit)`

**Hat Matrix Diagonals** 
$h_{ii}$ is a measure of how much $Y_i$ is contributing to the prediction of $\hat{Y}_i$. This depends on the distance between the $X$ values for the ith case and the means of the $X$ values. Observations with extreme values for the predictors will have more influence. 
A large value of $h_{ii}$ suggests that the $i^{th}$ case is distant from the center of all X’s. The average value is $p/n$. Values far from this average (say, twice as large) point to cases that should be examined carefully because they may have a substantial influence on the regression parameters. A common rule is to flag any observation whose leverage value, hii, is more than 3 times larger than the mean leverage value. 

Note that the leverage merely quantifies the potential for a data point to exert strong influence on the regression analysis. The leverage depends only on the predictor values. Whether the data point is influential or not also depends on the observed value of the reponse $y_i$.

#### 6. Autocorrelation 

Durbin Watson test on autocorrelation. Test statistic: 

$$D = \frac{\sum (e_{t} - e_{t-1})^2}{\sum e_t^2}$$

When autocorrelated error terms are found to be present, then one of the first remedial measures should be to investigate the omission of a key predictor variable. If such a predictor does not aid in reducing/eliminating autocorrelation of the error terms, then certain transformations on the variables can be performed. 

Remedy. 

The Hildreth-Lu procedure is a more direct method for estimating $\rho$. After establishing that the errors have an AR(1) structure, follow these steps:

1. Select a series of candidate values for $\rho$ (presumably values that would make sense after you assessed the pattern of the errors).
2. For each candidate value, regress $y_t^{\star}$ on the transformed predictors using the transformations established in the Cochrane-Orcutt procedure. Retain the SSEs for each of these regressions.
3. Select the value which minimizes the SSE as an estimate of $\rho$.

### 7. Model selection 


> 1. Subset selection 
    > Best subset selection: Fit all $p \choose k$ models for all $k = 1, ..., p$. Pick the best among these models, using CV prediction error, $C_p$ (AIC), BIC or adjusted-$R^2$.
    > Stepwise selection
2. Shrinkage
3. Dimension Reduction

> Large data set, high dimension or large sample size?
> If high dimension, start from simple univariate filter, remove variables that are not significant, say correlation that is less than a threshold. Then take care of multicollinearity if there is any. Then model selection procedure.


**Adjusted $R^2$.** 

The adjusted R2-value increases only if MSE decreases. That is, the adjusted R2-value and MSE criteria always yield the same "best" models.
$Adjust-R^2 = 1 - \frac{n - 1}{n - k}\frac{SSE}{SSTO} = 1 - \frac{MSE}{MSTO}$
    
**$C_p$**
$C_p = \frac{1}{n}(RSS + 2d\hat \sigma^2)$

Strategy for using $C_p$ to identify "best" models:

- Identify subsets of predictors for which the Cp value is near p (if possible).
- The full model always yields $C_p=p$, so don't select the full model based on $C_p$.
- If all models, except the full model, yield a large $C_p$ not near $p$, it suggests some important predictor(s) are missing from the analysis. In this case, we are well-advised to identify the predictors that are missing!
- If a number of models have $C_p$ near p, choose the model with the smallest $C_p$ value, thereby insuring that the combination of the bias and the variance is at a minimum.
- When more than one model has a small value of $C_p$ value near $p$, in general, choose the simpler model or the model that meets your research needs.

**AIC (Akaike’s Information Criterion)**
$AIC = -2\log(L) + 2p$, minimizing $-2\log(L)$ plus a penalty for more complex model.  the model with the lower value is preferred.

**BIC (Bayesian Information Criterion)**
$BIC = -2\log(L) + p \log n$, different penalty.

**Cross Validation**

1. Divide the data to $K$ folds
2. Decide on candidate values of $\lambda$
3. For each fold $k$ and value of $\lambda$, estimate $\beta$ on the out-of-fold sample, for each $x_n$ assigned to fold $k$, compute its squared error $\epsilon = (\hat{y}_n - y_n)^2$
4. Aggregate individual errors, the score for $\lambda$ is $MSE = \frac{1}{N}\sum \epsilon_n^2$
> CV provides a direct estimate of the test error, and doesn't require an estiamte of the error variance $\sigma^2$. 

### 8. Robust Regression

> Mean Squared Error vs Mean Absolute Error

- Robust regression methods provide an alternative to least squares regression by requiring less restrictive assumptions. These methods attempt to dampen the influence of outlying cases in order to provide a better fit to the majority of the data.

- Outliers have a tendency to pull the least squares fit too far in their direction by receiving much more "weight" than they deserve. Typically, you would expect that the weight attached to each observation would be on average $1/n$ in a data set with $n$ observations. However, outliers may receive considerably more weight, leading to distorted estimates of the regression coefficients. This distortion results in outliers which are difficult to identify since their residuals are much smaller than they would otherwise be (if the distortion wasn't present).

- Ordinary least squares is sometimes known as $L_2$-norm regression since it is minimizing the $L_2$-norm of the residuals (i.e., the squares of the residuals). Thus, observations with high residuals (and high squared residuals) will pull the least squares fit more in that direction. An alternative is to use what is sometimes known as least absolute deviation (or $L_1$-norm regression)

### 9. Penalized linear regression 

> Lasso, ridge, and elastic net 

> Shrink the coefficient estimates can significantly reduce the variance. 

> A decomposition of square errors is known as bias-variance tradeoff. As model becomes more complex, local structure can be picked up; but the coefficient estimates suffer from high variance as more terms are included in the model. So introducing a little bias in the estimate might lead to a substantial decrease in variance, and hence a substantial decrease in prediction error. 

1. The main difference between Lasso and Ridge is the penalty term they use. Rigde regression uses $L_2$ penalty term which limits the size of coefficient vector. Lasso uses $L_1$ penalty which imposes sparsity among the coefficients and thus, makes the fitted model more interpretable. Elasticnet is introduced as a compromise between these two techniques, and has a penalty which is a mixed of $L_1$ and $L_2$ norms. 
2. Lasso does a sparse selection, while Ridge does not.
3. When you have highly-correlated variables, Ridge regression shrinks the two coefficients towards one another. Lasso is somewhat indifferent and generally picks one over the other. Depending on the context, one does not know which variable gets picked. Elastic-net is a compromise between the two that attempts to shrink and do a sparse selection simultaneously.
4. Ridge estimators are indifferent to multiplicative scaling of the data. That is, if both X and Y variables are multiplied by constants, the coefficients of the fit do not change, for a given $\lambda$ parameter. However, for Lasso, the fit is not independent of the scaling. In fact, the $\lambda$ parameter must be scaled up by the multiplier to get the same result. It is more complex for elastic net.
5. Ridge penalizes the largest $\beta$'s more than it penalizes the smaller ones (as they are squared in the penalty term). Lasso penalizes them more uniformly. This may or may not be important. In a forecasting problem with a powerful predictor, the predictor's effectiveness is shrunk by the Ridge as compared to the Lasso.
6. Ridge regression trade off bias with variance. $\beta^{ridge} = \frac{\beta^{OLS}}{1 + \lambda}$. Variance of $\beta$, $Var(\beta) = \sigma^2 WX'XW$, where $W = (X'X + \lambda I) ^{-1}$
7. More often, Lasso performs better than Ridge when the response is a function of only a subset of $X$

### 10. Big Data Regression 

There are a couple of ways to do this, and it will generally depend on the dimensions of your data, and your environment.  There are several matrix factorization approaches that work well in shared-memory multiprocessors, but may not be viable for something such as MapReduce. 

A common approach is to parallelize the matrix multiplication needed to get $(X'X)^{-1}$ and $X'Y$. However, by a series of parallelized QR decompositions, it is possible to solve the least squares problem. The following is a useful reference: Communication-avoiding Parallel and Sequential QR Factorizations.

Parallelizing Matrix Multiplication

Two matrices are needed, $X'X$ and $X'Y$, each of which can be parallelized by distributing blocks of rows, observations, over several processors.  Each processor computes these matrices in parallel, and the final result can then be achieved by aggregating each of these subproblems. The problem is then solved on a single-machine. The symmetry of the $X'X$ makes the Cholesky decomposition particularly attractive. This is a pretty common implementation in a MapReduce setting. 

QR Decompositions in Parallel

With a QR decomposition, we can decompose a matrix $X=QR$.  Substituting the QR factorization into the original least-squares equation gives $R\beta = Q'Y$. By splitting $X$ across several machines, we can compute the QR decomposition on each of these subproblems, or "local" problems. A subsequent QR decomposition can be run by aggregating the R terms from each of the "local" problems, which gives the final $R$ matrix. The QQ matrix can be obtained by manipulating the two $Q$ matrices from the "local" and "intermediate" problems.

## Part 3. Analysis of Variance and Experimental Design

### 1. ANOVA

When we have more than two groups we cannot use the t test, instead we have to use analysis of variance (ANOVA). In one way ANOVA we have one continuous dependent variable and one independent grouping variable or factor. When we have two groups the t test and one way ANOVA are equivalent.

For our one way ANOVA results to be valid there are several assumptions that need to be satisfied. These assumptions are listed below.

- The dependent variable is required to be continuous
- The independent variable is required to be categorical with 2 or more categories.
- The dependent and independent variables have values for each row of data.
- Observations in each group are independent.
- The dependent variable is approximately normally distributed in each group.
- There is approximate equality of variance in all the groups.
We should not have any outliers

> When our data shows non-normality, unequal variance or presence of outliers you can transform your data or use a non-parametric test like Kruskal-Wallis. It is good to note Kruskal-Wallis does not require normality of data but still requires equal variance in your groups.

- Tukey HSD is to test all pairwise comparison in ANOVA
- Kruskal-Wallis is non-parametric method for the pairwise comparisons.

**One way ANOVA**
$Y_{ij} = \mu + \tau_i + \epsilon_{ij}$
$Y_{ij}$ are independent, $\epsilon_{ij} \sim N(0, \sigma^2)$
Test with $F^{\star} = \frac{MSR}{MSE} \sim F(r - 1, n - r)$ 
    
ANOVA table
source | DF | SS
--- | --- | ---
Model | $r-1$ | $\sum_in_i(\bar{Y}_{i.} - \bar{Y}_{..})^2$
Error | $n-r$ | $\sum_i\sum_j(Y_{ij} - \bar{Y}_{i.})^2$
Total | $n-1$ | $\sum_i\sum_j(Y_{ij} - \bar{Y}_{..})^2$
    
**Two way ANOVA**
$Y_{ijk} = \mu + a_i + b_j + (ab)_{ij} + \epsilon_{ijk}$
where $\sum_i a_i = 0$, $\sum_j b_j = 0$, $\sum_i (ab)_{ij}=0$, and $\sum_j (ab)_{ij} = 0$

Two-way ANOVA table 
source | DF | SS
--- | --- | ---
A | a - 1 | $bn \sum_i (\bar{Y}_{i..} - \bar{Y}_{...})^2$
B | b - 1 | $an \sum_j (\bar{Y}_{.j.} - \bar{Y}_{...})^2$
AB | (a-1)(b-1) | $n \sum_i \sum_j (\bar{Y}_{ij.}-\bar{Y}_{i..}-\bar{Y}_{.j.} + \bar{Y}_{...})^2$
Error | ab(n - 1) | $\sum_i\sum_j \sum_k (Y_{ijk} - \bar{Y}_{ij.})^2$

**Nested Design**

$Y_{ijk} = \mu + \alpha_i + \beta_{j(i)} + \epsilon_{ijk}$

source | DF
---- | ---
A | $a - 1$
B | $a(b - 1)$
Error | $ab(n - 1)$
Total | $abn - 1$

> Test using Tukey, Scheffe, Bonferroni

### 2. Random effect model 

Want to draw inference on population of levels
$Y_{ij} = \mu + \tau_i + \epsilon_{ij}$, where $\tau_i \sim N(0, \sigma_{\mu}^2)$ and $\epsilon_{ij} \sim N(0, \sigma^2)$
$Var(Y_{ij}) = \sigma_{\mu}^2 + \sigma^2$
$Cov(Y_{ij}, Y_{ik}) = \sigma_{\mu}^2$

Source | EMS (Expected Means Square) |
--- | --- | --- 
Factor | $n\sigma_{\mu}^2 + \sigma^2$ | 
Error  | $\sigma^2$ | 


- Estimation of $\sigma_{\mu}^2$ is given by $s^2_{trt} = \frac{MS_{trt} - MS_{error}}{n}$
- ICC (interclass correlation coefficients): $\frac{s^2_{trt}}{s^2_{trt} + s^2_{e}}$. Small values of ICC indicate a large spread of values at each level of the treatment, whereas large values of ICC indicate relatively little spread at each level of the treatment:
- Test $\sigma_{\mu}^2 = 0$ using $F^{\star} = \frac{MSTR}{MSE} \sim F(r - 1, n - r)$
- Confidence interval of $\sigma^2$ using $\frac{n(r-1)MSE}{\sigma^2} \sim \chi^2_{r(n - 1)}$
- Confidence interval of $\sigma^2_{\mu}$ using $\frac{(r-1) MSTR}{\sigma^2 + n\sigma^2_{\mu}} \sim \chi^2_{r - 1}$

```
library(lme4)
fit <- lmer(y ~ 1 + (1|x), data = data, REML = FALSE)
fixef(fit) # fixed effect
ranef(fit) # 
vcov(fit)
fitted(fit) # yhat
```

**Random Effect in Factorial Design**

Source | EMS | F
--- | --- | --- 
A | $nb\sigma_a^2 + n \sigma_{ab}^2 + \sigma^2_{\epsilon}$ | $\frac{MS_a}{MS_{ab}}$
B | $na\sigma_b^2 + n \sigma_{ab}^2 + \sigma^2_{\epsilon}$ | $\frac{MS_b}{MS_{ab}}$
AB | $n \sigma_{ab}^2 + \sigma^2_{\epsilon}$ | $\frac{MS_{ab}}{MS_e}$
Error | $\sigma^2_{\epsilon}$ | 

**Mixed Effect Model**

Consider two factors, A and B, in a factorial design in which factor A is a fixed effect and factor B is a random effect.

Source | EMS | F
--- | --- | ---
A | $\sigma^2 + n\sigma^2_{ab} + nb \frac{\sum \mu_i^2}{a - 1}$ | $\frac{MS_a}{MS_{ab}}$
B | $\sigma^2 + na \sigma_b^2$ | $\frac{MS_b}{MS_{e}}$
AB | $\sigma^2 + n\sigma^2_{ab}$ | $\frac{MS_{ab}}{MS_e}$
error | $\sigma^2$ | 

Note that the denominator for F test for the main effect of factor A is now the MS for the A × B interaction. For Factor B and the A × B interaction, the denominator is the MSE.

[https://onlinecourses.science.psu.edu/stat502/node/172](https://onlinecourses.science.psu.edu/stat502/node/172)

### 3. Linear mixed model 

$Y = X\beta + Z\gamma + \epsilon$, where $\gamma \sim N(0, \sigma_{\mu}^2)$ and $\epsilon \sim N(0, \sigma^2)$. 

![mixed model](https://onlinecourses.science.psu.edu/stat502/sites/onlinecourses.science.psu.edu.stat502/files/lesson13/mixed_proc_equation.png)

In the case where we only had fixed effects in the model and no repeated measures, the only source of random variation was $\epsilon$.  In the case where we didn't have any fixed effects (e.g. the fully nested random effects model) we only had the estimates for $\gamma$ and $\epsilon$ we were able to compute the variance components as percentages.  Finally, when we introduced the covariance structures in repeated measures, we were specifying the terms of $\sigma_{\mu}^2$ and evaluating which covariance structure provided the best fit to the data.

### 4. ANCOVA

Linear model combining indicator variables and predictor variables.
$Y_{ij} = \mu + \tau_i + \beta(X_{ij} - \bar{X}_{..}) + \epsilon_{ij}$

General procedure: 
1. Fit one-way model (Y = trt)
2. Fit one-way model (X = trt)
3. Regress residuals

Same as in regression with factors. [http://stats.stackexchange.com/questions/45559/should-i-use-ancova-or-multiple-regression-with-dummy-variables](http://stats.stackexchange.com/questions/45559/should-i-use-ancova-or-multiple-regression-with-dummy-variables)

### 5. Experimental Design

1. Completely random design

    ![CRD][1]

2. Randomized Complete Block Design

    ![RCBD][2]
    - Blocks are usually treated as random effects

> CRD vs RCBD are similar idea to simple random sampling vs stratefied random sampling. 

## Part 4. Generalized Linear Regression 

### 1. **Assumptions**

- The data $Y_1, Y_2, ..., Y_n$ are **independently distributed**, i.e., cases are independent.
- The dependent variable $Y_i$ does NOT need to be normally distributed, but it **typically assumes a distribution from an exponential family** (e.g. binomial, Poisson, multinomial, normal,...)
- GLM does NOT assume a linear relationship between the dependent variable and the independent variables, but it does assume **linear relationship between the transformed response in terms of the link function and the explanatory variables**; e.g., for binary logistic regression $logit(p) = \beta_0 + \beta_1X$.
- Independent (explanatory) variables can be even the power terms or some other nonlinear transformations of the original independent variables.
- The homogeneity of variance does NOT need to be satisfied. In fact, it is not even possible in many cases given the model structure, and overdispersion (when the observed variance is larger than what the model assumes) maybe present.
- **Errors need to be independent** but NOT normally distributed.
- It uses **maximum likelihood estimation** (MLE) rather than ordinary least squares (OLS) to estimate the parameters, and thus relies on large-sample approximations.
- Goodness-of-fit measures rely on sufficiently large samples, where a heuristic rule is that not more than 20% of the expected cells counts are less than 5.

### 2. Logistic regression

\begin{align*}
    p(x) & = \frac{\exp(\beta'x)}{1 + \exp(\beta'x)} \Rightarrow \log \frac{p}{1 - p} = \beta'x \\
    \log L & = \log \prod p_i^{y_i} (1 - p_i)^{1 - y_i} \\
    & = \sum y_i \log \frac{p_i}{1 - p_i} + \sum \log (1 - p_i) \\
    & = \sum y_i (\beta'x_i) + \sum \log (\frac{1}{1 + \exp (\beta' x_i)}) \\
    & = \sum y_i (\beta' x_i) - \sum \log (1 + \exp (\beta' x_i)) \\
    \frac{\partial}{\partial \beta} \log L & = \sum (y_i - p_i) x_i \\
    \end{align*}
    
**Model fit**:  

- Pearson Chi-square $\sum \frac{(X_{ij} - E_i)^2}{X_{ij}}$
- Deviance: $\Lambda = 2(\log L(saturated) - \log L(reduced)) = -2 (l_{null} - l_{full})$, where $l_{null}$ is the log likelihood of the model specified by the null hypothesis evaluated at the maximum likelihood. The statistic has a $\chi^2$ distribution with $p - r$ degree of freedom. 
- Hosmer-Lemeshow test 

**Parameter estimation**

- Iteratively re-weighted least squares (Newton). This likelihood is transcental function and has no closed-form solution.
$\beta^{(n + 1)} = \beta^{(n)} + \frac{f'(\beta^{(n)})}{f''(\beta^{(n)})}$

- In logistic regression, $l'(\beta)_j = \sum_i (y_i - p_i)x_{ij}$, and $l''(\beta)_{jk} = -\sum_i p_i (1 - p_i) x_{ij}x_{ik}$

**Inference and interpretation**

- Test of significance for individual regression coefficients in logistic regression can use Wald Test. $Z = \frac{\hat{\beta}}{s.e.(\hat{\beta})}$.

- Odds ratio $\frac{p_i}{1 - p_i} = \exp(X\beta)$. An odds ratio of 1 serves as the baseline for comparison and indicates there is no association between the response and predictor. The odds increase multiplicatively by $\exp(\beta)$ for every one-unit increase in $X$.

- Residual. The raw residual is th difference between the actual response and the estimated probability from the model. The Pearson residual corrects for the unequal variance in the raw residuals by dividing the standard deviation. $\hat{r}_i = \frac{r_i}{p_i(1-p_i)}$. The other popular one is the deviance residuals. 

**Overdispersion**

- Overdispersion means that the data show evidence that the variance of the response $y_i$ is greater than $\mu_i(n_i - \mu_i) / n_i$. 
- Overdispersion occurs when the discrepancies between the observed responses $y_i$ and their predicted values $\mu_i = n_i p_i$ are larger than what the binomial model would predict. It arises when the dataset are not identically distributed, or not independent. 
- Adjust by including the scale parameter $\sigma^2$, where $\hat \sigma^2 = \chi^2 / (N - P)$
- This will make the confidence intervals wider.

**ROC curve**

ROC (Receiver Operating Characteristic) is for summarizing classifier performance over a range of trade-offs between true positive (TP) and false positive (FP) error rates. ROC curve is a plot of sensitivity (the ability of the model to predict an event correctly) versus 1-specificity for the possible cut-off classification probability values $\pi_0$. **True Positive Rate** vs **False Positive Rate**

> When the classes are well-separated, the parameter
estimates for the logistic regression model are surprisingly
unstable. 

> For a two-class problem, LDA has the same form as Logistic regression. The difference is how the parameters are estimated. 

### 3. Polytomous Regression 

**Nominal Logistic Regression** 

$P_j = \frac{\exp(X\beta_j)}{1 + \sum_{j = 2}^k \exp(X\beta_j)}$ for $j = 2, ..., k$. $\beta_1 = 0$. 

Goodness of fit test statistic: $G^2 = 2 \sum_i \sum_j y_{ij}\log \frac{y_{ij}}{p_{ij}}$ with $df = (N - p)(r - 1)$. 

```r
library(VGAM)
```

**Ordinal Logistic Regression** 

$\sum_j^k p_j = \frac{exp(\beta_{0, k} + X\beta)}{1 + exp(\beta_{0, k} + X\beta)}$, such that $p_1 \le p_2 \le ... p_k$. Notice that this model is a cumulative sum of probabilities which involves just changing the intercept of the linear regression portion.

```r
library(MASS)
```

### 4. Poisson Regression

**Assumption** 

$g(\mu)=\beta_0+\beta_1 x_1+\beta_2 x_2+\ldots+\beta_k x_k=x_{i}^{T}\beta$. 

Response $Y$ has a Poisson distribution that is $y_i \sim Poisson(\mu_i)$ for $i=1,...,N$. $g(\mu)$ is the link function and $E(Y_i) = \mu$. Link functions are usually takes identity link or natural log link $\log \mu = X\beta$. 

**Overdispersion**. 
A poisson distribution has the same mean and variance. Overdispersion means that the observed variance is larger than the assumed variance. Two typical solutions: 

1) Adjust for overdispersion, estimate $\varphi$, where $\varphi$ is a scale parameter adjust the standard errors and test statistics. 
2) Use negative binomial regression instead. 

```r
df5 <- fread('https://onlinecourses.science.psu.edu/stat501/sites/onlinecourses.science.psu.edu.stat501/files/data/poisson_simulated.txt')

df5 <- data.frame(df5)
df5 <- df5[c('x', 'y')]
mod5_full <- glm(y ~ x, df5, family = poisson(link = 'log'))
df5['pred'] <- predict(mod5_full, type = 'response')

ggplot(df5) + 
  geom_point(aes(x, y), color = 'red') + 
  geom_point(aes(x, pred), color = 'blue') + 
  geom_line(aes(x, pred))
```

### 5. Wald, Score, LR test

**Score test**: 

$U(\theta) = \frac{\partial \log L(\theta)}{\partial \theta}$
Under null hypothesis, $E[U(\theta)] = 0$, $Var(U(\theta))=I_n(\theta)$, the score test is on $S(\theta) = \frac{U^2(\theta)}{I_n(\theta)} \sim^{appro}\chi^2_1$
$I_n(\theta) = -E[\frac{\partial^2 l(\theta) }{\partial \theta^2}]$

```
xbar=mean(survtime)
n=length(survtime)
lambda0=1/60
score=n*(1-lambda0*xbar)^2
p.value=1-pchisq(score,df=1)
```

**Wald test**
$W(\theta) = I_n(\hat\theta)(\theta_0 - \hat\theta)^2$, where $\hat\theta$ is MLE of $\theta$. 
When the null hypothesis is true, $W \sim \chi_1^2$
It does not calculate the standard error assuming the null hypothesis is true.
Wald like confidence interval $\hat\theta \pm z_{1 - \alpha/2} \sqrt{I^{-1}(\hat\theta)}$
```r
library(aod)
mydata <- read.csv('http://www.ats.ucla.edu/stat/data/binary.csv')
fit <- glm(admit ~ ., data = mydata, family = binomial(link = 'logit'))
summary(fit)
# wald test
wald.test(b = coef(fit), Sigma = vcov(fit), Terms = 4:6)
```
> [http://isites.harvard.edu/fs/docs/icb.topic1383356.files/Lecture%2016%20-%20Score%20and%20Wald%20Tests%20-%201%20per%20page.pdf](http://isites.harvard.edu/fs/docs/icb.topic1383356.files/Lecture%2016%20-%20Score%20and%20Wald%20Tests%20-%201%20per%20page.pdf)
> [http://www.ats.ucla.edu/stat/mult_pkg/faq/general/nested_tests.htm](http://www.ats.ucla.edu/stat/mult_pkg/faq/general/nested_tests.htm)

## Part 5. Time Series


### 1. AR model

**AR(1)**: 
$x_t = \delta + \phi_1 x_{t-1} + \epsilon_t$, where $|\phi_1| < 1$ and $\epsilon \sim N(0, \sigma^2)$
mean: $E[x_t]=\frac{\delta}{1-\phi_1}$
variance: $Var(x_t) = \frac{\sigma^2}{1 - \phi_1^2}$
correlation: $\rho(x_t, x_{t-h}) = \phi_1^h$
![AR_acf](https://onlinecourses.science.psu.edu/stat510/sites/onlinecourses.science.psu.edu.stat510/files/L01/graph_08.gif)

### 2. MA model 

**MA(1)**
$x_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1}$, where $\epsilon_t \sim N(0, \sigma^2)$
mean: $E[x_t] = \mu$
variance: $Var(x_t) = (1 + \theta_1^2)\sigma^2$
correlation: $\rho(x_t, x_{t-1}) = \frac{\theta_1}{1 + \theta_1^2}$
![MA_acf](https://onlinecourses.science.psu.edu/stat510/sites/onlinecourses.science.psu.edu.stat510/files/L02/graph_19.gif)
![ma_pacf](https://onlinecourses.science.psu.edu/stat510/sites/onlinecourses.science.psu.edu.stat510/files/L02/graph_27.gif)
    
### 3. ACF and PACF and diagnostic for ARMA

The ACF and PACF should be considered together. It can sometimes be tricky going, but a few combined patterns do stand out.

1. AR models have theoretical PACFs with non-zero values at the AR terms in the model and zero values elsewhere. The ACF will taper to zero in some fashion. 
2. MA models have theoretical ACFs with non-zero values at the MA terms in the model and zero values elsewhere. (Example)
3. ARMA models (including both AR and MA terms) have ACFs and PACFs that both tail off to 0. These are the trickiest because the order will not be particularly obvious.  Basically you just have to guess that one or two terms of each type may be needed and then see what happens when you estimate the model.
4. If the ACF and PACF do not tail off, but instead have values that stay close to 1 over many lags, the series is non-stationary and differencing will be needed.  Try a first difference and then look at the ACF and PACF of the differenced data.
5. If all autocorrelations are non-significant, then the series is random (white noise; the ordering matters, but the data are independent and identically distributed.)  You’re done at that point.
6. If you have taken first differences and all autocorrelations are non-significant, then the series is called a random walk and you are done.

### 4.  ARIMA wiht seasonal components

The seasonal ARIMA model incorporates both non-seasonal and seasonal factors in a multiplicative model.  One shorthand notation for the model is

$ARIMA(p, d, q) × (P, D, Q)S$ 

with p = non-seasonal AR order, d = non-seasonal differencing, q = non-seasonal MA order, P = seasonal AR order, D = seasonal differencing, Q = seasonal MA order, and S = time span of repeating seasonal pattern.

Without differencing operations, the model could be written more formally as $$\Phi(B^S) \phi(B) (x_t - \mu) = \Theta(B^S) \theta(B) w_t$$, the components are 
    \begin{align*}
    \Phi(B^S) & = 1 - \Phi_1 B^S - \dots - \Phi_P B^{PS} \\
    \phi(B) & = 1 - \phi_1 B - \dots - \phi_p B^{p} \\
    \Theta(B^S) & = 1 + \Theta_1 B^S + \dots + \Theta_QB^{QS} \\
    \theta(B) & = 1 + \theta_1 B + \dots + \theta_q B^{q}
    \end{align*}

Note that on the left side of equation (1) the seasonal and non-seasonal AR components multiply each other, and on the right side of equation (1) the seasonal and non-seasonal MA components multiply each other.

### 5. ARIMA prediction 

The ARIMA forecasting equation for a stationary time series is a linear (i.e., regression-type) equation in which the predictors consist of lags of the dependent variable and/or lags of the forecast errors.  That is:
    **Predicted value of Y = a constant and/or a weighted sum of one or more recent values of Y and/or a weighted sum of one or more recent values of the errors.**

### 6. GARCH(1, 1) model
    
An ARCH (autoregressive conditionally heteroscedastic) model is a model for the variance of a time series.  ARCH models are used to describe a changing, possibly volatile variance.

Next period forecast of variance is a blend of our last period forecast and last period’s squared return. 
[http://cims.nyu.edu/~almgren/timeseries/Vol_Forecast1.pdf](http://cims.nyu.edu/~almgren/timeseries/Vol_Forecast1.pdf)

### 7. Why Stationary is important?

Stationarity is a one type of dependence structure.

Suppose we have a data $X_1,...,X_n$. The most basic assumption is that $X_i$ are independent, i.e. we have a sample. The independence is a nice property, since using it we can derive a lot of useful results. The problem is that sometimes (or frequently, depending on the view) this property does not hold.

Now independence is a unique property, two random variables can be independent only in one way, but they can be dependent in various ways. So stationarity is one way of modeling the dependence structure. It turns out that a lot of nice results which holds for independent random variables (law of large numbers, central limit theorem to name a few) hold for stationary random variables (we should strictly say sequences). And of course it turns out that a lot of data can be considered stationary, so the concept of stationarity is very important in modeling non-independent data.

When we have determined that we have stationarity, naturally we want to model it. This is where ARMA models come in. It turns out that any stationary data can be approximated with stationary ARMA model, thanks to Wold decomposition theorem. So that is why ARMA models are very popular and that is why we need to make sure that the series is stationary to use these models.

Now again the same story holds as with independence and dependence. Stationarity is defined uniquely, i.e. data is either stationary or not, so there is only way for data to be stationary, but lots of ways for it to be non-stationary. Again it turns out that a lot of data becomes stationary after certain transformation. ARIMA model is one model for non-stationarity. It assumes that the data becomes stationary after differencing.

In the regression context the stationarity is important since the same results which apply for independent data holds if the data is stationary.

[http://stats.stackexchange.com/questions/19715/why-does-a-time-series-have-to-be-stationary](http://stats.stackexchange.com/questions/19715/why-does-a-time-series-have-to-be-stationary)

## Part 6. Sampling, Resampling, Simulation, and More

### 1. Basic Idea of Sampling and Resampling

The desired properties of sampling: 

1. Unbiased
2. Low MSE (mean square error)
3. Robust, do not fluctuate too much with respect to extreme values

Example: 
> Sample mean: $\bar X = \frac{\sum X_i}{n}$
> Sample standard deviation: $s^2 = \frac{(X_i - \bar X)^2}{n - 1}$
> Under finite sample without replacement $Var(\bar X) = \frac{N - n}{N}\frac{\sigma^2}{n}$. The first part disappears when sampling with replacement or sample size is large enough to be view as infinity. 

> [http://www.et.bs.ehu.es/~etptupaf/nuevo/ficheros/stat4econ/muestreo.pdf](http://www.et.bs.ehu.es/~etptupaf/nuevo/ficheros/stat4econ/muestreo.pdf)

**Resampling**

Classical statistical parametric tests compare observed statistics to theoretical sampling distributions. Resampling a data-driven, not theory-driven methodology which is based upon repeated sampling within the same sample.

Resampling refers to methods for doing one of these

- Estimating the precision of sample statistics (medians, variances, percentiles) by using subsets of available data (jackknifing) or drawing randomly with replacement from a set of data points (bootstrapping)
- Exchanging labels on data points when performing significance tests (permutation tests, also called exact tests, randomization tests, or re-randomization tests)
- Validating models by using random subsets (bootstrapping, cross validation)


### 2. Simple random sampling vs Stratified random sampling

1. **Simple Random Sampling**: A simple random sample (SRS) of size n is produced by a scheme which ensures that each subgroup of the population of size n has an equal probability of being chosen as the sample.
2. **Stratified Random Sampling**: Divide the population into "strata". There can be any number of these. Then choose a simple random sample from each stratum. Combine those into the overall sample. That is a stratified random sample. (Example: Church A has 600 women and 400 women as members. One way to get a stratified random sample of size 30 is to take a SRS of 18 women from the 600 women and another SRS of 12 men from the 400 men.)

SRS vs Strat

1. Stratification may produce a smaller error of estimation than would be produced by a simple random sample of the same size. This result is particularly true if measurements within strata are very homogeneous.
2. The cost per observation in the survey may be reduced by stratification of the population elements into convenient groupings.
3. Estimates of population parameters may be desired for subgroups of the population. These subgroups should then be identified

Naive sample size for each strata: $n_i = \frac{N_i}{N}n$; min-variance for each strata: $n_i \propto N_i \sigma_i$ (finite population results)

### 3. Hansen-Hurwitz estimator

> Rational: For example, we want to estimate the number of job openings in a city by sampling firms in that city. Many of the firms in the city are small firms. If one uses s.r.s, size of a firm is not taken into consideration and a typical sample will consist of mostly small firms. However, the number of job openings is heavily influenced by large firms. Thus, we should be able to improve the estimate of number of job openings by **giving the large firms a greater chance to appear in the sample**, for example, with probability proportional to size or proportional to some other relevant aspects.

Let $p_i$, $i = 1, ... , N$ denote the probability that a given population unit will be selected. Then, $\hat{\tau}_p=\dfrac{1}{n} \sum\limits^n_{i=1} \dfrac {y_i}{p_i}$ is the Hansen-Hurwitz estimator of population total. 

> http://wiki.awf.forst.uni-goettingen.de/wiki/index.php/Hansen-Hurwitz_estimator_examples

### 4. Random sampling from large populations

**Reservoir sampling.** 
1. Keep the first element
2. When the $i$th element arrives, replace the new element with $1/i$ probability, reject the new one with $(i - 1) / i$ probability.
> [https://en.wikipedia.org/wiki/Reservoir_sampling](https://en.wikipedia.org/wiki/Reservoir_sampling)

### 5. Auxillary Data and Ratio Estimator

Suppose we have population of size $N$, we are interested in statistic $\mu_y = \mathbb E[Y]$, whereas $Y_i$ is hard to measure. If we have another statistic $\mu_x = \mathbb E[X]$, and $X_i$ is easy to reach and highly linear correlated with $Y_i$. Then we can use $\hat \tau_y = \frac{\bar Y}{\bar X} \tau_x$ as ratio estimator of $\tau_y$. It is useful when 

A. When X and Y are highly linearly correlated through the origin, then: $Var(\hat \tau_r) < Var(N\bar y)$
B. The case where N is unknown, then it provides a way to estimate $\tau$ since when N is unknown.

> [https://onlinecourses.science.psu.edu/stat506/node/20](https://onlinecourses.science.psu.edu/stat506/node/20)

### 6. Bootstrap

Used as a non-parametric inference method, i.e., for median, percentile, correlation.

> Bootstrap for estimating point estimators, confidence intervals. Not for constructing test statistics (permutation test) or predicting. 

*Example 1: Estimation of Correlation*

1. Gather a bootstrap sample of size n.
2. Calculate the sample correlation from the bootstrap sample.
3. Repeat steps (1)-(2) $K$ times. ($K$ = 1000 is a good number)
4. To find the $(1-\alpha)100\%$ CI for $\rho$, find the $\alpha/2$ and $1-\alpha/2$ percentiles as the lower and upper bounds.

R code
```
## confidence interval for mean as non-parametric inference
library(ggplot2)
myData <- rexp(100, 3)
myAvg <- mean(myData)
num_replicate <- 10000
avg_bs_lst <- rep(0, num_replicate)
for (i in 1:num_replicate) {
    myData_bs <- myData[sample(1:length(myData), length(myData), replace=TRUE)]
    avg_bs_lst[i] <- mean(myData_bs)
}
qplot(avg_bs_lst)
ci <- quantile(avg_bs_lst, c(0.025, 0.975))
```

*Example 2: Exponential Distribution $\lambda$*

```
## confidence interval for lambda in Exponential distribution 
myData <- rexp(100, 3)
empirical.lambda <- 1 / mean(myData)
ci <- c(empirical.lambda - qnorm(0.975) * empirical.lambda / sqrt(1000), 
    empirical.lambda + qnorm(0.975) * empirical.lambda / sqrt(1000))

boot.sampling.dist <- matrix(1, 2000)
for (i in 1:2000) {
    boot.sampling.dist[i] <- 1 / mean(rexp(100, empirical.lambda))
}
ci.bs <- quantile(boot.sampling.dist, c(0.025, 0.975))
```

**Bootstrap Estimate of Correlation**

- Gather a bootstrap sample of size n.
- Calculate the sample correlation, $\rho$ , from the bootstrap sample.
- Repeat steps (1)-(2) B times. Typically want B to be larger than 1000.
- Find the $(1-\alpha)100\%$ CI for $\rho$.

> Bootstrap sample has significant overlap with the original data. About two-thrids of the original data points appear in each bootstrap sample. Bootstrap cannot be used to estimate prediction error.

### 7. Cross Validation 

- CV is to estimate the test error by holding out a subset of the training observations from the fitting process, and then applying the statistical learning method to those held out observations.

- In CV, we randomly divide the available set of samples into $K$ roughly equal-sized parts. Compute $CV_{(K)} = \sum\frac{n_k}{n}MSE_k$, where $MSE_k = \sum (y_i - \hat y_i)^2/n_k$, and $\hat y_i$ is the fit for observation $i$, obtained from the data with part $k$ removed. 

- Leave-one out cross-validation (LOOCV). In LOOCV, the estimates from each fold are highly correlated and hence their average can have high variance. [http://robjhyndman.com/hyndsight/loocv-linear-models/](http://robjhyndman.com/hyndsight/loocv-linear-models/)

- Since each training set is only $\frac{K-1}{K}$ as big as the original training set, the estimates of prediction error will typically be biased upward.

- A better choice is $K = 5\  or \ 10$. Good compromise for the bias-variance tradeoff. 

- To correctly conduct CV, the sample dividing process should be done before viewing the labels. 

[https://lagunita.stanford.edu/c4x/HumanitiesScience/StatLearning/asset/cv_boot.pdf](https://lagunita.stanford.edu/c4x/HumanitiesScience/StatLearning/asset/cv_boot.pdf)

### 8. Simulate a bivariate normal

> $Z \sim N(0, 1)$
 $X = Z_1$ and $Y = \sqrt{1 - \rho^2}Z_2 + \rho Z_1$

### 9. Generate random variable within a circle. 

> $U_1, U_2 \sim Unif(0, 1)$
> $(X_1, X_2) = (\sqrt{U_1}cos(2\pi U_2), \sqrt{U_1}sin(2\pi U_2))$
```
n <- 1e4
rho <- sqrt(runif(n))
theta <- runif(n, 0, 2*pi)
x <- rho * cos(theta)
y <- rho * sin(theta)
plot(x, y, pch=19, cex=0.6, col="#00000020")
```

### 10. Bagging vs Boosting

**Bagging**: bootstrap aggregating, is a general purpose procedure for reducing the variance of a statistical learning mehtod. As in bagging, we build a number of decision trees on bootstrapped training samples, and then combine all of the trees in order to create a single predictive model. Each tree is built on a bootstrap dataset, independent of the other trees. 

- parallel ensemble: each model is built independently 
- aim to decrease variance, not bias
suitable for high variance low bias models (complex models)
- an example of a tree based method is random forest, which develop fully grown trees (note that RF modifies the grown procedure to reduce the correlation between trees)

> When building random forest, each time a split in a tree is considered, a random selection of m predictors is chosen as split candidates from the full set of p predictors. 
    
**Boosting**: each tree is grown using information from previous grown trees. Instead of fitting a single large decision tree, boosting approach learns slowly. Given the current model, we fit a decision tree to the residuals from the model. We then add this new decision tree into the fitted function in order to update the residuals. 

- sequential ensemble: try to add new models that do well where previous models lack
- aim to decrease bias, not variance
- suitable for low variance high bias models
- an example of a tree based method is gradient boosting
    
[http://stats.stackexchange.com/questions/18891/bagging-boosting-and-stacking-in-machine-learning](http://stats.stackexchange.com/questions/18891/bagging-boosting-and-stacking-in-machine-learning)

![Bagging][3]

## Part 7. Misc

1. Conjugate family 

    > Baysian, prioer distribution and posterior distribution are of the same family of distributions. For example, for coin flipping experiments, beta prior guarantees that the posterior distribution is also beta.

    > $p(\theta|x) \propto p(x) \cdot p(\theta) \propto \theta^x (1-\theta)^{n - x} \theta^{\alpha-1}(1-\theta)^{\beta-1}\sim B(\alpha + x, \beta + n - x)$

2. Truncated normal. Data point is given from a normal distribution with given variance, how to estimate the mean. If the data was given from a truncated normal distribution (record the data when it is positive), how to compute the posterior distribution.

    > Likelihood function
    $$f(x|\mu) \propto \frac{1}{\sqrt{2\pi\sigma^2}}\exp\{-\frac{(x-\mu)^2}{2\sigma^2}\}1\{x > 0\}$$
    > Prior on $\mu \sim N(\mu_0, \sigma^2_0)$
    $$f(\mu) =  \frac{1}{\sqrt{2\pi\sigma^2_0}}\exp\{-\frac{(\mu-\mu_0)^2}{2\sigma^2_0}\}$$
    > Posterior on $\mu$
    $$f(\mu|x) \propto  \exp\{-\frac{(x-\mu)^2}{2\sigma^2}-\frac{(\mu-\mu_0)^2}{2\sigma^2_0}\}$$
    > [http://stats.stackexchange.com/questions/48897/maximum-likelihood-estimators-for-a-truncated-distribution](http://stats.stackexchange.com/questions/48897/maximum-likelihood-estimators-for-a-truncated-distribution)
    > [http://math.stackexchange.com/questions/573694/bayesian-posterior-with-truncated-normal-prior]


3. Unbiased and Minimum variance estimation. Three observations, $X_1$ and $X_2$ are normal with variance at 1; $X_3$ normal with variance at 2. 

    > $Y = a X_1 + a X_2 + (1 - 2a) X_3$
    > $Var(Y) = 2a^2 + 2(1 - 2a)^2$
    > Take derivative $a = \frac{2}{5}$
    > If $Var(X_3) = 0.05$, then $a = 0.04545...$

4. The probability of raining is $p$, the cost is $k$ when raining and not bringing an umbrella, is $2$ if not raining but with an umbrella, is $1$ if taking an unbrella and raining. How to make the decision. 

    > Let $q$ be probability to take an umbrella
    > $C = kp(1-q) + 2(1-p)q + pq$
    > If $kp > 2(1 - p) + p$, $p > \frac{2}{k + 1}$, then take umbrella

5. For example, when you google 'Flower' a commercial query. Usually on top or right of the web, some ad links are highlighted. Now, apply a new search algorithm that leads color change of these ad links. How do we know whether the algorithm change is reasonable?

    > Design an A B test. 
    > 1. Choose a metric, say, click rate per query; and define a practical significant margin, that is how significant the difference that is worth concerns and implementation
    > 2. Set up the experiment using the old version as control group, the new version as experimental group. Two treatments differ only in the proposed feature, i.e., whether the links are highlighted.
    > 3. Randomly assign the two versions to users and collect their responses. 
    > 4. After collecting enough data, run a two sample t-test to check if there is a significant difference between the two treatments. 
    
    > Related questions: when you type a letter, say A, in google search, some suggestions will appear below, such as America, Apple, Amazon….there are two models for those suggestions, how do you know which model is better. He refers to A/B test.

6. A guy randomly picks an integer K from 1 - 100, then he tossed a fair coin for K times, he wins if he get exactly one head, what's the prob. that he wins

    > $P = \frac{1}{100}\sum_{k = 1 .. 100} k (0.5)^k \approx 0.04$

7. A coin being tossed 100 times with 80 heads and 20 tails. What is the probability of head? A coin being tossed 10 times with 8 heads and 2 tails. What is the probability of head?

    > For the first one, we may assume the probability is 0.8. For the second one, we may still believe the probability is 0.5. 
    > Or we can use a uniform or beta distribution as prior. 
    > Or both probability are 0.8, but with different confidence interval. 

8. 10,000 data point, unkown distribution. want to find the center of the data. a). discuss about the center, give senarios when mean or median is better. b). find the median from the data.

    > 1. For symmetric distribution, mean and median are the same. Thus, sample mean is also a good estimator of sample median. For skewed distribution, say, positive skewness, median is smaller than mean. For example, national income is a skewed distribution, and median, is better than mean as the statistics describing the average incomes. 
    > 2. Bootstrap to find the confidence interval

9. How to examine if a die is fair. followup: how many times you need to throw the die. Roughly how many times do I need to roll a 6-sided die to feel confident that it's giving "fair" results?

    > Chi-square test 
    > $\sum \frac{(observed - expected)^2}{expected} \sim \chi_5$, size depends on significant level, power, and effect size. 
    > A typical requirement for Chi-square test to work is $n$ is large enough to have $E_i \ge 5$ for all $i$. Thus, 30 rolls will be a good start. (also, no empty cells)
    > [http://math.stackexchange.com/questions/57624/how-many-rolls-do-i-need-to-determine-if-my-dice-are-fair](http://math.stackexchange.com/questions/57624/how-many-rolls-do-i-need-to-determine-if-my-dice-are-fair)
    
10. Online poll: choose your favourite color (4 choose 1, e.g. red, green, blue...). Assume there is no position bias for answers. Among m responses, x1 are red. 1. Construct a confidence interval for the proportion of red. 2. If we know that some people are picking answer randomly (say, the proportion of people who choose answer randomly is known to be r). Can you adjust your estimate for the proportion of people who like red? 3. How the adjusted estimate's variance change? 4. What if we do not know r? How can we estimate it?

    > 1. $\hat p_r = \frac{x_r}{n}$, and the confidence interval $\hat p_r \pm z_{1-\alpha/2}\sqrt{\frac{\hat p_r(1 - \hat p_r)}{n}}$
    > 2. $\hat p_r = \frac{x_r - 0.25 rn}{(1-r)n}$, and the confidence interval $\hat p_r \pm z_{1 - \alpha/2}\sqrt{\frac{\hat p_r (1 - \hat p_r)}{(1-r)n}}$
    > 3. If $r$ is unknow, we can set redundent questions in the survey to identify the proportion of subjects that answer randomly. Repeat this question on the same subjects, if say $k$ subjects change their answer, then roughly, $\frac{4}{3}k$ subjects are randomly answering their questions. Thus, we can estimate the random responding rate $r = \frac{4k}{3n}$

11. Assume the probabilities of number of childrens in a family from 0 to 4 are $p_0, p_1, p_2, p_3, p_4$. What is the probability that a randomly sampled girl has at least one sister?

    > $A$: the child is a girl
    > $B$: the girl has a sister
    $P(B|A) = \frac{P(A|B)P(B)}{P(A)} = \frac{0.25p_2 + 0.375p_3 + 0.43875p_4}{0.5}$

12. Randomized response

    > Randomized response is a research method used in survey interview. Chance decides, unknown to the interviewer, whether the question is to be answered truthfully, or "yes", regardless of the truth.
    > For example, Ask a man whether he had sex with a prostitute this month. Before he answers ask him to flip a coin. Instruct him to answer "yes" if the coin comes up tails, and truthfully, if it comes up heads. Only he knows whether his answer reflects the toss of the coin or his true experience. It is very important to assume that people who get heads will answer truthfully, otherwise the surveyor are not able to speculate.
    
    > [https://en.wikipedia.org/wiki/Randomized_response](https://en.wikipedia.org/wiki/Randomized_response)
    > [http://www.cl.cam.ac.uk/~dq209/publications/spotme.pdf](http://www.cl.cam.ac.uk/~dq209/publications/spotme.pdf)

13. 1000 hard drives, test for 6 months and 4 are broken. What can be infered? What if your engineer friend told you that "failure rate" is $50\%$? How you react?

    > 1) Under Bayesian, assume a prior on $p$, say, Beta distribution, then the posterier distribution is also of Beta distribution. 
    > 2) Proportion test $\frac{p - 0.5}{\sqrt{p(1-p)/1000}}$

14. Observational Studies, Selection bias
    
    There are mainly two approaches for statistical experimentals. Randomized experiments and observational studies. Randomized experiments are perfect for causal inference. However, when ideas cannot be tested in randomized experiments, for example, treatment might be self-selected, or an experiment might incur damage to user experience. Then observational approachs to causal inferences should be in use. 
    Observational techniques allow us to estimate counterfactuals in the absence of an experiment, provided that we can convincingly control for other factors that might explain an observed effect. Propensity scores, instrumental variables, and synthetic controls are all methods for estimating counterfactuals when randomized treatment is not an option. 

15. You have an 50-50 mixture of two normal distributions with the same standard deviation. How far apart do the means need to be in order for this distribution to be bimodal? Why?

    $f(x) = \frac{1}{2}\phi(x, \mu_1, \sigma^2) + \frac{1}{2}\phi(x, \mu_2, \sigma^2)$
    Take derivative on $x$, we have at least one root at $x_0 = \frac{\mu_1 + \mu_2}{2}$, then take the second derivative, and at $x_0$, the second derivative is strictly less than 0 for $|\mu_1 -\mu_2|< 2\sigma$.
    > [https://www.quora.com/You-have-an-50-50-mixture-of-two-normal-distributions-with-the-same-standard-deviation-How-far-apart-do-the-means-need-to-be-in-order-for-this-distribution-to-be-bimodal-Why](https://www.quora.com/You-have-an-50-50-mixture-of-two-normal-distributions-with-the-same-standard-deviation-How-far-apart-do-the-means-need-to-be-in-order-for-this-distribution-to-be-bimodal-Why)

16. How to answer the case study questions?

    1. Q: question
    2. A: assumptions
    3. H: hypothesis.
    4. M: metrics.
    5. D: Design. Randomized or observational; A/B or longitutinal
    6. I: Implement. Collect qualitative data about user engagement to allow statistical analysis. 
    7. A: Analyze results 

---
    
Selection bias occurs when two populations are not homogenity and thus may cause the simple mean comparsion invalid. For example, to test the popularity of Google among users by throwing survey on chrome users is invalid since more other than not, chrome users may be prone to google given their selection on google's product. Thus, it is necessary to reduce the potential bias for the causal inference. 
    The formal approach is propensity score. The idea is to match users by their background, such as, age, gender, and so forth, to reduce potential bias. 
Randomization is at the core of experimentation because it balances out these confounding variables.
Selection bias, in general, is a problematic situation in which error is introduced due to a non-random population sample. For example, if a given sample of 100 test cases was made up of a 60/20/15/5 split of 4 classes which actually occurred in relatively equal numbers in the population, then a given model may make the false assumption that probability could be the determining predictive factor. Avoiding non-random samples is the best way to deal with bias; however, when this is impractical, techniques such as resampling, boosting, and weighting are strategies which can be introduced to help deal with the situation. 

## Part 6. Theory

1. Chebyshev's Inequality

    > $P(|X - \mathbb E (X)| \ge a) \le \frac{Var(X)}{a^2}$
    > $P(X \ge a) \le \frac{\mathbb E[f(x)]}{f(a)}$

2. Weak Law of Large Number 

    > If $X_1, ...$ are independent random variables such that $\mathbb E[X_n] = \mu$ and $Var[X_n] \le \sigma^2$ for each $n$, then $\frac{X_1 + X_2 + \cdots X_n}{n}\rightarrow \mu$ in probability. 
3. Borel Cantelli Lemma

    > Suppose $A_1, ...$ is a sequence of events. 
    (1) If $\sum P(A_n) < \infty$, then $P(A_n \ i.o.) = P(\lim \sup A_n) = 0$
    (2) If $\sum P(A_n) = \infty$, and $A_i$ are independent, then $P(A_n \ i.o.) = P(\lim \sup A_n) = 1$
    
    > $\lim \sum A_n = \cap_{n=1}^{\infty} \cup_{m=n}^{\infty} A_m$
    
4. Strong Law of Large Number

    > Let $X_1,...$ be independent random variables each with mean $\mu$. Suppose there exists an $M < \infty$ such that $\mathbb E[X_n^4]\le M$ for each $n$. Then $w.p.1$, 
    $$\frac{X_1 + X_2 + \cdots X_n}{n} \rightarrow \mu$$
    > Counter example:
    > $P(X_n = 2^n) = 2^{-n}$, then $\mathbb E[X_n] = 1$ but $\frac{X_1 + X_2 + ... X_n}{n} \rightarrow 0$
    
5. Central Limit Theorem 

    > Let $X_1, ...$ be independent, identically distributed random variables with mean $\mu$ and finite variance. Then 
    $$\frac{X_1 + X_2 + ... X_n}{\sqrt{n}} \rightarrow N(\mu, \sigma^2)$$
    > Proof scratch. 
    > Let $Y_i =\frac{X_i - \mu}{\sigma}$ with characteristic function $\phi(Y)$. Then $\phi(0) = 1$, $\phi'(0) = 0$ and $\phi''(0) = -1$. 
    > Then $\phi_{\bar{Y}}(t) = (\phi(\frac{t}{\sqrt{n}}))^n = \phi(0) + \frac{t}{\sqrt{n}}\phi'(0) + \frac{t^2}{2n}\phi''(0) + \epsilon \frac{t^2}{n}$
    > $$\lim_{n \rightarrow \infty} \phi_{\bar{Y}}(t) = e^{-\frac{t^2}{2}}$$

---


  [1]: https://onlinecourses.science.psu.edu/stat502/sites/onlinecourses.science.psu.edu.stat502/files/lesson07/bench_diagram_03.png
  [2]: https://onlinecourses.science.psu.edu/stat502/sites/onlinecourses.science.psu.edu.stat502/files/lesson07/bench_diagram_05.png
  [3]: https://i.stack.imgur.com/RFfqb.png