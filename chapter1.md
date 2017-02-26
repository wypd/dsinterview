# Chapter 1. Math and Optimization

## Part 1. Basic Calc

1. The number $$2^{29}$$ has 9 digits, all different. Find the missing digit without computing $$2^{29}$$?
   
    > $$\sum x_i \text{mod} 9 = \sum 10^i x_i \text{mod} 9$$
   
2. What is value of $$i^i$$?

    > $$e^{i\theta} = \cos(\theta) + i \sin(\theta) \Rightarrow e^{i\pi / 2}= i \Rightarrow i^i = e^{-\pi/2}$$
    
    > [http://math.stackexchange.com/questions/191572/prove-that-ii-is-a-real-number](http://math.stackexchange.com/questions/191572/prove-that-ii-is-a-real-number)

3. Prove that the area of a triangle is given by $$A = \sqrt{s(s-a)(s-b)(s-c)}$$ where a, b, and c are the side lengths, and $$s=\frac{a+b+c}{2}$$ is half the perimter.

    > Denote $$\theta$$ as $$\cos(\theta) = \frac{a^2 + b^2 - c^2}{2ab}$$.
    > 
    $$\begin{align*}
    A & = \frac{1}{2}a b \sin(\theta) \\
    & = \frac{1}{2} \sqrt{a^2 b^2 (1 - (\frac{a^2 + b^2 - c^2}{2ab})^2)} \\
    & = \frac{1}{4}\sqrt{4a^2 b^2 - (a^2 + b^2 - c^2)^2}
	\end{align*}$$
    > [Heron's Formula](https://en.wikipedia.org/wiki/Heron%27s_formula)

4. Does the infinite sum $$\sum_{n=1}^{\infty}e^{-\sqrt{n}}$$ converge?

    > $$\int_{x=1}^{\infty}e^{-\sqrt{x}}dx = 2\int_{t = 1}^{\infty}te^{-t}dt = 4e^{-1}$$
    
    > [Ratio comparison](http://math.stackexchange.com/questions/1174116/showing-sum-infty-n-1-frac-1-e-sqrt-n-converges)

5. What are $$\sum_{k = 1}^n k$$, $$\sum_{k = 1}^n k^2$$, $$\sum_{k = 1}^n k^3$$.

    > $$\sum_{k = 1}^n k^4 = \sum_{k = 0}^{n-1} (k + 1)^4 \Rightarrow \sum_{k = 1}^n k^3 = \frac14n^2(n+1)^2$$    
    > [http://math.stackexchange.com/questions/320985/how-to-determine-equation-for-sum-k-1n-k3](http://math.stackexchange.com/questions/320985/how-to-determine-equation-for-sum-k-1n-k3)

6. What is the integral of $$\sec(x)$$ from $$x=0$$ to $$x=\frac{\pi}{6}$$?

    > $$d\sec(x) = \sec(x) \tan(x) dx$$
    > $$d\tan(x) = \sec(x) dx$$

7. Solution to ODE

    > 1. If $$r_1$$ and $$r_2$$ are real and $$r_1 \neq r_2$$, then the general solution is $$y=c_1 e^{r_1 x} + c_2 e^{r_2 x}$$;
    > 2. If $$r_1$$ and $$r_2$$ are real and $$r_1 = r_2 = r$$, then the general solution is $$y=c_1 e^{rx} + c_2 x e^{rx}$$;
    > 3. If $$r_1$$ and $$r_2$$ are complex number $$\alpha + i\beta$$, then the general solution is $$y=e^{\alpha x}(c_1 \cos \beta x + c_2 \sin \beta x)$$.

8. What is the $$100$$th digit to the right of the decimal point in the decimal representation of $$(1+\sqrt{2})^{3000}$$?

9. Fibonacci formula

    > $$F_n = \frac{(\frac{1+\sqrt{5}}{2})^n - (\frac{1-\sqrt{5}}{2})^n}{\sqrt{5}}$$

10. Given $$\frac{1}{a} + \frac{1}{b} = \frac{1}{c}$$ and $$gcd(a, b, c) = 1$$, is $$a + b$$ a perfect square?

    > Let $$gcd(a, b) = k$$, then $$c(a_1 + b_1) = k a_1 b_1$$, then we have
    $$(a_1 + b_1) | k$$ and $$k | (a_1 + b_1)$$. So $$a + b = k^2$$.
    > [http://quantquestions.tumblr.com/post/81580564393/quant-interview-question-perfect-square](http://quantquestions.tumblr.com/post/81580564393/quant-interview-question-perfect-square)

11. Unit root

    > Let $$\omega = e^{\frac{2\pi i}{n}}$$, which implies $$\omega^n = 1$$, then we have $$1 + \omega + \omega^2 + \omega^{n - 1} = 0$$
    
    > [vieta formula](https://en.wikipedia.org/wiki/Vieta%27s_formulas)

12. Stirling's approximation

    > $$n! \sim \sqrt{2\pi n}(\frac{n}{e})^n$$
    > [https://en.wikipedia.org/wiki/Stirling%27s_approximation](https://en.wikipedia.org/wiki/Stirling%27s_approximation)
    
13. Prove the following inequality: $$1/2 \cdot 3/4 \cdot 5/6 \cdots 99/100 < 1/10$$

14. A worm can crawl at a velocity c. He is attempting to cross a rubber sheet of initial length $$L$$. One end of the sheet is fixed, the other end is being pulled at a velocity s such that the sheet is being stretched. How long does it take the worm to cross the sheet?

    > $$x' = c + \frac{s\cdot x}{s\cdot t + L}$$

15. One morning, in Springfield, somewhere in the US, it started snowing at a heavy but constant rate. Homer Simpson had just started his own snowplow business. His snowplow started out at 8:00 A.M. At 9:00 A.M. it had gone 2 miles. By 10:00 A.M. it had gone 3 miles. Assuming that the snowplow removes a constant volume of snow per hour, determine the time at which it started snowing.

16. Integrate $$\sin(x)/x$$ from -infinity to +infinity

    > $$\int_0^{\infty} \frac{sin(x)}{x} dx = \int L(sin(x))(s) ds = \int \frac{1}{1+s^2}ds = \frac{\pi}{2}$$
    
    > $$ \int_0^\infty \left({\int_0^\infty e^{- x y} \sin x \, \mathrm d y}\right) \, \mathrm d x = \int_0^\infty \left({\int_0^\infty e^{- x y} \sin x \, \mathrm d x}\right) \, \mathrm d y$$
    
    > [https://en.wikipedia.org/wiki/Dirichlet_integral](https://en.wikipedia.org/wiki/Dirichlet_integral)

17. Solve ODE 1) $$y' = y + 1$$, 2) $$y' = \frac{y}{x} + 1$$

## Part 2. Linear Algebra

1. Assuming that all entries of an correlation matrix which are not on the main diagonal are equal to $$\rho$$, find upper bound and lower bound for $$\rho$$.

    > $$A = \rho M + (1 - \rho) I$$    
    > [http://stats.stackexchange.com/questions/72790/bound-for-the-correlation-of-three-random-variables](http://stats.stackexchange.com/questions/72790/bound-for-the-correlation-of-three-random-variables)

2. Let $$A = \begin{bmatrix} 2 & -2 \\ -2 & 5 \end{bmatrix}$$, find  1) $$M$$, such that $$M^2 = A$$; 2) M, such that $$M'M = A$$.

3. Let A and B be square matrices of the same size. Show that the traces of the matrices AB and BA are equal

    > $$tr(AB) = \sum_i (AB)_{ii} = \sum_i \sum_j a_{ij}b_{ji} = \sum_j \sum_i b_{ji}a_{ij} = \sum_j (BA)_{jj} = tr(BA)$$
    > [http://www2.math.ou.edu/~dmccullough/teaching/slides/maa2010.pdf](http://www2.math.ou.edu/~dmccullough/teaching/slides/maa2010.pdf)

4. QR decomposition

    > For each non-singular $$n\times n$$ matrix $$A$$, there is a unique pair of orthogonal matrix $$Q$$ and upper-triangular matrix $$R$$ with positive diagonal elements such that $$A=QR$$. QR decomposition is often used to solve linear system $$Ax=b$$ when A is a non-singular matrix. Since $Q$ is an othogonal matrix, $$Q^{-1} = Q'$$ and $$QRx = b \Rightarrow Rx=Q'b$$, Because $$R$$ is an upper-triangular matrix, we can begin with $$x_n$$, and recursively calculate all $$x_i$$.

5. LU decomposition (non-singular square)

    > Let $$A$$ be a nonsingular $$n\times n$$ matrix, **LU decomposition** expresses $$A$$ as the product of a lower and upper triangular matrix: $$A=LU$$. There exists a unique unit lower triangular $L$ with diagonal elements all equal to one and a unique upper triangular matirx $U$ such that $$A=LU$$. 
    > The LU decomposition is the Gaussian transform process that zero the $$a_{ij}$$ where $$j > i$$. 
    > LU decomposition can be used to solve $$Ax=b$$ and calcualte the determinant of $$A$$: $$LUx=b \Rightarrow Ux=y, Ly=b$$.
    
6. Cholesky decomposition (symmetric positive definite square)

    > When $$A$$ is a symmetric positive definite matrix, **Cholesky decomposition** expresses $$A$$ as $$A=R'R$$, where $$R$$ is a unique upper-triangular matrix with positive diagonal entries. Essentially, it is a $$LU$$ decomposition with property $$L=U'$$.
    **Cholesky decomposition is useful in Monte Carlo simulation to generate correlated random variables.**    
    > [http://math.stackexchange.com/questions/163470/generating-correlated-random-numbers-why-does-cholesky-decomposition-work](http://math.stackexchange.com/questions/163470/generating-correlated-random-numbers-why-does-cholesky-decomposition-work)

7. SVD decomposition.

    > For any $$n\times p$$ matrix $$X$$, there exists a factorization of the form $$X=UDV'$$, where $$U$$ and $$V$$ are $$n\times p$$ and $$p \times p$$ orthogonal matrices, with columns of $$U$$ spanning the column space of $$X$$, and the columns of $$V$$ spanning the row space; $$D$$ is a $$p\times p$$ diagonal matrix called the singular values of $$X$$. 
    > Properties:
    > 1. Denote $$\sigma_i$$ as the diagonal elements in singular matrix $$D$$, $$u_i$$ and $$v_i$$ are the corresponding singular vector, we have $$Xv_i = \sigma_i u_i$$ and $$X'u_i = \sigma_i v_i$$.
    > 2. For a positive definite (covariance) matrix, we have $$V=U$$ and $$\Sigma = UDU'$$. Furthermore, $$D$$ is the diagonal matrix of eigenvalues and $$U$$ is the matrix of $$n$$ corresponding eigenvectors.
    > 3. $$v_i$$ are orthonormalized eigenvectors of $$X'X$$ and $$u_i$$ are orthonormalized eigenvectors of $$XX'$$.
    > 4. $$\sigma_j = \sqrt{\lambda_j(X'X)}$$, where $$\lambda_j(X'X)$$ is the $$j$$th largest eigenvalue of $$X'X$$.

8. Special matrixs.

    > 1) **Nilpotent**: Square matrix $N$ such that $$N^k=0$$.
    > 2) **Idempotent matrix**: $$A^2=A$$
    > 3) **Hermitian matrix**: A complex matrix $$A$$ is a Hermitian matrix if it equals its own complex conjugate transpose, that is $$A = A^H$$.
    > $$\begin{bmatrix} 2 & 2 + i & 4 \\ 2 - i &  3 & i \\ 4 & -i & 1\end{bmatrix}$$
    > A complex matrix $$U$$ is unitary matrix if the inverse of $$U$$ equals the complex conjugate transpose of $$U$$, $$U^{-1} = U^H$$.

## Part 3. Numerical Methods

1. What is convergence rate of Newton method?

    > $$||x_{t+1} - x_t|| <= c ||x_t - x_{t-1}||^2$$
    > [](https://www.math.washington.edu/~burke/crs/408/lectures/L10-Rates-of-conv-Newton.pdf)

2. What is convergence rate of Monte Carlo methods?

    > $$\frac{1}{\sqrt{n}}$$

3. Generate IID Normal random variable.

    > 1. Box-Muller $$U_1, U_2 \sim Unif(0, 1)$$, then $$X_1 = \cos(2\pi U_1) \sqrt{-\log(U_2)}$$ and $$X_2 = \sin(2\pi U_1) \sqrt{-\log(U_2)}$$.
    > 2. Acceptance-Rejection method

4. Lagrangian method

    > Given general form
    > $$ \begin{align*}
    \min \ & f(x) \\
    \text{s.t.} \ & h(x) \le 0 \\
    & \ g(x) = 0
    \end{align*}$$
    > The Lagrangian function is $$L = f(x) + \mu h(x) + \lambda g(x)$$.
    > The KKT condition are
    $$ \begin{align*}
    \frac{\partial L}{\partial x} & = 0 \\
    h(x) & \le 0 \\
    \mu & \ge 0 \\
    \mu h(x) & = 0 \\
    g(x) & = 0
    \end{align*}$$

5. Newton method
    
    > **Newton's method**
    > Newton's method is to find the new iterate $$x_{k + 1}$$ as a function of $$x_k, x_{k + 1} = x_k + p_k$$, where $$p_k$$ given by $$p_k = -B_k^{-1}\nabla f_k$$, from minimizing Taylor expansion  $$m_k(p) = f_k + p'\nabla f_k + \frac{1}{2}p' B_k p$$.
    > **DFP**
    > Instead of computing the Hessian matrix directly, DFP approximate it based on the change in gradient between iterations. The idea is to approximate Hessian via several properties
    > 1) $$B_k$$ must be symmetric
    > 2) $$B_k$$ muct be formed such that the gradient of the model is equal to the function's gradient at the points $$x_k$$ and $$x_{k-1}$$. 
    > 3) Close to $$B_{k-1}$$
    > **BFGS**
    > BFGS is to approximate the inverse of Hessian directly using the same technique as in DFP. 
    > L-BFGS (limited-memory BFGS) stores the approximate Hessian in a compressed form that requires storing only a constant multiple of vectors of length $$n$$.
    
    
## Part 4. Stochastic Calculus

1. Martingale

    > 1. $X_t$ is adapted. Measurable for all t
    > 2. $X_t$ is integrable, $\mathbb E[|X_t|] < \infty$ for all t
    > 3. $\mathbb E[X_t|F_s] = X_s$ almost surely for all s < t
    
    > **Martingale examples**:
    > Brownian motion: $W(t)$, $W(t)^2 - t$, $\exp\{\lambda W(t) - \frac{1}{2} \lambda^2 t\}$
    > Random walk: $S_n$, $S_n^2 - n$

2. Ito's isometry

    > *Let $f_t$ and $g_t$ be progressively measurable and square integrable. Then
    $$\mathbb E[\int_0^tf_sdW_s\int_0^tg_sdW_s] = \int_0^t \mathbb E[f_sg_s] ds$$
    In particular, if $f_s = g_s$, it follows that
    $$\mathbb E[(\int_0^tf_sdW_s)^2] = \int_0^t \mathbb E[f_s^2] ds$$*

3. Change of measure. 

    > Given $Z\sim N(0, 1)$ on probability space $(\Sigma, F, \mathbb{P})$, define a new probability measure $\mathbb{Q}$ by taking $$G(z) = \frac{d\mathbb{Q}}{d\mathbb{P}} = \exp(-\gamma z - \frac{\gamma^2}{2})$$ where $\gamma \in \mathbb{R}$ is a free parameter. Prove that $\tilde{Z}=\gamma + Z$ is $\tilde{Z} \sim N(\gamma, 1)$ w.r.t $\mathbb{P}$, $\tilde{Z} \sim N(0, 1)$ w.r.t $\mathbb{Q}$.

    > $d\mathbb{Q}(z) = G(z) d\mathbb{P}(z)$

4. Feynman Kac 

    > The Ito process of the form $dX_t = \mu(t, X_t)dt + \sigma(t, X_t)dW_t$, there is a measurable function $g(x)$ such that 
    $$g_t(t,x) + g_x(t, x) \mu(t,x) + \frac{1}{2} g_{xx}(t,x)\sigma(t,x)^2 = 0$$
    with an appropriate boundary condition. $g(T,x) = h(x), \ g(t,x)=\mathbb{E}\left[h(X_T) \big| X_t=x\right].$
    > [http://quant.stackexchange.com/questions/10359/is-there-an-intuitive-explanation-for-the-feynman-kac-theorem](http://quant.stackexchange.com/questions/10359/is-there-an-intuitive-explanation-for-the-feynman-kac-theorem)

5. Girsanov
    
    > Let $\gamma_t$ be a process adapted to the Brownian filtration $F_t^W$ and satisfy the condition $$E_{\mathbb{P}}[e^{\frac{1}{2}\int_0^t\gamma_s^2ds}] < \infty$$. 
    Define a new measure $\mathbb{Q}$ such that 
$$\frac{d\mathbb{Q}}{d\mathbb{P}}=\exp(-\int \gamma_s dW_s - \frac{1}{2}\int_0^t\gamma_s^2 ds)$$
or
$$E_{\mathbb{Q}}[X] = E_{\mathbb{P}}[X\exp(-\int \gamma_s dW_s - \frac{1}{2}\int_0^t\gamma_s^2 ds)]$$
The the process $\tilde{W}_t$ defined by $d\tilde{W}_t = \gamma_tdt + dW_t$ is a $\mathbb{Q}$-Brownian motion.

6. Mean and variance of $\int W_s d W_s$, $\int W_s^2 d W_s$, $\int \sqrt{s} \exp(W_s^2 / 8) dW_s$, $\int_0^t W_s ds$

    > $\mathbb E[\int W_s d W_s]^2 = \frac{t^2}{2}$
    > $\mathbb E[\int W_s^2 d W_s] = t^3$
    > $\mathbb E[\int_0^t W_s ds]^2 = \frac{t^2}{3}$
    > http://www.slideshare.net/ssakpi/stochastic-calculus-main-results

7. Is following process a martingale? $X_t = W_t^2$, $X_t = W_t^3 - 3tW_t$

    > No and Yes.

8. Let $W_t$ be a Brownian motion starting at 0. Let $\beta_k(t)=\mathbb E[W^k_t]$. Using Ito’s formula, show that $$\beta_k(t) = \frac{k (k - 1)}{2} \int_0^t \beta_{k-2}(s) ds$$

9. Brownian Bridge.

    > $BB(X_s | X_0 = 0, X_t = x) \sim N(\frac{s}{t}x, \frac{s(t-s)}{t})$

10. Let $X_t$ be $dX_t = \mu X_t dt + \sigma X_t dW_t$, find $\alpha$ such that $X^\alpha$ is a martingale. 

    > $dX^{\alpha} = (\alpha \mu + \frac{\alpha(\alpha - 1)}{2}\sigma^2) X^{\alpha} dt + \alpha \sigma X^{\alpha} dW_t$

