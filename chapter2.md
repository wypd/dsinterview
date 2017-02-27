# Chapter 2. Probability

---
## Part 1. Permutation and Combinatin

1. Given N points drawn randomly on the circumference of a circle, what is the probability that they are all within a semicircle?

    > $\frac{n}{2^{n-1}}$
    
    > [http://math.stackexchange.com/questions/325141/probability-that-n-points-on-a-circle-are-in-one-semicircle](http://math.stackexchange.com/questions/325141/probability-that-n-points-on-a-circle-are-in-one-semicircle)
    
2. Five letters to five people. What is the probability that all five letters are all in the wrong envelopes? What if there are $n$ letters? <font color = 'red'> What is the probability of exactly $r$ matches?

    > Inclusion and exclusion.
    ```
    P(A or B or C .... or E) = P(A) + P(B) + P(C) + P(D) + P(E)
                          - P(A and B) - P(B and C) -....
                          + P(A and B and C) + P(B and C and D) + ....
                          - P(A and B and C and D) - P(...) -......
                          + P(A and B and C and D and E)
    ```
    > $1 - (1 - \frac{1}{2} + \frac{1}{6} - \frac{1}{24} + \frac{1}{120}) = \frac{44}{120} = \frac{11}{30}$
    > For $n$ folders
    > $P = 1 - 1 + \frac{1}{2!} - \frac{1}{3!} + \frac{1}{4!} ... \rightarrow e^{-1}$
    
    > [http://mathforum.org/library/drmath/view/56592.html](http://mathforum.org/library/drmath/view/56592.html)

3. You are taking out candies one by one from a jar that has 10 red candies, 20 blue candies, and 30 green candies in it. What is the probability that there are at least 1 blue candy and 1 green candy left in the jar when you have taken out all the red candies?

    > Consider two events: green before blue and red, blue before green and red.
    > $\frac{2}{6}\frac{3}{4} + \frac{3}{6}\frac{2}{3}=\frac{7}{12}$

4. What is the expected number of cards that need to be turned over in a regular 52-card deck in order to see the first ace?

    > For each card, there is $1/5$ chance that it is drawn before ace.
    > $\frac{48}{5} + 1=\frac{53}{5}$

5. (**Coupon Problem**) There are $N$ distinct types of coupons in cereal boxes and each types, independent of prior selections, is equally likely to be in a box. 1) If a child wants to collect a complete set of coupons with at least one of each type, how many coupons on average are needed to make such a complete set? 2) If the child has collected n coupons, what is the expected number of distinct coupon types?

    > 1) $\mathbb{E}[T] = \sum_{n=1}^N \frac{N}{n}$
    > 2) 
    \begin{align*}
    \mathbb{E}[N] & = \sum_i P(\text{coupon i is collected}) \\ 
    & = N \cdot P(\text{coupon 1 is collected}) \\
    & = N (1 - (\frac{N-1}{N})^n)
    \end{align*}
    
    > [https://en.wikipedia.org/wiki/Coupon_collector%27s_problem](https://en.wikipedia.org/wiki/Coupon_collector%27s_problem)

6. A fair coin is tossed n times, what is the probability that no two consecutive heads appear? 
    
    > $f(n)$: number of paths that do not have two consecutive heads appear. 
    > $f(1) = 2$, $f(2) = 3$
    > $f(n) = f(n - 1) + f(n - 2)$
    > Fibonacci!
    
    > [http://math.stackexchange.com/questions/1355511/probability-that-no-two-consecutive-heads-occur](http://math.stackexchange.com/questions/1355511/probability-that-no-two-consecutive-heads-occur)

    > Similar question: Suppose we roll a fair die 100 times. What is the probability of a run of at least 10 sixes?


7. Poker combinations

    > A single pair: ${13 \choose 1} \cdot {4 \choose 2} \cdot {12 \choose 3} \cdot {4 \choose 1} ^3$
    > Two pairs: ${13 \choose 2} \cdot {4 \choose 2}^2 \cdot {44 \choose 1}$
    > A triple: ${13 \choose 1} \cdot {4 \choose 3} \cdot {12 \choose 2} {4 \choose 1}^2$
    > Full house: ${13 \choose 1} \cdot {4 \choose 3} \cdot {12 \choose 1} \cdot {4 \choose 2}$
    > Four of a kind: ${13 \choose 1} \cdot {48 \choose 1}$
    > A Straight: $10 \cdot {4 \choose 1}^5$

8. How many different ways can you invest \$20,000 into five funds in increments of \$1,000?

    > ${24 \choose 4} = \frac{24 \times 23 \times 22 \times 21}{24} = 10626$
    > [Stars and bars](https://en.wikipedia.org/wiki/Stars_and_bars_%28combinatorics%29)
    > [https://en.wikipedia.org/wiki/Twelvefold_way](https://en.wikipedia.org/wiki/Twelvefold_way)
    
9. You are making chocolate chip cookies. You add N chips randomly to the cookie dough, and you randomly split the dough into 100 equal cookies. How many chips should go into the dough to give a probability of at least $90\%$ that every cookie has at least one chip?

    > Splitting $N$ chips on $100$ cookies, we have $N + 100 - 1 \choose N$ different choices.
    > Among all choices, we have $N - 1 \choose N - 100 $ different ways to spread at least one chip on each cookie. 

10. I throw 1 die 4 times, trying to reach at least one  6, you throw 2 dice 24 times and try to reach at least one double 6 (6,6). Who has greater chance of winning?

    > At least one 6 out of 4 rolls: $p_1 = 1 - (\frac{5}{6})^4$
    > At least double 6 in 24 rools: $p_2 = 1 - (\frac{35}{36})^{24}$
    > $p_1 > p_2$
    
11. Suppose we roll a fair die 10 times. What is the probability that the sequence of rolls is non-decreasing (i.e., the next roll is never less than the current roll)?

    > ${15 \choose 5}/6^{10}$
    > stars and bins. 
    
12.  Suppose we roll n dice and keep the highest one. What is the distribution of values?

    > 
    \begin{align*}
    P(K_{max}=k) & = P(K_{max} \le k) - P(K_{max} \le k - 1)\\
    & = (\frac{k}{6})^n-(\frac{k-1}{6})^n 
    \end{align*}

13. Four fair, 6-sided dice are rolled. The highest three are summed. What is the distribution of the sum?
    
    > 3 - 18. Not for hand calculation!

14. <font color = 'red'>Suppose we roll a die 6k times. What is the probability that each possible face comes up an equal number of times (i.e., k times)? Find an asymptotic expression for this probability in terms of k.</font>

    > $P = \frac{(6k)!}{6^{6k}(k!)^6}$
    > By Stirling approximation, $n!\sim \sqrt{2\pi n}(\frac{n}{e})^n$, 
    > $\lim_{k\rightarrow \infty} \frac{P}{\sqrt{6} (2\pi k)^{5/2}} = 1$

15. Your n guests (n>0) are all baseball fans and they wear baseball caps. There is a total of s teams (s>0) in the league. Everyone of your guests is equally likely to be a fan of any one of these teams. Compute the expected number of people who will pick a cap from their own team.

    > [http://www.cseblog.com/2010/10/baseball-party.html](http://www.cseblog.com/2010/10/baseball-party.html)
    
    > For each guest, the probability to pick up the correct hat is $\frac{n + s - 1}{ns}$, for $n$ people, the total number is $\frac{n + s - 1}{s}$

## Part 2. Bayesian

1. (**Envelop paradox**) You are given two indistinguishable envelopes, each containing money, one contains twice as much as the other. You may pick one envelope and keep the money it contains. Having chosen an envelope at will, but before inspecting it, you are given the chance to switch envelopes. Should you switch?
    
    > [exchange paradox](https://en.wikipedia.org/wiki/Two_envelopes_problem)

2. **Monty Hall.**

    > Let 1 be the chosen door, 2 be the openned door. $C_i$ means the event that the car is behind door $i$, $G_2$ be the event that door 2 is opened.
    > $P(C_i) = \frac{1}{3}$, $P(G_2|C_1) = \frac{1}{2}$, $P(G_2|C_2) = 0$, and $P(G_2|C_3)=1$. So we have $P(C_1|G_2) = \frac{1}{3}$.
    
    > [https://en.wikipedia.org/wiki/Monty_Hall_problem](https://en.wikipedia.org/wiki/Monty_Hall_problem)
    
3. I have two children. One is a boy born on a Tuesday. What is the probability I have two boys?

    > Denote $A$ a boy born on a Tuesday, $B$ two boys, $\bar{B}$ 1 boy.
    \begin{align*}
    P(B|A) & = \frac{P(A|B)P(B)}{P(A|B)P(B) + P(A|\bar{B})P(\bar{B}) } \\
    & = \frac{\frac{13}{49}\frac{1}{3}}{\frac{13}{49}\frac{1}{3} + \frac{1}{7}\frac{2}{3}} \\
    & = \frac{13}{27}
    \end{align*}

    > [http://math.stackexchange.com/questions/4400/boy-born-on-a-tuesday-is-it-just-a-language-trick](http://math.stackexchange.com/questions/4400/boy-born-on-a-tuesday-is-it-just-a-language-trick)

4. <font color = 'red'>Choosing the largest dowry. The king offers a wise man a chance to marry the young lady in the court with the largest dowry. The amounts of the dowies are written on slips on paper and mixed. A slip is drawn at random and the wise man must decide whether that is the larget dowry or not. If he decides it is, he gets the lady and her dowry if he is correct; otherwise he gets nothing. How should the wise man make his decision? What if we know that the dowry is uniformly distributed from (0, 1)?</font>

    > The optimal policy is to find $r$, then after first $r$ samples, take the first sample that are larger than all previous ones. 
    > 
    \begin{align*} 
    P(r) & = \sum_{i = 1}^n P(\text{applicant i is selected} \cap \text{applicant i is the best} ) \\
    & = \sum_{i = r + 1}^n \frac{1}{n} P(\text{best among i - 1 is in first r applicants} | \text{applicant i is the best}) \\
    & = \sum_{i = r + 1}^n \frac{r}{n (i - 1)} \\
    & \rightarrow -\frac{r}{n} \ln(\frac{r}{n})
    \end{align*}
    
    > [https://en.wikipedia.org/wiki/Secretary_problem](https://en.wikipedia.org/wiki/Secretary_problem)

## Part 3. Random Variables

### Poisson Distribution

1. To organize a charity event that costs \$100K, an organization raises funds. Independent of each other, one donor after another donates some amount of money that is exponentially distributed with a mean of \$20K. The process is stopped as soon as \$100K or more has been collected. Find the distribution, mean, and variance of the number of donors needed until at least \$100K has been collected. 

    > Poisson distribution and exponential distribution
    
    > $X_i \sim \exp(20)$ and $N_{100} \sim Poisson(5)$.
    
2. What is the probability that random variable of poisson distribution takes even values?

    > Denote $S_o = \sum_{n = 2k + 1} P(N = n)$ and $S_e = \sum_{n = 2k} P(N = n)$, where $P(N = n) = \frac{\lambda^n}{n!} e^{-\lambda}$.
    > We have $S_o + S_e = \sum_n P(N=n) = 1$ and $S_e - S_o = \sum_n (-1)^n P(N = n) = e^{-2\lambda}$, so $S_e = (1 + e^{-2\lambda}) / 2$.

3. Nearest neighbor distribution.

    > The nearest neighbor function, as opposed to the spherical contact distribution function, is defined in relation to some point of a point process already existing in some region of space. More precisely, for some point in the point process $N$, the nearest neighbor function is the probability distributioni of the distance from that point to the nearest or closest neighboring point. 
    
    > For a Poisson point process in two dimensional space, $P(D\ge r) = P(N(r) = 0) = e^{-\lambda \pi r^2}$. Then, the expected distance is $\mathbb{E}[D] = \frac{1}{2\sqrt{\lambda}}$
    
    > [https://en.wikipedia.org/wiki/Nearest_neighbour_distribution](https://en.wikipedia.org/wiki/Nearest_neighbour_distribution)
    > [http://mathoverflow.net/questions/218751/nearest-neighbor-for-planar-poisson-is-normally-distributed](http://mathoverflow.net/questions/218751/nearest-neighbor-for-planar-poisson-is-normally-distributed0)

4. Denote as $\{N(t), t ≥ 0\}$ a Poisson process with rate $\lambda$. Let $W$ be the waiting time until for the ﬁrst time no event in $\{N(t)\}$ has occurred for the last $\tau$ time units. Derive $\mathbb E[W]$. For example, $\{N(t)\}$ might stand for the sequence of cars passing at a crosswalk. A pedestrian needs $\tau$ seconds to cross the road — how long does it on average take before he has reached the other side of the road?

    > Related problem: If a car passes at the crosswalk on average every 10 seconds and you need 20 seconds to pass the road, how long does it take you on average to cross the road? [http://www.cseblog.com/2011/12/crossing-road.html](http://www.cseblog.com/2011/12/crossing-road.html)

    > \begin{align*}
    \mathbb E[W] & = \int_{t=0}^{\infty} \mathbb E [W|t_1 = t] f(t) dt\\
    & = \int_{0}^{\tau} (t + \mathbb E[W]) f(t) dt + \tau (1 - F(\tau)) \\
    & = (1 - e^{-\lambda \tau}) \mathbb E[W] + \frac{1}{\lambda}(1-e^{-\lambda \tau}) \\
    \mathbb{E}[W] & = \frac{1}{\lambda}(e^{\lambda\tau} - 1)
    \end{align*}

    > Note: interval questions can be written in a recursive way that depends on the first event. 
    
5. Related to the preceding problem, let $U$ be the waiting time until two events occur within $\tau$ time units. Derive $\mathbb E[U]$. In some applications, the event that has latency $U$ is called a “coincidence.” For example, a volume of biological tissue could be permanently destroyed when two damaging particles are absorbed within $\tau$ (or less) time units. The idea here is that following the ﬁrst absorption the tissue needs to recover for $\tau$ time units; this opens a window of vulnerability during which a further (second) particle has a lethal eﬀect.

    > \begin{align*}
    \mathbb E [U] & = \int_{0}^{\tau} (\frac{1}{\lambda} + t)f(t) dt + \int_{\tau}^{\infty}(\frac{1}{\lambda} + \tau + \mathbb E[U])f(t) dt\\
    & = \frac{1}{\lambda}(2 - e^{-\lambda\tau}) + \mathbb E[U] e^{-\lambda \tau} \\
    \mathbb E[U] & = \frac{1}{\lambda}\frac{2-e^{-\lambda \tau}}{1 - e^{-\lambda \tau}}
    \end{align*}
    

### Uniform distirbution 

1. Alice writes two distinct real numbers between 0 and 1 on two sheets of paper. Bob selects one of the sheets randomly to inspect it. He then has to declare whether the number he sees is the bigger or smaller of the two. Find and prove a strategy so that Bob can guess correctly with more than 0.5 chance.

    > Denote the two number Alice wrote as $x$ and $y$. WLOG, Bob took $x$. Then Bob generated another uniform number $u \sim Unif(0, 1)$. Bob would claim $x$ is the larger one if $x\ge u$, otherwise $x$ is the smaller one.
    > 
    \begin{align*}
    P(\text{Bob win}) & = P(x > u | x > y)P(x > y) + P(x < u | x < y)P(x < y) \\
    & = \frac{\max(x, y) - \min(x, y) + 1}{2} 
    \end{align*}

2. Select numbers uniformly distributed between 0 and 1, one after the other, as long as they keep decreasing; stop selecting when you obtain a number that is greater than the previous one you selected. 1) On average, how many numbers have you selected? 2) What is the average value of the smallest number you have selected?

    > For any $n$ RVs, there is only $\frac{1}{n!}$ chance that it is a strickly decreasing sequence. That is $P(N \ge n) = \frac{1}{(n - 1)!}$ since the first $n - 1$ RVs should be strickly decreasing. 
    > So the expected number of runs $\mathbb{E}[N] = e$. 
    > $\mathbb{E}[X_{1, N}] = \mathbb{E}[\mathbb{E}[X_{1, n}]|N=n] = \sum_{n = 1}\frac{1}{n + 1}[\frac{1}{(n - 1)!} - \frac{1}{n!}] = 3 - e$
    
    > [http://math.stackexchange.com/questions/1447915/choosing-increasing-numbers-from-a-uniform-distribution/1447933](http://math.stackexchange.com/questions/1447915/choosing-increasing-numbers-from-a-uniform-distribution/1447933)

3. The stick drops and breaks at one place. Then the larger piece is taken and dropped again, breaking at one place. What is the probability that the three pieces could form a triangle?

    > WLOG, $X_1 \sim Unif(0, 1/2)$ and $X_2 \sim Unif(0, 1 - X_1)$ be the two pieces. The probability to form a triangle is 
    > $\int_{0}^{1/2}2\frac{2x}{1 - x}dx = 2(\ln(2) - 1)$

4. Take a stick and break it randomly into three pieces. What is the probability you can form a triangle from the pieces?

    > [http://mathoverflow.net/questions/2014/if-you-break-a-stick-at-two-points-chosen-uniformly-the-probability-the-three-r](http://mathoverflow.net/questions/2014/if-you-break-a-stick-at-two-points-chosen-uniformly-the-probability-the-three-r)
    > http://www.cut-the-knot.org/Curriculum/Probability/TriProbability.shtml

5. The average length of the shortest / longest segment if the stick is broken into $n$ pieces. 

    > Shortest:
    > $P(X_{(1)} > c) = (1 - nc)^{n - 1}$
    > $\mathbb{E}[X_{(1)}] = \int_{0}^{1/n}P(X_{(1)}>c)dc= \frac{1}{n^2}$
    > Longest:
    > $P(X_{(n)} > c) = n(1-x)^{n-1} - \binom{n}{2} (1 - 2x)^{n-1} + \cdots + (-1)^{k-1} \binom{n}{k} (1 - kx)^{n-1} + \cdots $
    > $\mathbb{E}[X_{(n)}] = \int P(X_{(n)} > x) dx = \sum_{k=1}^n \binom{n}{k} (-1)^{k-1} \int_0^{1/k} (1 - kx)^{n-1} dx = \sum_{k=1}^n \binom{n}{k} (-1)^{k-1} \frac{1}{nk} = \frac{1}{n} \sum_{k=1}^n \frac{\binom{n}{k}}{k} (-1)^{k-1} = \frac{H_n}{n}$
    
    > [http://math.stackexchange.com/questions/14190/average-length-of-the-longest-segment](http://math.stackexchange.com/questions/14190/average-length-of-the-longest-segment)

6.  Suppose $X_1, ..., X_n$ are independent identical distributed from [0, 1] and uniform on the interval. What is the expected value of the maximum? What is the expected value of the difference between maximum and the minimum?

    > $\frac{n}{n+1}$ and $\frac{n-1}{n+1}$

7. Buffon's needle with horizontal and vertial rulings. Suppose we toss a needle of length 2l on a gride with both horizontal and vertical rulings spaced one unit apart. What is the mean number of lines the needle crosses?

    > $\mathbb{E}N = 2 \frac{2}{\pi} \int_0^{\pi/2} 2l \cdot sin(\theta) d\theta = \frac{8l}{\pi}$

8. <font color = 'red'> A fair, n-sided die is rolled and summed until the sum is at least n. What is the expected number of rolls? </font>

    > \begin{align*}
    \mathbb E[N_1] & = 1 \\
    \mathbb E[N_2] & = 1 + \frac{1}{n}\mathbb E[N_1] \\
    \cdots \\
    \mathbb E[N_n] & = 1 + \sum_{i=1}^n \frac{1}{n} \mathbb E[N_i] \\
    & = (1+\frac{1}{n})^{n-1} \rightarrow e 
    \end{align*}

9. <font color = 'red'> A camel is loaded with straws until it's back breaks. Each straw has a weight uniformly distributed between 0 and 1, independent of other straws. The camel's back breaks as soon as the total weight of all the straws exceeds 1. Find the expected weight of the last straw that breaks the camel's back. </font>

    > Let $f(x)$ be the density function of the last straw. 
    > $f(x) = \sum_n P(S_n < 1 \le S_n + x) = \sum_n P(S_n < 1) - \sum_n P(S_n < 1 - x)$
    > where $\sum_n P(S_n < x) = x + \frac{x^2}{2!} + \frac{x^3}{3!} ... = e^x - 1$
    > So, $f(x) = e - e^{1-x}$, $\mathbb E X = \int xf(x) dx = 2 - e / 2$
    
    > [http://math.stackexchange.com/questions/734700/draws-from-the-uniform-distribution-are-taken-until-the-sum-exceeds-1-what-is-t](http://math.stackexchange.com/questions/734700/draws-from-the-uniform-distribution-are-taken-until-the-sum-exceeds-1-what-is-t)


### Normal Distribution 

1. If $X$ follows normal distribution $N(\mu, \sigma^2)$, what is $\mathbb{E}[X^n]$?
    
    > $\mathbb{E}[e^X] = \exp(\mu + \frac{\sigma^2}{2})$
    > Let $Y\sim N(0,1)$, $\mathbb{E}[Y^n] = \int y^n \exp(-\frac{y^2}{2})dy = (n - 1)\int y^{n-2} \exp(-\frac{y^2}{2})dy=(n-1)\mathbb{E}[Y^{n-2}]$
    > $\mathbb{E}[(X-\mu)^n] = \begin{cases}0, & n~\text{odd},\\ \sigma^n(n-1)(n-3)\cdots 3\cdot 1,& n~\text{even}\end{cases}$

2. Counter case when $X$ and $Y$ are normal, but their joint is not normal. 

    > $X \sim N(0, 1)$, $Z \sim Unif(0, 1)$, $Y = XZ$ 

3. <font color = 'red'>Given two standard normal random variables $W$ and $Z$, what is the distribution of $W/Z$?</font>

    > \begin{align*}
    P(\frac{W}{Z}=r) & = P(W = rZ) \\
    & = \int \frac{1}{2\pi} \exp\{-\frac{z^2}{2}-\frac{z^2r^2}{2}\}dz \\
    & = \frac{1}{\sqrt{2\pi(1+r^2)}}
    \end{align*}

## Part 4. Stochastic Process

1. There is one amoeba in a pond. After every minute the amoeba may die, stay the same, split into two or split into three with equal probability. All its offspring, if it has any, will behave the same (and independent of other amoebas). What is the probability the amoeba population will die out?

    > $4p = 1 + p + p^2 + p^3$
    
    > For absorbing probability with one barrier, say, 0, then $p_n = p_1^n$. 
    
2. Basketball scores. A basketball player is taking 100 free throws. She scores one point if the ball passes through the hoop and zero if she misses. She has scored on her first throw and missed on her second. For each of the following throw the probability of her scoring is the fraction of throws she has made so far. For example, if she has scored 23 points after throws (including the first and the second), what is the probability that she scores exactly 50 baskets?

    > $P((win=x, loss=y)) = \frac{1}{x+y-1}$
    
3. Two players, $A$ and $B$, alternatively toss a fair coin ($A$ tosses the coin first, then $B$ tosses the coin, then $A$, then $B$, ...). The sequence of heads and tails is recorded. If there is a head followed by a tail (HT subsequence), the game ends and the person who tosses the tail wins. what is the probability that $A$ wins the game?

    > $P(A) = \frac{1}{2}(1-P(A)) + \frac{1}{2}\frac{1}{3}$
    > $P(A) = \frac{4}{9}$

4. You have two boxes each containing two homing pigeons. To send a message, you select a box at random and pick a pigeon from that box. The pigeon delivers the message and returns to a random box (not necessarily the one it was picked from). What is the expected number of messages delivered before you discover that the box you choose at random is empty? (From Nick’s Quant blog)

    > Define states as $(2,2)$, $(1,3)$, $(0,4)$, and $End$.
    
    > [http://quantquestions.tumblr.com/post/35700179655/homing-pigeons](http://quantquestions.tumblr.com/post/35700179655/homing-pigeons)

5. At a theater ticket office, 2n people are waiting to buy tickets. n of them have only \$5 bills and the other n people have only \$10 bills. The ticket seller has no change to start with. If each person buys \$5 ticket, what is the probability that all people will be able to buy their tickets without having to change positions?

    > Reflection theorem. Among all $2n \choose n$ paths, $2n \choose n + 1$ paths will dip $S_n = -1$. So the probability is $\frac{n}{n+1}$.
    
6. (**Ballot problem**) In an election, two candidates, Albert and Benjamin, have in a ballot box $a$ and $b$ voted respectively, $a<b$. If ballots are randomly drawn and tallied, what is the chance that at least once after the first tally the candidates have the same number of tallies?

    > [http://webspace.ship.edu/msrenault/ballotproblem/](http://webspace.ship.edu/msrenault/ballotproblem/)
    > [https://en.wikipedia.org/wiki/Bertrand%27s_ballot_theorem](https://en.wikipedia.org/wiki/Bertrand%27s_ballot_theorem)

7. <font color = 'red'>Players A and B match pennies for $N$ times. They keep a tally of their gains and losses. After the first toss, what is the chance that at no time during the game will they be even? </font>

    > $P(\text{no tie}) = \frac{N - 1 \choose n}{2^{N-1}}$

8. Dynamic dice game. A casino comes up with a fancy dice game. It allows you to roll a dice as many times as you want unless a 6 appears. After each roll, if 1 - 5 appears, you will win that amount; but if 6 appears, all the moneys you have won in the game is lost and the game stops. After each roll, if the dice number is 1-5, you can decide whether to keep the money or keep on rolling. How much are you willing to pay to play the game (if you are risk neutral)?

    > Start from the stopping time when $\mathbb{E}X \le \frac{5}{6}(\mathbb{E}X + 3)$

9. On average, how many times must a 6-sided die be rolled until there are two rolls in a row that differ by 1 (such as a 2 followed by a 1 or 3, or a 6 followed by a 5)? What if we roll until there are two rolls in a row that differ by no more than 1 (so we stop at a repeated roll, too)?

    > $\mathbb E = 1 + \frac{1}{6}(\mathbb E_1 + \mathbb E_2 + \mathbb E_3 + \mathbb E_4 + \mathbb E_5 + \mathbb E_6)$

10. On average, how many times must a pair of 6-sided dice be rolled until all sides appear at least once?

    > Define the state and get the transition matrix
    
11. Suppose we can roll a 6-sided die up to n times. At any point we can stop, and that roll becomes our “score”. Our goal is to get the highest possible score, on average. How should we decide when to stop?

    > dynamic programming. Start from the ending state and deduct backward

12. Given the set of numbers from 1 to n: $\{ 1, 2, 3 ... n \}$. We draw n numbers randomly (with uniform distribution) from this set (with replacement). What is the expected number of distinct values that we would draw? 

    > $f(k) = \mathbb{E} [\text{number of distinct number among k draws}]$
    > $f(n) = \sum_i P(\text{number i is picked}) = n(1 - (\frac{n-1}{n})^n)$
    
13. There are 26 black(B) and 26 red(R) cards in a standard deck. A run is a maximum block of consecutive cards of the same color. For example, a sequence RRRRBBBRBRB of only 11 cards has 6 runs; namely, RRRR, BBB, R, B, R, B. Find the expected number of runs in a shuffled deck of cards.

    > Let $Y_i$ denote the event that $X_i$ and $X_{i + 1}$ are of the different colors. 
    > Then $\mathbb E [ \sum_{i = 1}^{n - 1}Y_i ] + 1 = n/2 + 1$

14. <font color = 'red'>Three people start with integer amounts a,b and c. In each round, each one tosses a fair coin. If not all faces are the same, the person with the different face gets a rupee from each of the other two. If all faces are the same, no money is exchanged. This process is repeated till one of them gets bankrupt. What is the expected number of rounds till the game ends?</font>
    
    > $Y_n = A_n \cdot B_n \cdot C_n$, and $S_n = a_n + b_n + c_n$ where $A_n = A_{n - 1} + a_n$ and so forth.
    > $\mathbb{E} [Y_n] = Y_{n-1} - \frac{3}{4}(S_{n-1} + 2)$, thus $Y_n - \frac{3n}{4}(S_n - 2)$ is a martingale. 
    > $\tau = \frac{4abc}{3(a + b + c - 2)}$

    > [http://www.cseblog.com/2011/02/coin-toss-bankruptcy.html](http://www.cseblog.com/2011/02/coin-toss-bankruptcy.html)

15. <font color = 'red'>Consider a random walk around the edges of a square. From any vertex, the probability of moving to any adjacent vertex is 0.5. Suppose the walk stops as soon as after all traversing through all the vertices, you return to your starting vertex. What is the expected path length?</font>

    > Denote $A, B, C, D$ as four vertexs. 
    > Step 1. Average steps from $A$ to $B$, $A$ to $C$ is $N_{AB} = N_{AC} = 3$, and $N_{AD} = 4$
    > Step 2. Probability of visit $D$ before $B$, two absorbing state random walk, $P_B = \frac{1}{3}$. 
    > Step 3. Three situations. 1) Pass A, B, C, then to D and back to A, 2) Pass A, B, D, C, then back to A, 3) Pass A, D, C, then to B, and to A. 
    > $N = \frac{2}{3}(4 + 3 + 3) + \frac{1}{3}(4 + 4) = \frac{28}{3}$

16. Recursive of random walk. (1D, 2D, and 3D)

    > Let $X_i = \begin{cases} 1, & w.p.\frac{1}{2} \\ -1, & w.p.\frac{1}{2}\end{cases}$, and $S_n = \sum_i X_i$. 
    > $P(S_{2n} = S_0) = {2n \choose n} (\frac{1}{2})^{2n} \rightarrow \frac{1}{\sqrt{\pi n}}$ by Stirling approximation.
    > The expected number of return to $S_0$ is $\mathbb {E} = \sum_n P(S_{2n} = S_0) \rightarrow \infty$.
    > The probability of return to $S_0$ is 1. 
    
    > Zero-One law. 
    > If $P(S_n = S_0) = 1$, then $P(S_n = S_0 \text{ infinity often}) = 1$; if $P(S_n = S_0) < 1$, then $(S_n = S_0 \text{ infinity often}) = 0$
    
17. <font color = 'red'>Coin flips. 1) Given a string $s$, find $\mathbb E [s]$ = expected waiting time for first occurence.</font>

    > Let $F(s)$ be frequency of a string, e.g., $F(s) = \frac{1}{2^{n(s)}}$, and $V(x) = \{t | t \in R(k) \text{for some k and s overlaps at t} \}$, e.g., $V(HTHTH) = \{H, HTH\}$. 
    > $\mathbb E [s] = \frac{1}{F(s)} + \sum_{t \in V(s)}\frac{1}{F(t)}$
    
18. <font color = 'red'>Random walk first hitting time.</font>

    > Let $\tau_k = \min \{t |  S_t = k, t > 0\}$ .
    > 
    \begin{align*} 
    P(\tau_0 = 2k) & = P(\tau_0 = 2k | X_1 = 1)\\
    & = P(\tau_1 = 2k - 1)\\
    & = \frac{2k - 1 \choose k}{2^{2k - 1}} \frac{1}{2k - 1} \\
    & = 2^{-2k+1}\frac{1}{k}{2k - 2 \choose k - 1} 
    \end{align*}

19. <font color = 'red'>Brownian motion first fitting time .</font> Let $W(t)$ be a standard Wiener process and $\tau_x$ be the first passage time to level $x$. What is the probability density function of $\tau_x$ and the expected value of $\tau_x$?

    > 1) Without drift term
    > Let $\tau_x = \min \{ t | W_t = x, t > 0\}$, where $x>0$.
    \begin{align*}
    P(\tau_x < t) & = 2P(W_t \ge x) = 2\Phi(-\frac{x}{\sqrt{t}}) \\
    & = 2 \int^{-x/\sqrt{t}}\frac{1}{\sqrt{2\pi}}\exp(-\frac{z^2}{2})dz\\
    f_{\tau_x}(t)& = \frac{xt^{-3/2}}{\sqrt{2\pi}}\exp(-\frac{x^2}{2t})
    \end{align*}
    > 2) With drift term $Z_t = \mu t + W_t$, then $\exp(-2\mu Z_t)$ is a martingale.

    > [without drift](http://math.stackexchange.com/questions/840634/brownian-motion-first-hitting-time-distribution?rq=1)
    > [with drift](http://math.stackexchange.com/questions/1053294/density-of-first-hitting-time-of-brownian-motion-with-drift)

## Part 5. Correlation

1. If there is a $50\%$ probability that bond $A$ will default next year and a $30\%$ probability that bond $B$ will default. What is the range of probability that at least one bond defaults and what is the range of their correlation?

    > $0.5 - 0.8$, $\rho = \frac{\mathbb E XY - \mathbb EX \mathbb EY}{\sqrt{p_x (1-p_x) p_y (1-p_y)}}$

2. Counter example that $A$ and $B$ are independent, $A$ and $C$ are independent. Are $A$ and $B \cap C$ independent?

    > Roll two dice:
    > $A$: sum is even
    > $B$: first is even
    > $C$: second is even

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
    
----------



## Unsolved


1. Lady tea testing. 

    Fisher's exact test

9. A box contains n balls of n different colors. Each time, you randomly select a pair of balls, repaint the first to match the second, and put the pair back into the box. What is the expected number of steps until all balls in the box are of the same color?
    
    > http://mathoverflow.net/questions/41939/a-balls-and-colours-problem

18. A die is rolled repeatedly and summed. Show that the expected number of rolls until the sum is a multiple of n is n.

3. Suppose we have vv and uu, both are independent and exponentially distributed random variables with parameters μμ and λλ, respectively. How can we calculate the pdf of v−uv−u?

    [http://math.stackexchange.com/questions/115022/pdf-of-the-difference-of-two-exponentially-distributed-random-variables?noredirect=1&lq=1](http://math.stackexchange.com/questions/115022/pdf-of-the-difference-of-two-exponentially-distributed-random-variables?noredirect=1&lq=1)
    [http://math.stackexchange.com/questions/1805587/prove-that-mathbb-pxy-fracba-b-if-x-y-are-exponentially-distrib?noredirect=1&lq=1](http://math.stackexchange.com/questions/1805587/prove-that-mathbb-pxy-fracba-b-if-x-y-are-exponentially-distrib?noredirect=1&lq=1)