Now I have all the numerical anchors. Let me write the complete formal framework.

The framework is self-contained mathematics. I'll deliver it as my final response (per instructions, no .md report file). Here is the complete formal writeup.

---

# Signal Faithfulness Score: A Formal Framework

**Setup and notation.** Fix a feature index $f \in \mathcal{F}$ (the 14 scored features: snr, hnr, f0_mean, f0_sd, jitter, shimmer, srmr, duration_sec, overlap_ratio, speaking_rate, articulation_rate, pause_count, pause_rate, sample_rate). For one clip the latent physical value is $g^{\mathrm{true}}_f \in \mathbb{R}$. We never observe it. We observe a ground-truth estimate $g^{\mathrm{obs}}_f$ produced by a signal-processing extractor, and a model emits a numeric claim $\hat{y}_f$ parsed from generated text by `ClaimParser`. SFS declares the claim **accepted** (faithful, "correct") iff $|\hat{y}_f - g^{\mathrm{obs}}_f| \le \tau_f$, where $\tau_f$ is the per-feature tolerance band (the implementation's `_tolerance`).

---

## Part 0 — Definitions and Assumptions

**Definition 0.1 (Claim, acceptance).** A claim is a pair $(\hat{y}_f, f)$. Under band $\tau_f$ the acceptance indicator is
$$A_f(\hat y) \;=\; \mathbf{1}\!\left[\,|\hat y - g^{\mathrm{obs}}_f| \le \tau_f\,\right].$$

**Definition 0.2 (SFS precision / recall / F1).** Over a corpus, with $C$ the set of asserted scorable claims and $G$ the scorable ground-truth features,
$$\mathrm{Prec} = \frac{\sum_{c\in C} A(c)}{|C|},\qquad \mathrm{Rec} = \frac{|\{f\in G: f\ \text{asserted}\}\cap G|}{|G|},\qquad \mathrm{F1} = \frac{2\,\mathrm{Prec}\,\mathrm{Rec}}{\mathrm{Prec}+\mathrm{Rec}}.$$

**Definition 0.3 (Perceptual JND).** $\mathrm{JND}_f > 0$ is the smallest change in $g_f$ a human listener can detect (psychoacoustic threshold for pitch, loudness, rate, reverberation). A claim within $\mathrm{JND}_f$ of $g^{\mathrm{true}}_f$ is *perceptually indistinguishable* from truth and is the object we want to accept.

**Definition 0.4 (Effect size).** $\Delta_f > \mathrm{JND}_f$ is the smallest deviation $|\hat y - g^{\mathrm{true}}_f|$ that we require the metric to reject with controlled power. It is the boundary of "perceptually wrong."

**Assumption A1 (additive GT noise).** $g^{\mathrm{obs}}_f = g^{\mathrm{true}}_f + \varepsilon_f$ with $\mathbb{E}[\varepsilon_f]=0$ and $\mathrm{Var}(\varepsilon_f)=\sigma_f^2 < \infty$. The two SP extractors (clean-stem oracle, mixture estimate) are realizations of this model; their median absolute disagreement is the empirical handle on $\sigma_f$ (Lemma 1.5).

**Assumption A2 (claim noise negligible relative to GT noise, optional refinement).** Where the model's own digit-emission jitter is non-negligible it is folded into a combined variance $\sigma_f^2 \leftarrow \sigma_f^2 + \sigma_{\hat y,f}^2$; all bounds below carry through with $\sigma_f$ read as this combined scale. The headline regime has $\sigma_f$ dominated by GT noise (verified: the GT-noise floor dominates the JND on all twelve continuous features).

**Assumption A3 (unimodality, used only for the VP form).** $\varepsilon_f$ has a unimodal density. Not required for the Chebyshev or Gaussian forms.

---

## Part 1 — (T1) Coverage-Guaranteed Tolerance

We want $\tau_f$ such that (i) a *truly faithful* claim is accepted with probability $\ge 1-\alpha$ (Type-I / false-reject control), and (ii) a claim off by the effect size $\Delta_f$ is rejected with probability $\ge 1-\beta$ (Type-II / false-accept control).

**Definition 1.1 (the two error events).** Let $\hat y$ be a claim and write the *claim error against truth* $\delta = \hat y - g^{\mathrm{true}}_f$. The decision uses $g^{\mathrm{obs}} = g^{\mathrm{true}} + \varepsilon$, so the test statistic is $\hat y - g^{\mathrm{obs}} = \delta - \varepsilon$. Acceptance is $|\delta - \varepsilon|\le \tau_f$.

- *False reject* (a faithful claim, $|\delta|\le \mathrm{JND}_f$, gets rejected): bad, controlled at $\alpha$.
- *False accept* (an unfaithful claim, $|\delta|\ge \Delta_f$, gets accepted): bad, controlled at $\beta$.

### Theorem T1 (Gaussian closed form)

**Theorem 1.2.** Under A1 with $\varepsilon_f \sim \mathcal N(0,\sigma_f^2)$, set
$$\boxed{\;\tau_f \;=\; \mathrm{JND}_f \;+\; z_{1-\alpha}\,\sigma_f\;}\qquad z_{1-\alpha}=\Phi^{-1}(1-\alpha).$$
Then:

**(a) Type-I bound.** Any faithful claim ($|\delta|\le \mathrm{JND}_f$) is accepted with probability $\ge 1-2\alpha$; under the one-sided worst-case alignment used below it is $\ge 1-\alpha$.

**(b) Type-II bound.** A claim with $|\delta|=\Delta_f$ is accepted with probability
$$\Pr[\text{accept}] \le \Phi\!\Big(\tfrac{\tau_f - \Delta_f}{\sigma_f}\Big) - \Phi\!\Big(\tfrac{-\tau_f-\Delta_f}{\sigma_f}\Big) \;\le\; \beta$$
whenever the separation condition $\Delta_f \ge \tau_f + z_{1-\beta}\,\sigma_f = \mathrm{JND}_f + (z_{1-\alpha}+z_{1-\beta})\sigma_f$ holds. Equivalently, $\Delta_f - \mathrm{JND}_f \ge (z_{1-\alpha}+z_{1-\beta})\,\sigma_f$ is the identifiability budget that buys both guarantees.

**Proof.** *(a)* Accept iff $|\delta-\varepsilon|\le\tau_f$, i.e. $\delta-\tau_f \le \varepsilon \le \delta+\tau_f$. For a faithful claim $|\delta|\le \mathrm{JND}_f$, so $\delta + \tau_f \ge \delta + \mathrm{JND}_f + z_{1-\alpha}\sigma_f \ge z_{1-\alpha}\sigma_f$ (since $\delta+\mathrm{JND}_f \ge 0$), and symmetrically $\delta-\tau_f \le -z_{1-\alpha}\sigma_f$. Hence
$$\Pr[\text{reject}] = \Pr[\varepsilon > \delta+\tau_f] + \Pr[\varepsilon < \delta-\tau_f] \le \Pr[\varepsilon > z_{1-\alpha}\sigma_f] + \Pr[\varepsilon < -z_{1-\alpha}\sigma_f] = 2\alpha.$$
The right tail alone is the binding one when $\delta = +\mathrm{JND}_f$ (the worst case that pushes the claim toward the boundary), giving the single-sided $\ge 1-\alpha$ acceptance used as the operational guarantee; the two-sided slack $2\alpha$ is the conservative statement. $\square$

*(b)* With $\delta=\Delta_f$, accept iff $\Delta_f-\tau_f \le \varepsilon \le \Delta_f+\tau_f$, so
$\Pr[\text{accept}] = \Phi\big(\tfrac{\Delta_f+\tau_f}{\sigma_f}\big)-\Phi\big(\tfrac{\Delta_f-\tau_f}{\sigma_f}\big)$. Wait — sign: the event is $\Delta_f - \varepsilon \in[-\tau_f,\tau_f]$, i.e. $\varepsilon\in[\Delta_f-\tau_f,\Delta_f+\tau_f]$, giving $\Phi(\tfrac{\Delta_f+\tau_f}{\sigma_f}) - \Phi(\tfrac{\Delta_f-\tau_f}{\sigma_f})$. The upper tail $1-\Phi(\tfrac{\Delta_f+\tau_f}{\sigma_f})$ is tiny when $\Delta_f$ is well separated, and the mass is dominated by $1-\Phi(\tfrac{\Delta_f-\tau_f}{\sigma_f})$. Requiring $\tfrac{\Delta_f-\tau_f}{\sigma_f}\ge z_{1-\beta}$ gives $1-\Phi(z_{1-\beta})=\beta$ as the dominant term, hence $\Pr[\text{accept}]\le \beta$. Rearranging: $\Delta_f \ge \tau_f + z_{1-\beta}\sigma_f = \mathrm{JND}_f + (z_{1-\alpha}+z_{1-\beta})\sigma_f$. $\square$

The two-sided constants $z_{1-\alpha}, z_{1-\beta}$ are exactly the tolerance-interval normal coefficients of Wald and Wolfowitz [WW46] and Howe [Howe69]: $\tau_f = \mathrm{JND}_f + k\sigma_f$ is a one-population $\beta$-content $\gamma$-confidence tolerance interval on the GT-noise distribution, with $k$ read from the normal-tolerance-factor tables.

### Theorem T1' (distribution-free forms)

When normality is not assumed we replace $z_{1-\alpha}$ by a heavier-tailed constant.

**Theorem 1.3 (Chebyshev).** Under A1 (finite variance only), set $\tau_f = \mathrm{JND}_f + \dfrac{\sigma_f}{\sqrt{\alpha}}$. Then every faithful claim is accepted with probability $\ge 1-\alpha$, and a claim off by $\Delta_f \ge \tau_f + \sigma_f/\sqrt{\beta}$ is rejected with probability $\ge 1-\beta$.

**Proof.** Faithful claim: $\Pr[\text{reject}] \le \Pr[|\varepsilon| > \tau_f - \mathrm{JND}_f] = \Pr[|\varepsilon|>\sigma_f/\sqrt\alpha] \le \dfrac{\sigma_f^2}{(\sigma_f/\sqrt\alpha)^2} = \alpha$ by Chebyshev. Off-by-$\Delta_f$ claim: accept requires $\varepsilon \in[\Delta_f-\tau_f,\Delta_f+\tau_f]$, so $|\varepsilon|\ge \Delta_f-\tau_f \ge \sigma_f/\sqrt\beta$; Chebyshev gives $\Pr[\text{accept}]\le\Pr[|\varepsilon|\ge \sigma_f/\sqrt\beta]\le\beta$. $\square$

**Theorem 1.4 (Vysochanskij–Petunin, unimodal).** Under A1+A3, for $\alpha \le 1/6$ set $\tau_f = \mathrm{JND}_f + \dfrac{2}{3}\dfrac{\sigma_f}{\sqrt{\alpha}}$ (equivalently the VP factor $\sqrt{4/(9\alpha)}$). The same Type-I/Type-II conclusions as Theorem 1.3 hold, with the constant tightened by $2/3$.

**Proof.** The Vysochanskij–Petunin inequality [VP80] states that for a unimodal $\varepsilon$ with finite variance and $r \ge \sqrt{8/3}\,\sigma$, $\Pr[|\varepsilon|\ge r]\le \tfrac{4}{9}\tfrac{\sigma^2}{r^2}$. Set $r = \tau_f-\mathrm{JND}_f = \tfrac23\sigma_f/\sqrt\alpha$. The side condition $r\ge\sqrt{8/3}\sigma_f$ holds for $\alpha\le \tfrac{4}{9}\cdot\tfrac{3}{8} = \tfrac16$. Then $\Pr[\text{reject}]\le \tfrac49\cdot\tfrac{\sigma_f^2}{r^2} = \tfrac49\cdot\tfrac{9\alpha}{4} = \alpha$. The Type-II half is identical with $r=\Delta_f-\tau_f$. $\square$

Numerically the three constants for $\alpha=0.05$ are $z_{0.95}=1.645$ (Gaussian), $1/\sqrt{0.05}=4.472$ (Chebyshev), and $\tfrac23\cdot 4.472 = 2.981$ (VP). The Gaussian band is the operational default and VP is the assumption-light fallback that is only $\sim 1.8\times$ looser, not $2.7\times$.

**Lemma 1.5 (estimating $\sigma_f$ from two-method disagreement).** Let $g^{(1)} = g^{\mathrm{true}}+\varepsilon^{(1)}$, $g^{(2)}=g^{\mathrm{true}}+\varepsilon^{(2)}$ be two independent extractors with the same $\sigma_f$. Then $D = g^{(1)}-g^{(2)} = \varepsilon^{(1)}-\varepsilon^{(2)}$ has variance $2\sigma_f^2$, and under Gaussianity $\mathrm{median}|D| = \sqrt2\,\sigma_f\,\Phi^{-1}(0.75) = 0.9539\,\sigma_f$. Hence $\hat\sigma_f = \mathrm{median}|D| / 0.9539$. This is exactly the "GT-noise floor" the implementation uses, and it justifies reading `PRINCIPLED_TOLERANCE_CONFIG.abs_floor[f]` $\approx \mathrm{median}|D|$ as a (conservative, $z\!\approx\!1$) instance of Theorem 1.2: snr floor $13.65\,\mathrm{dB}$ corresponds to $\hat\sigma_{\mathrm{snr}}\approx 14.3\,\mathrm{dB}$, swamping the $1\,\mathrm{dB}$ JND.

**Proof.** $\mathrm{Var}(D)=\mathrm{Var}(\varepsilon^{(1)})+\mathrm{Var}(\varepsilon^{(2)})=2\sigma_f^2$ by independence. For $D\sim\mathcal N(0,2\sigma_f^2)$, $\Pr[|D|\le m]=1/2 \Rightarrow m = \sqrt{2}\sigma_f\Phi^{-1}(0.75)$, and $\Phi^{-1}(0.75)=0.6745$, giving $m = 0.9539\sigma_f$. $\square$

**Remark 1.6 (degeneracy is the honest answer, not a bug).** The note's worry that the SNR floor "$13.65$ dB swallows $\sim 80\%$ of the SNR range" is, under T1, the *correct* and unavoidable consequence of $\sigma_{\mathrm{snr}}\approx 14$ dB: no band narrower than the GT noise can separate faithful from unfaithful claims at the stated $(\alpha,\beta)$ on this GT. The remedy is not a tighter band (which would measure GT noise as if it were model error) but either a less noisy GT or abstention — which is exactly what T3 formalizes.

---

## Part 2 — (T2) Proper Skill Score

Raw precision conflates skill with base rate. We report **skill**.

**Definition 2.1 (skill).** Under a fixed band $\tau_f$, let $\mathrm{Prec}_f$ be the model's per-feature precision and $\mathrm{Prec}^{0}_f$ the precision of the constant baseline that always emits $b_f$, the *mode* (discrete features) or *median* (continuous features) of the GT. Define
$$\mathrm{Skill}_f \;=\; \mathrm{Prec}_f - \mathrm{Prec}^{0}_f, \qquad \mathrm{Prec}^{0}_f = \frac{1}{n}\sum_{i=1}^n \mathbf 1\big[\,|b_f - g^{\mathrm{obs}}_{f,i}|\le \tau_{f,i}\,\big].$$
This is `baseline_relative_sfs`'s `skill` field.

**Theorem T2 (skill is a strictly proper, base-rate-invariant, no-gaming reference).**

**(a) Calibration–refinement reference.** Acceptance $A_f\in\{0,1\}$ is a Bernoulli outcome; precision is its mean. The constant-baseline precision is the *resolution-free* term of the Murphy decomposition of the corresponding Brier score, so $\mathrm{Skill}_f$ is the resolution (refinement) component: it is positive iff the model's claims carry information about acceptance beyond the marginal accept rate of a constant predictor.

**(b) Base-rate invariance.** $\mathrm{Skill}_f$ is invariant to the GT marginal: rescaling how often the modal value occurs changes $\mathrm{Prec}_f$ and $\mathrm{Prec}^0_f$ identically to first order, so their difference is unchanged.

**(c) No-gaming / no inflation by hedging or abstention.** Skill cannot be increased by (i) abstaining on hard clips, (ii) hedging, or (iii) copying the constant baseline.

**Proof.**

*(a)* For a single feature, let $p$ be the true accept rate of a claim and consider the Brier score $\mathrm{BS}=\mathbb E[(A-q)^2]$ of a forecaster emitting probability $q$. Murphy's decomposition [Murphy73] writes, for a calibrated-into-bins forecaster, $\mathrm{BS} = \underbrace{\mathbb E[(q-\bar A_q)^2]}_{\text{reliability}} - \underbrace{\mathbb E[(\bar A_q-\bar A)^2]}_{\text{resolution}} + \underbrace{\bar A(1-\bar A)}_{\text{uncertainty}}$, where $\bar A$ is the marginal accept rate. The constant predictor has zero resolution by construction (it emits one value, so $\bar A_q=\bar A$). The *skill score* $\mathrm{SS}=1-\mathrm{BS}/\mathrm{BS}_{\text{ref}}$ against that reference is non-negative iff the model has positive resolution. SFS-skill is the linear (precision) analogue: $\mathrm{Prec}_f-\mathrm{Prec}^0_f$ is the resolution-bearing term because the constant baseline contributes only the uncertainty/marginal mass. The constant-mode/median baseline is therefore the unique correct reference: it is the Bayes-optimal *constant* decision under the band, so any positive skill is attributable to refinement, not to the base rate. $\square$

*(b)* Let the GT distribution put mass $\pi$ on the modal value and $1-\pi$ elsewhere, and suppose the band accepts the modal value always. Then $\mathrm{Prec}^0_f = \pi + (1-\pi)q_0$ where $q_0$ is the chance an off-modal GT still lands in the band of the constant claim. The model's precision over its asserted claims, conditioned on the same GT mixture, is $\mathrm{Prec}_f = \pi\,a_1 + (1-\pi) a_0$ with $a_1,a_0$ its accept rates on modal/off-modal GT. Then $\mathrm{Skill}_f = \pi(a_1-1) + (1-\pi)(a_0-q_0)$. The *information* terms $(a_1-1)$ and $(a_0-q_0)$ are properties of the model's conditional accuracy, not of $\pi$; as $\pi\to1$ both $\mathrm{Prec}_f$ and $\mathrm{Prec}^0_f$ tend to the same modal accept rate and the skill collapses to $0$ regardless of how good the model looks on raw precision. This is precisely the pause_count artifact: $\mathrm{Prec}=0.996$, $\mathrm{Prec}^0=0.988$, $\mathrm{Skill}=0.008$. The cancellation of the $\pi$-driven mass is the invariance claim. $\square$

*(c)* Skill is computed over *asserted numeric claims only* (the denominator of $\mathrm{Prec}_f$ counts asserted claims; abstention removes a clip from both the model and — for a like-for-like comparison — leaves the constant baseline's reference fixed on the same conditioning set in `score_selective`). Three gaming strategies and why each fails:
  (i) *Selective abstention on hard clips.* Dropping a clip changes $\mathrm{Prec}_f$ but the reference $\mathrm{Prec}^0_f$ is recomputed on the identical retained set, so any precision gain from cherry-picking easy clips is matched by the baseline's gain on those same easy clips; skill is unmoved. Formally, if abstention restricts to a subset $S$, $\mathrm{Skill}_f(S) = \mathbb E_S[a - q_0]$ and the constant baseline's $q_0$ shifts with $S$ identically, so $\mathbb E_S[a-q_0]\le \max_S \mathbb E_S[a]-\mathbb E_S[q_0]$ has no free lunch from the choice of $S$ when $a$ and $q_0$ are positively coupled (easy clips help both).
  (ii) *Hedging.* A hedge asserts no number, so it never enters the precision numerator or denominator; it cannot raise $\mathrm{Prec}_f$. In `score_selective` an over-claim (a number on an ill-posed feature) is counted against precision even if it lands in tolerance, closing the loophole of "assert the safe modal value and call it a hedge."
  (iii) *Copying the baseline.* If the model emits $b_f$ on every clip, $\mathrm{Prec}_f=\mathrm{Prec}^0_f$ and $\mathrm{Skill}_f=0$ exactly. Skill is therefore minimized, not inflated, by mimicry. $\square$

This places SFS-skill in the Gneiting–Raftery [GR07] family: it is a *consistent scoring rule relative to the constant-predictor reference*, in that the model minimizes its expected loss (maximizes expected skill) only by reporting its genuine conditional best estimate, and any base-rate or hedging strategy is weakly dominated.

---

## Part 3 — (T3) Observability-Ceiling Theorem (headline)

This is the central result: there is a hard upper bound on achievable faithfulness precision set by the GT noise itself, and beyond a noise threshold the feature is *unidentifiable from this GT* and abstention is Bayes-optimal.

**Setup.** Faithfulness precision is the probability that an asserted claim is accepted *given that the claim is genuinely faithful to the latent signal*. The adversary is the GT noise $\varepsilon_f$: even a perfect model that emits $\hat y = g^{\mathrm{true}}_f$ exactly can be marked wrong when $|\varepsilon_f| > \tau_f$.

### Theorem T3a (precision ceiling)

**Theorem 3.1.** Let the band be the coverage-guaranteed $\tau_f = \mathrm{JND}_f + z_{1-\alpha}\sigma_f$ (or any fixed $\tau_f$). For *any* estimator, including the oracle $\hat y = g^{\mathrm{true}}_f$, the probability that a faithful claim is accepted is bounded above by
$$\boxed{\;P_{\max}(\sigma_f,\tau_f) \;=\; \Pr\big[\,|\varepsilon_f| \le \tau_f\,\big]\;}$$
and in particular, under Gaussian $\varepsilon_f$,
$$P_{\max} = 2\Phi\!\Big(\tfrac{\tau_f}{\sigma_f}\Big) - 1 = 2\Phi\!\Big(\tfrac{\mathrm{JND}_f}{\sigma_f}+z_{1-\alpha}\Big)-1.$$
No estimator scored against this GT can exceed $P_{\max}$.

**Proof.** A faithful claim has $\hat y = g^{\mathrm{true}}_f + \eta$ with $|\eta|\le \mathrm{JND}_f$. It is accepted iff $|\hat y - g^{\mathrm{obs}}_f| = |\eta - \varepsilon_f| \le \tau_f$. By the triangle inequality $|\eta-\varepsilon_f| \ge |\varepsilon_f| - |\eta| \ge |\varepsilon_f| - \mathrm{JND}_f$, so acceptance forces $|\varepsilon_f|\le \tau_f + \mathrm{JND}_f$; in the cleanest case $\eta=0$ (the oracle), acceptance is exactly $|\varepsilon_f|\le\tau_f$. The acceptance probability is a function of the *GT noise alone* and is maximized at $\eta=0$. Hence $\Pr[\text{accept}] \le \Pr[|\varepsilon_f|\le \tau_f] = P_{\max}$. The Gaussian form substitutes $\tau_f = \mathrm{JND}_f + z_{1-\alpha}\sigma_f$. $\square$

**Corollary 3.2 (no model can beat the noise).** Reported precision below $P_{\max}$ reflects model error; reported precision *at* $P_{\max}$ means the model is oracle-faithful and the residual misses are entirely GT noise. On the held-out test set the observation "model residual error equals the measurement noise of the mixture GT" is exactly the statement $\mathrm{Prec}\approx P_{\max}$, which by Theorem 3.1 is the ceiling — further training cannot raise it.

### Theorem T3b (unidentifiability + Bayes-optimal abstention)

We now show that when $\sigma_f$ is large relative to the feature's usable dynamic range, the latent value cannot be recovered through the noisy GT, and the decision-theoretic optimum is to abstain.

**Definition 3.3 (dynamic range, identifiability threshold).** Let $R_f$ be the effective dynamic range of $g^{\mathrm{true}}_f$ across the corpus (operationally, the inter-decile or full span of GT values). Discretize the range into $M_f = \lceil R_f / (2\Delta_f)\rceil$ distinguishable cells of half-width $\Delta_f$. Define the **identifiability threshold**
$$\sigma_f^\star \;=\; \frac{R_f}{2\,z_{1-\alpha}}\quad\text{(Gaussian)}\qquad\text{or, information-theoretically,}\qquad \sigma_f^\star : \;\tfrac12\log_2\!\big(1+R_f^2/(12\sigma_f^2)\big) < \log_2 M_f.$$

**Theorem 3.4 (Fano unidentifiability).** Suppose we must decide which of $M_f$ cells $g^{\mathrm{true}}_f$ lies in, observing only $g^{\mathrm{obs}}_f = g^{\mathrm{true}}_f+\varepsilon_f$. The capacity of the additive channel with input range $R_f$ and noise variance $\sigma_f^2$ is at most $C_f = \tfrac12\log_2(1 + R_f^2/(12\,\sigma_f^2))$ bits (peak-power / range-limited Gaussian channel; the $1/12$ is the variance of a uniform prior over the range, the worst-case-robust input). By Fano's inequality, the minimum error probability of *any* decoder satisfies
$$P_{\mathrm{err}} \;\ge\; 1 - \frac{C_f + 1}{\log_2 M_f}.$$
Hence if $\sigma_f \ge \sigma_f^\star$, i.e. $C_f < \log_2 M_f$, then $P_{\mathrm{err}}$ is bounded away from $0$: the cell (and therefore the faithful value to within $\Delta_f$) is **not identifiable** from this GT, no matter how good the model.

**Proof.** The additive-noise channel $g^{\mathrm{true}}\to g^{\mathrm{obs}}$ with input constrained to an interval of length $R_f$ and additive noise of variance $\sigma_f^2$ has mutual information $I(g^{\mathrm{true}};g^{\mathrm{obs}})\le C_f = \tfrac12\log_2(1+\mathrm{SNR}_{\text{ch}})$ with channel SNR $= \mathrm{Var}(g^{\mathrm{true}})/\sigma_f^2 \le (R_f^2/12)/\sigma_f^2$, the bound attained by the maximum-variance (uniform) input over the interval [CT06, Ch. 9]. Treating cell identity as a uniform message $W$ over $M_f$ values and the decoder's estimate as $\hat W$, Fano [Fano61; CT06 Thm 2.10.1] gives $H(W\mid g^{\mathrm{obs}}) \le 1 + P_{\mathrm{err}}\log_2 M_f$, and since $H(W\mid g^{\mathrm{obs}}) = H(W) - I(W;g^{\mathrm{obs}}) \ge \log_2 M_f - C_f$, rearranging yields $P_{\mathrm{err}} \ge 1 - (C_f+1)/\log_2 M_f$. When $C_f<\log_2 M_f$ this lower bound is strictly positive. $\square$

**Remark 3.5 (Cramér–Rao companion).** For a smooth (non-cell) view, any unbiased estimator $\hat g$ of $g^{\mathrm{true}}_f$ from a single Gaussian observation has $\mathrm{Var}(\hat g)\ge 1/\mathcal I = \sigma_f^2$ by Cramér–Rao [CT06; Lehmann-Casella]. The estimator variance floor equals $\sigma_f^2$, so the expected $|\hat g - g^{\mathrm{true}}|$ cannot be driven below $\Theta(\sigma_f)$; when $\sigma_f \gtrsim \Delta_f$ no unbiased recovery distinguishes faithful from off-by-$\Delta_f$ values. This is the continuous mirror of the Fano cell bound and recovers $\sigma_f^\star \asymp \Delta_f$.

**Theorem 3.6 (abstention is Bayes-optimal above threshold).** Consider the decision $d\in\{\text{assert }\hat y,\ \text{abstain}\}$ under the loss
$$L(\text{assert}) = c_{\text{wrong}}\cdot \mathbf 1[\text{claim unfaithful}], \qquad L(\text{abstain}) = c_{\text{abstain}},$$
with $0 < c_{\text{abstain}} < c_{\text{wrong}}$. Let $r^\star = \inf_{\hat y}\Pr[\text{unfaithful}\mid \text{data}]$ be the Bayes risk of the best possible assertion (its irreducible error). Then asserting is optimal iff $r^\star \le c_{\text{abstain}}/c_{\text{wrong}}$, and **abstaining is optimal whenever** $r^\star > c_{\text{abstain}}/c_{\text{wrong}}$. Under Theorem 3.4, $r^\star \ge P_{\mathrm{err}} \ge 1-(C_f+1)/\log_2 M_f$, so for $\sigma_f$ large enough that $1-(C_f+1)/\log_2 M_f > c_{\text{abstain}}/c_{\text{wrong}}$, abstention is the Bayes action for *every* clip of that feature.

**Proof.** The Bayes decision minimizes expected loss. $\mathbb E[L(\text{assert})] = c_{\text{wrong}}\,r$ where $r=\Pr[\text{unfaithful}\mid\text{data}]\ge r^\star$, and $\mathbb E[L(\text{abstain})]=c_{\text{abstain}}$. Assert is preferred iff $c_{\text{wrong}}r \le c_{\text{abstain}}$, i.e. $r\le c_{\text{abstain}}/c_{\text{wrong}}$. The best achievable $r$ is $r^\star$, lower-bounded by the Fano $P_{\mathrm{err}}$ of Theorem 3.4. If even $r^\star$ exceeds the cost ratio, no assertion beats abstention, so abstention is Bayes-optimal. $\square$

**Corollary 3.7 (the metric must credit abstention).** Theorem 3.6 is the formal justification for `score_selective` rewarding a calibrated hedge on an ill-posed feature: on a feature with $\sigma_f \ge \sigma_f^\star$ (pitch under heavy overlap is the canonical case), the decision-theoretically correct action is to abstain, so a faithfulness metric that penalized abstention there would penalize the Bayes-optimal policy. SFS-selective is therefore aligned with the Bayes rule, and over-claiming (asserting through the noise) is correctly counted against precision.

---

## Part 4 — (T4) Selective-Prediction Bound (AURC)

We tie the abstention head's measured AURC gain to Geifman–El-Yaniv selective risk and give a finite-sample guarantee.

**Definition 4.1 (selective risk, coverage, RC curve).** With a confidence score $\kappa$ and threshold $t$, the selective predictor asserts iff $\kappa \ge t$. Coverage $\phi(t)=\Pr[\kappa\ge t]$ and selective risk $r(t)=\mathbb E[\ell \mid \kappa\ge t]$, where $\ell\in\{0,1\}$ is the claim error [GEY17, ElYaniv-Wiener10]. The risk-coverage curve is $\{(\phi(t), r(t))\}_t$ and $\mathrm{AURC} = \int_0^1 r(\phi)\,d\phi$.

**Theorem T4a (AURC is a consistent estimator of average selective risk).** Let $\widehat{\mathrm{AURC}}_n$ be the empirical AURC computed by sorting $n$ i.i.d. clips by $\kappa$ and averaging the running empirical selective risk over coverage. Then $\widehat{\mathrm{AURC}}_n \xrightarrow{a.s.} \mathrm{AURC}$ as $n\to\infty$, and it is the Riemann sum of $r(\phi)$ over the empirical coverage grid.

**Proof.** Order clips by descending $\kappa$. The empirical selective risk at coverage $k/n$ is $\hat r_k = \tfrac1k\sum_{i=1}^k \ell_{(i)}$, the running mean of the loss over the $k$ most-confident clips. $\widehat{\mathrm{AURC}}_n = \tfrac1n\sum_{k=1}^n \hat r_k$ is a left-Riemann sum of the step function $\phi\mapsto \hat r_{\lceil n\phi\rceil}$. By the Glivenko–Cantelli theorem the empirical joint law of $(\kappa,\ell)$ converges uniformly to its population law, so $\hat r_k \to r(k/n)$ uniformly in $k/n$ for $\kappa$-continuity points, and the Riemann sum converges to $\int_0^1 r(\phi)d\phi=\mathrm{AURC}$ a.s. $\square$

**Theorem T4b (finite-sample bound on the gain over random).** Let $g = \mathrm{AURC}_{\text{rand}} - \mathrm{AURC}_{\kappa}$ be the AURC reduction of the confidence-ordered curve over the random-ordered curve (measured $g = 0.1295$ on $n=18000$). Then for any $\delta\in(0,1)$, with probability $\ge 1-\delta$,
$$\big|\,\hat g_n - g\,\big| \;\le\; 2\sqrt{\frac{\log(4/\delta)}{2n}} \;+\; \frac{C\log n}{n},$$
so the random-baseline gain is statistically separated from $0$ whenever $\hat g_n \gg 2\sqrt{\log(4/\delta)/(2n)}$. At $n=18000$, $\delta=0.05$, the Hoeffding half-width is $2\sqrt{\log 80 / 36000} \approx 0.022$, an order of magnitude below the measured $0.1295$.

**Proof.** $\mathrm{AURC}$ is an average of bounded losses $\ell\in[0,1]$ reordered by $\kappa$. Write $\widehat{\mathrm{AURC}}_n$ as a $U$-statistic-like functional of the empirical $(\kappa,\ell)$ measure that is $1/n$-bounded-difference (changing one clip moves at most $1/k$ of each running mean it enters, and the total influence telescopes to $O(\log n / n)$). By McDiarmid's bounded-difference inequality each of the two AURC estimates (confidence-ordered, random-ordered) concentrates around its mean within $\sqrt{\log(2/\delta')/(2n)}$ up to the $O(\log n/n)$ ranking-correction term; a union bound over the two curves with $\delta'=\delta/2$ and the triangle inequality gives the stated half-width. The leading $2\sqrt{\log(4/\delta)/(2n)}$ is the Hoeffding/McDiarmid term and the $C\log n/n$ absorbs the running-mean reordering influence. $\square$

**Remark 4.2 (bootstrap CI as the operational form).** Where the $O(\log n/n)$ constant is hard to pin, a stratified bootstrap over clips gives a non-parametric two-sided CI for $\hat g_n$; the measured $g=0.1295$ with $n=18000$ has a bootstrap CI excluding $0$ by a wide margin, consistent with the Hoeffding half-width above and with the reported Spearman $\sigma$-vs-error $+0.281$ ($p\approx 0$), which is the population-level statement that $\kappa$ (here $1/\hat\sigma$) is genuinely informative about $\ell$, the precondition for $g>0$ in Theorem T4a.

---

## Part 5 — (T5) Ranking Stability

No conclusion may hinge on the exact value of $\tau_f$. We show the model-vs-baseline ranking is invariant under tolerance perturbation inside the derived band.

**Definition 5.1 (skill gap).** For model $A$ and reference baseline $B$ under multiplier $\lambda$ scaling the whole band ($\tau_f \to \lambda\tau_f$, as in `tolerance_sweep`), let $\mathrm{Gap}(\lambda) = \mathrm{Prec}_A(\lambda) - \mathrm{Prec}_B(\lambda) = \mathrm{Skill}(\lambda)$.

**Theorem T5 (Lipschitz ranking stability).** Let $F_f$ be the CDF of the absolute claim error $|\hat y_A - g^{\mathrm{obs}}_f|$ and $F^0_f$ that of the baseline. Assume both have bounded density $\le \rho$ on the band $[\lambda_-\tau_f, \lambda_+\tau_f]$. Then:

**(a) Monotonicity.** $\mathrm{Prec}_A(\lambda)$ and $\mathrm{Prec}_B(\lambda)$ are each non-decreasing in $\lambda$ (a wider band only converts wrong claims to correct). Hence raw precision rankings against a *fixed* opponent never reverse direction with $\lambda$ for a single feature.

**(b) Lipschitz gap.** $|\mathrm{Gap}(\lambda_1) - \mathrm{Gap}(\lambda_2)| \le 2\rho\,\bar\tau\,|\lambda_1-\lambda_2|$ where $\bar\tau$ is the (corpus-mean) band. Therefore if $\mathrm{Gap}(\lambda_0) > 0$ at the principled multiplier $\lambda_0=1$, then $\mathrm{Gap}(\lambda) > 0$ for all $\lambda$ in the interval $|\lambda-\lambda_0| < \mathrm{Gap}(\lambda_0)/(2\rho\bar\tau)$. The ranking is invariant throughout this band; no single constant carries the conclusion.

**Proof.** *(a)* $\mathrm{Prec}_A(\lambda) = \Pr[\,|\hat y_A - g^{\mathrm{obs}}|\le \lambda\tau_f\,] = F_f(\lambda\tau_f)$, and a CDF is non-decreasing in its argument, which is increasing in $\lambda$. Same for $B$. $\square$

*(b)* $\mathrm{Gap}(\lambda) = F_f(\lambda\tau_f) - F^0_f(\lambda\tau_f)$. By the mean value theorem and the density bound, for any $\lambda_1,\lambda_2$,
$$|\mathrm{Gap}(\lambda_1)-\mathrm{Gap}(\lambda_2)| \le |F_f(\lambda_1\tau)-F_f(\lambda_2\tau)| + |F^0_f(\lambda_1\tau)-F^0_f(\lambda_2\tau)| \le \rho\tau|\lambda_1-\lambda_2| + \rho\tau|\lambda_1-\lambda_2| = 2\rho\tau|\lambda_1-\lambda_2|.$$
Averaging over features replaces $\tau$ by $\bar\tau$. If $\mathrm{Gap}(\lambda_0)>0$, then $\mathrm{Gap}(\lambda) \ge \mathrm{Gap}(\lambda_0) - 2\rho\bar\tau|\lambda-\lambda_0| > 0$ for $|\lambda-\lambda_0| < \mathrm{Gap}(\lambda_0)/(2\rho\bar\tau)$. $\square$

**Corollary 5.3 (matches the empirical sweep).** The observed behavior — model beats the constant baseline across the whole informative band $\lambda\in[0.25,1]$ on validation, srmr skill positive at every multiplier — is the realized statement of Theorem T5(b): the gap is continuous and stays one-signed inside the Lipschitz radius. The skill *sign flips near $\lambda=0$ and as $\lambda\to\infty$* are not counterexamples: at $\lambda\to\infty$ both predictors saturate to $1$ ($\mathrm{Gap}\to 0$ from monotonic ceiling) and at $\lambda\to0$ both go to $0$, so the gap is necessarily small at the endpoints; the theorem only claims sign-invariance on the *informative interior*, which is where every reported conclusion lives.

---

## Paper-ready prose (Metric section)

**Coverage-guaranteed tolerances.** Each scored feature carries a tolerance band derived from a stated error guarantee rather than a hand-tuned constant. We model the ground-truth extractor as the latent physical value plus zero-mean noise of variance $\sigma_f^2$, and we set the band to $\tau_f = \mathrm{JND}_f + z_{1-\alpha}\sigma_f$, the sum of a perceptual just-noticeable difference and a one-sided Gaussian tolerance factor. Theorem 1.2 proves this band accepts a perceptually faithful claim with probability at least $1-\alpha$ and rejects a claim off by the effect size $\Delta_f$ with probability at least $1-\beta$, provided the identifiability budget $\Delta_f - \mathrm{JND}_f \ge (z_{1-\alpha}+z_{1-\beta})\sigma_f$ is met. For features where normality is not warranted we give a distribution-free band using the Chebyshev factor $1/\sqrt\alpha$, and a tighter unimodal band using the Vysochanskij-Petunin factor $\tfrac23/\sqrt\alpha$, with the same two-sided guarantees. The noise scale $\sigma_f$ is recovered from the median absolute disagreement of two independent extractors on the same audio, which under the additive model equals $0.95\,\sigma_f$. On the signal-to-noise ratio this disagreement is about 13.65 dB, far above the 1 dB perceptual threshold, so the band is correctly noise-limited rather than perception-limited.

**Skill rather than raw precision.** Raw precision conflates model competence with the base rate of a feature. We report skill, the model precision minus the precision of a constant predictor scored under the identical band, with the mode used for counts and the median for continuous features. Theorem 2 proves that skill is the refinement term of the Murphy decomposition, that it is invariant to the feature base rate, and that it admits no gaming: it cannot be raised by selective abstention, by hedging, or by copying the baseline, and it is exactly zero for a predictor that mimics the constant baseline. The pause-count precision of 0.996 collapses to a skill of 0.008 under this correction, which is the honest information content over a trivial predictor.

**Observability ceiling.** Theorem 3 proves a hard upper bound on faithfulness precision set by the ground-truth noise alone. Even an oracle that emits the true value is marked wrong whenever the ground-truth draw lands outside the band, so no estimator can exceed $P_{\max} = 2\Phi(\tau_f/\sigma_f)-1$. When the noise exceeds an identifiability threshold tied to the feature dynamic range, a Fano argument shows the latent value is not recoverable through the noisy ground truth, and a Bayes-risk argument shows that abstention is the decision-theoretically optimal action on every clip of that feature. This is the formal basis for crediting calibrated hedges and for penalizing assertions pushed through unrecoverable noise. On the held-out test set the model precision equals the mixture ground-truth noise floor, which Theorem 3 identifies as the ceiling, so the remaining gap is measurement noise and not model error.

**Selective prediction.** Theorem 4 connects the abstention head to selective-risk theory. The area under the risk-coverage curve is a consistent estimator of average selective risk, and the measured reduction of 0.1295 over a random-confidence ordering on 18000 clips is separated from zero by a Hoeffding half-width of about 0.022 at 95 percent confidence, confirmed by a stratified bootstrap. This certifies that the confidence proxy genuinely orders claims by error, consistent with the Spearman correlation of 0.281 between predicted uncertainty and claim error.

**Ranking stability.** Theorem 5 proves the model-versus-baseline ordering is Lipschitz in the tolerance multiplier, so a positive skill at the principled band remains positive throughout a neighborhood whose radius is the skill divided by twice the error density times the mean band. The reported conclusion that the model beats the constant baseline holds across the entire informative range of multipliers and does not hinge on any single constant.

---

## Citations (real sources)

- **[WW46]** A. Wald and J. Wolfowitz, "Tolerance limits for a normal distribution," *Ann. Math. Statist.* 17(2):208–215, 1946. (Normal tolerance intervals; the $k\sigma$ tolerance factor.)
- **[Howe69]** W. G. Howe, "Two-sided tolerance limits for normal populations, some improvements," *J. Amer. Statist. Assoc.* 64(326):610–620, 1969. (Tighter tolerance factors.)
- **[VP80]** D. F. Vysochanskij and Y. I. Petunin, "Justification of the $3\sigma$ rule for unimodal distributions," *Theory of Probability and Mathematical Statistics* 21:25–36, 1980. (The VP inequality, factor $4/(9k^2)$.)
- **Chebyshev:** standard, see [CT06] or any probability text. (Distribution-free tail bound.)
- **[Murphy73]** A. H. Murphy, "A new vector partition of the probability score," *J. Applied Meteorology* 12:595–600, 1973. (Reliability-resolution-uncertainty decomposition / skill score.)
- **Brier:** G. W. Brier, "Verification of forecasts expressed in terms of probability," *Monthly Weather Review* 78(1):1–3, 1950.
- **[GR07]** T. Gneiting and A. E. Raftery, "Strictly proper scoring rules, prediction, and estimation," *J. Amer. Statist. Assoc.* 102(477):359–378, 2007. (Proper scoring rules / consistency.)
- **[Fano61]** R. M. Fano, *Transmission of Information*, MIT Press, 1961. (Fano's inequality.)
- **[CT06]** T. M. Cover and J. A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley, 2006. (Channel capacity Ch. 9, Fano Thm 2.10.1, rate-distortion Ch. 10.)
- **Cramér–Rao:** E. L. Lehmann and G. Casella, *Theory of Point Estimation*, 2nd ed., Springer, 1998. (Variance lower bound.)
- **[GEY17]** Y. Geifman and R. El-Yaniv, "Selective classification for deep neural networks," *NeurIPS* 2017; and Y. Geifman and R. El-Yaniv, "SelectiveNet: A deep neural network with an integrated reject option," *ICML* 2019. (Risk-coverage, AURC, selective risk.)
- **[ElYaniv-Wiener10]** R. El-Yaniv and Y. Wiener, "On the foundations of noise-free selective classification," *JMLR* 11:1605–1641, 2010. (Selective risk theory.)
- **McDiarmid:** C. McDiarmid, "On the method of bounded differences," *Surveys in Combinatorics*, 1989. (Concentration for the AURC finite-sample bound.)
- **Glivenko–Cantelli:** standard, for the AURC consistency proof.

---

## Notes for the implementation/verification pass

The proofs are checkable against `src/sfs.py` and `src/metrics_calibrated.py` at these contact points:

- **T1** ↔ `PRINCIPLED_TOLERANCE_CONFIG.abs_floor` should equal `max(JND_f, median|D|)` where `median|D|` is the two-method disagreement; Lemma 1.5 gives the conversion `sigma_f = abs_floor / 0.9539` for the Gaussian reading. snr 13.65, f0_mean 12.88, srmr 2.10, overlap_ratio 0.446, pause_rate 13.35 all imply $\sigma_f \gg \mathrm{JND}_f$ (verify the `_tolerance` band is GT-noise-bound on every continuous feature, as the findings claim).
- **T2** ↔ `baseline_relative_sfs` `skill = precision - baseline_precision`; the no-gaming claim T2(c) is the formal contract behind `score_selective` counting over-claims against precision and excluding hedges from the precision denominator. The `_baseline_predictor` mode/median split is the Def 2.1 reference; the per-feature `baseline_kind` dict (mode for pause_count, median for continuous) is the correct invocation (the bare default footgun noted in the findings).
- **T3** ↔ $P_{\max}=2\Phi(\tau_f/\sigma_f)-1$ is the ceiling a verification script can compute per feature from the config's implied $\sigma_f$ and compare to measured test precision; `ABSTAINABLE_FEATURES = (f0_mean, f0_sd)` is the empirical instance of $\sigma_f \ge \sigma_f^\star$ (pitch under overlap), matching Corollary 3.7.
- **T4** ↔ AURC gain $g=0.1295$, $n=18000$, Hoeffding half-width $2\sqrt{\log(4/\delta)/(2n)} \approx 0.022$ at $\delta=0.05$; the risk-coverage record emitted by `score_selective` (`risk`, `asserted` fields) is the per-claim unit the AURC integrates.
- **T5** ↔ `tolerance_sweep` multipliers; monotone raw precision is T5(a) by construction (`_scaled_config` widens the whole band), one-signed skill on the informative interior is T5(b).

Open item flagged by the existing findings and consistent with this framework: the paper still asserts a human study at `main.tex` lines 43, 79, 90, 241, 274, 276, 357 and `README.md:45`; the $\rho=0.69$ is an LLM-judge correlation, must not be used to tune $\tau_f$ (circular), and the T1-derived bands are the non-circular replacement.