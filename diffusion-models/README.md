# Diffusion Models

**Level:** Advanced  ·  **Topic:** Generative Models

▶️ **Watch the video:** https://www.youtube.com/@mlzerotoherousa

> Destroy data by adding Gaussian noise over many small steps, then train a neural network to undo one step at a time; generating = starting from pure noise and denoising repeatedly. The training objective reduces to "predict the noise that was added" (a simple MSE), and the same model is simultaneously a score estimator and a velocity field for an ODE, which is why diffusion, score matching and flow matching are one family.

## The intuition
Think of a sandcastle on a windy beach. Every second the wind blows a little sand off, until after an hour there is just a flat, random pile. Watching that is easy. The magic trick would be to film it and play the video **backwards**: a random pile slowly turns into a castle.

A diffusion model learns to play that video backwards. We take real photos and add a tiny bit of random "static" (like TV snow), then a bit more, a thousand times, until the photo is pure static. That direction needs no learning at all. Then we train a neural network on one small job: *given a slightly-staticky picture, guess which static was added.* If you can guess the static, you can subtract it and get a slightly cleaner picture.

Tiny example with one pixel. The true brightness is 0.8. We add noise and get 0.8 + 0.3 = 1.1. The network sees 1.1 and guesses the noise was 0.25. Subtract it: 0.85, close to 0.8. The network is graded by how far its guess (0.25) was from the real noise (0.3): error = 0.05² = 0.0025. Do that billions of times on millions of photos and it gets very good at guessing.

To make a brand-new picture, start with pure static that never came from any photo, and ask the network "clean this up a little" a few dozen to a thousand times. A face, a cat or a landscape appears out of the snow, one small step at a time.

## Core notes
- **Forward process** is fixed, not learned: $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$, so any $x_t$ can be sampled in one shot (no simulation during training).
- **Reverse process** is learned: a network $\epsilon_\theta(x_t,t)$ predicts the noise; the mean of $p_\theta(x_{t-1}\mid x_t)$ is a closed-form function of it.
- **Training loss** (DDPM "simple"): $\mathbb{E}_{t,x_0,\epsilon}\|\epsilon - \epsilon_\theta(x_t,t)\|^2$ — derived from the ELBO of a hierarchical VAE with reweighting.
- **Sampling** is iterative: $T$ network evaluations (NFEs) for ancestral DDPM sampling; 10–50 with DDIM/DPM-Solver; 1–4 with distillation.
- **Equivalent targets**: noise $\epsilon$, clean data $x_0$, score $\nabla\log p_t$, $v = \alpha_t\epsilon - \sigma_t x_0$, or flow velocity; all are affine in each other given $x_t$.
- **Score view** links to Score-Based Generative Models; **ODE view** links to Flow Matching and continuous normalizing flows.
- **Conditioning** via Cross-Attention (text), concatenation (images, masks), or adaptive norm (class / time); steer with Classifier-Free Guidance.
- **Compute trick**: run diffusion in a VAE latent space (Latent Diffusion and Stable Diffusion) — 48× fewer dimensions for SD's 512²×3 → 64²×4.
- **Backbones**: U-Net with Residual Connections + attention, now increasingly Diffusion Transformers.
- Compared with GANs: stable training and mode coverage, at the cost of slow sampling; compared with Autoregressive Generative Models: parallel over dimensions, iterative over time.

## The math
**Notation.** $x_0\sim q(x_0)$ data; $\beta_t\in(0,1)$ variance schedule; $\alpha_t = 1-\beta_t$; $\bar\alpha_t = \prod_{s=1}^t \alpha_s$; $\epsilon\sim\mathcal{N}(0,I)$; $d$ = data dimension.

**Forward process and its marginal.**
$$
q(x_t\mid x_{t-1}) = \mathcal{N}\big(x_t;\sqrt{\alpha_t}\,x_{t-1}, \beta_t I\big), \qquad q(x_t\mid x_0) = \mathcal{N}\big(x_t; \sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t) I\big)
$$
(compose Gaussians: means multiply by $\sqrt{\alpha_t}$, variances accumulate). Hence $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$. Signal-to-noise ratio $\text{SNR}(t) = \bar\alpha_t/(1-\bar\alpha_t)$ decreases monotonically.

**Forward posterior (tractable because we condition on $x_0$).** By Bayes Theorem on Gaussians:
$$
q(x_{t-1}\mid x_t, x_0) = \mathcal{N}\big(x_{t-1};\tilde\mu_t(x_t,x_0), \tilde\beta_t I\big),\quad
\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,\quad
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
$$

**ELBO.** With $p_\theta(x_{0:T}) = p(x_T)\prod_t p_\theta(x_{t-1}\mid x_t)$ and Jensen's inequality,
$$
\begin{aligned}
-\log p_\theta(x_0) &\le \mathbb{E}_q\Big[-\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\Big] \\
&= \underbrace{D_{\mathrm{KL}}\big(q(x_T\mid x_0)\,\|\,p(x_T)\big)}_{L_T\ \text{(no params)}} + \sum_{t=2}^{T}\underbrace{\mathbb{E}_q\, D_{\mathrm{KL}}\big(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\big)}_{L_{t-1}} \underbrace{-\ \mathbb{E}_q\log p_\theta(x_0\mid x_1)}_{L_0}
\end{aligned}
$$
The regrouping uses $q(x_t\mid x_{t-1}) = q(x_t\mid x_{t-1},x_0)$ (Markov) and Bayes to rewrite each forward factor as $q(x_{t-1}\mid x_t,x_0)q(x_t\mid x_0)/q(x_{t-1}\mid x_0)$; the ratio telescopes. See KL Divergence, Variational Inference.

**From KL to noise prediction.** Both distributions in $L_{t-1}$ are Gaussian with fixed variance $\sigma_t^2$, so
$$
L_{t-1} = \mathbb{E}_q\Big[\frac{1}{2\sigma_t^2}\big\|\tilde\mu_t(x_t,x_0) - \mu_\theta(x_t,t)\big\|^2\Big] + C.
$$
Substitute $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$ into $\tilde\mu_t$:
$$
\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\Big).
$$
Choose the same functional form for the model, $\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\big)$. Then
$$
L_{t-1} = \mathbb{E}_{x_0,\epsilon}\Big[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\big\|\epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,\ t\big)\big\|^2\Big],
$$
and dropping the $t$-dependent weight gives Ho et al.'s objective
$$
L_\text{simple}(\theta) = \mathbb{E}_{t\sim\mathcal{U}\{1..T\},\,x_0,\,\epsilon}\big\|\epsilon - \epsilon_\theta(x_t,t)\big\|^2 .
$$
In continuous time the ELBO is $-\tfrac12\mathbb{E}\int_0^1 \text{SNR}'(t)\,\|x_0 - \hat x_\theta(x_t;t)\|^2 dt$, invariant to the schedule except at its endpoints (Kingma et al., 2021); every practical loss is this integrand with a different weighting $w(t)$.

**Score connection (Tweedie).** $\nabla_{x_t}\log q(x_t\mid x_0) = -\epsilon/\sqrt{1-\bar\alpha_t}$, and the minimiser of the denoising loss is a conditional expectation, so
$$
s_\theta(x_t,t) = -\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}} \approx \nabla_{x_t}\log q_t(x_t),\qquad \mathbb{E}[x_0\mid x_t] = \frac{x_t + (1-\bar\alpha_t)\nabla\log q_t(x_t)}{\sqrt{\bar\alpha_t}}.
$$

**SDE / ODE limit.** For a forward SDE $dx = f(x,t)dt + g(t)dw$ (VP: $f=-\tfrac12\beta(t)x$, $g=\sqrt{\beta(t)}$),
$$
\text{reverse SDE: } dx = \big[f(x,t) - g(t)^2\nabla_x\log p_t(x)\big]dt + g(t)\,d\bar w, \qquad
\text{PF-ODE: } \frac{dx}{dt} = f(x,t) - \tfrac12 g(t)^2\nabla_x\log p_t(x).
$$

**Flow-matching connection.** For a general Gaussian path $x_t = a_t x_0 + b_t\epsilon$, the marginal velocity is $v_t(x) = \mathbb{E}[\dot a_t x_0 + \dot b_t\epsilon\mid x_t = x]$. Using $\mathbb{E}[\epsilon\mid x_t] = -b_t\nabla\log p_t$,
$$
v_t(x) = \frac{\dot a_t}{a_t}x - \Big(b_t\dot b_t - \frac{\dot a_t}{a_t}b_t^2\Big)\nabla_x\log p_t(x),
$$
so a trained $\epsilon_\theta$ defines a velocity field and vice versa. Rectified flow is the special case $a_t = 1-t$, $b_t = t$.

## Run the code
```bash
pip install diffusers torch
python diffusion_models.py
```

## Check yourself
<details><summary>Derive why the DDPM loss is "predict the noise".</summary>

The ELBO decomposes into KLs between the tractable forward posterior $q(x_{t-1}\mid x_t,x_0)$ and the model $p_\theta(x_{t-1}\mid x_t)$. Both are Gaussians with fixed variance, so each KL is a scaled squared distance between their means. Writing $x_0$ in terms of $x_t$ and $\epsilon$ makes the posterior mean $\frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon)$. Parameterising the model mean the same way with $\epsilon_\theta$ turns the KL into a weighted $\|\epsilon - \epsilon_\theta\|^2$. Ho et al. drop the weight, which up-weights high-noise steps relative to the true ELBO and improves sample quality.
</details>

<details><summary>How is a diffusion model related to score matching and to flow matching?</summary>

The optimal noise predictor is $\mathbb{E}[\epsilon\mid x_t]$, and Tweedie's formula gives $\nabla\log p_t(x_t) = -\mathbb{E}[\epsilon\mid x_t]/\sigma_t$, so $\epsilon_\theta$ is a scaled score estimate and the DDPM loss is denoising score matching at many noise levels. In continuous time, the score defines both a reverse SDE and a deterministic probability-flow ODE with the same marginals. Flow matching regresses the velocity of an interpolation path directly; for Gaussian paths the velocity is an affine combination of $x_t$ and the score. So the three differ in parameterisation, loss weighting over time and noise schedule, not in what is fundamentally learned.
</details>

<details><summary>Why is sampling slow, and what are the main ways to speed it up?</summary>

Each step is a full network forward pass and the reverse chain is sequential; ancestral DDPM uses as many steps as training timesteps (e.g. 1000). Speed-ups: (1) treat sampling as solving the PF-ODE with better solvers (DDIM, Heun, DPM-Solver++) for 10–50 steps; (2) distil the multi-step sampler into a few-step student (progressive distillation, consistency distillation, adversarial or distribution-matching distillation) for 1–4 steps; (3) shrink each step with latent diffusion, smaller backbones, caching, quantisation; (4) straighten trajectories with rectified flow so Euler steps are accurate.
</details>

<details><summary>What goes wrong if the noise schedule does not reach zero SNR at t = T?</summary>

Training never shows the model a truly pure-noise input: at $t=T$ some low-frequency signal (e.g. the mean brightness) remains. At sampling time we start from pure $\mathcal{N}(0,I)$, which is off-distribution, and the model implicitly assumes medium brightness, so it struggles to make very dark or very bright images. The fix (Lin et al.) is to rescale the schedule to zero terminal SNR, switch to $v$-prediction (since $\epsilon$-prediction is degenerate at zero SNR), and start sampling at the last timestep.
</details>

<details><summary>You train a small diffusion model and the loss plateaus early but samples are blurry or noisy. What do you check?</summary>

Check data scaling ($[-1,1]$ or unit-variance latents), that the timestep embedding actually reaches the network, and that sampling uses the same schedule and parameterisation as training. Evaluate with EMA weights and enough sampling steps; a mismatch between $\sigma_t$ choices or an off-by-one in timestep indexing produces residual noise. Blurriness often means too little capacity at low-noise timesteps or an $x_0$-prediction loss with poor weighting; try min-SNR or EDM weighting. Finally, monitor FID on a fixed seed grid rather than the MSE, which barely moves.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.