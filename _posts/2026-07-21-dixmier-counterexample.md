---
layout: post
title: An explicit counterexample to the Dixmier conjecture for the third Weyl algebra
date: 2026-07-21 00:00:00-0000
description: The explicit non-surjective endomorphism of A₃ implied by Alpöge's Jacobian-conjecture counterexample, with Lean 4 certification
tags: weyl-algebra dixmier jacobian-conjecture lean
categories: algebra
---

Alpöge's counterexample to the Jacobian conjecture in dimension three
([announced July 19, 2026](https://x.com/__alpoge__/status/2079028340955197566))
implies, through the classical implication $$\mathrm{DC}_3 \Rightarrow \mathrm{JC}_3$$,
that the Dixmier conjecture fails for the third Weyl algebra $$A_3$$: there is an
injective, non-surjective $$\mathbb{C}$$-algebra endomorphism of $$A_3$$. The existence
statement is immediate to experts. This note supplies what the observation does not,
namely the explicit witness, which is to my knowledge the first explicit counterexample
endomorphism of any Weyl algebra since Dixmier posed the problem in 1968.

**[Full write-up (PDF)]({{ '/assets/pdf/dixmier3.pdf' | relative_url }})** &middot;
**[sympy certificate]({{ '/assets/dixmier3_check.py' | relative_url }})** &middot;
**[Lean 4 proofs](https://github.com/frozaken/dixmier3)**

## The construction

With $$u = 1 + x_1 x_2$$, Alpöge's map $$F = (F_1, F_2, F_3)$$ is

$$
\begin{aligned}
F_1 &= u^3 x_3 + x_2^2\,u\,(4 + 3 x_1 x_2), \\
F_2 &= x_2 + 3 x_1 u^2 x_3 + 3 x_1 x_2^2 (4 + 3 x_1 x_2), \\
F_3 &= 2 x_1 - 3 x_1^2 x_2 - x_1^3 x_3,
\end{aligned}
$$

with $$\det J = -2$$ identically and the explicit collision

$$
F(0, 0, -\tfrac{1}{4}) = F(1, -\tfrac{3}{2}, \tfrac{13}{2})
= F(-1, \tfrac{3}{2}, \tfrac{13}{2}) = (-\tfrac{1}{4}, 0, 0).
$$

Set $$N = -\tfrac{1}{2}\,\mathrm{adj}\,J = J^{-1}$$, a matrix of polynomials.

> **Theorem.** The assignment
> $$
> \begin{aligned}
> \varphi(\hat{x}_i) &= F_i(\hat{x}_1, \hat{x}_2, \hat{x}_3), \\
> \varphi(\partial_i) &= \sum_{k=1}^{3} N_{ki}(\hat{x}_1, \hat{x}_2, \hat{x}_3)\,\partial_k
> \end{aligned}
> $$
> extends to a $$\mathbb{C}$$-algebra endomorphism $$\varphi \colon A_3 \to A_3$$ which is
> injective and not surjective. The Dixmier conjecture therefore fails for $$A_n$$ for
> every $$n \ge 3$$.

The lift $$F \mapsto \varphi$$ is classical (Bass–Connell–Wright p. 297, crediting
Vaserstein and Kac; van den Essen Ch. 10; Bavula
[arXiv:2112.03177](https://arxiv.org/abs/2112.03177)). What is new is the input: until
this month the recipe had no non-invertible Keller map to apply it to. Because the lifted
operators have order at most one, no quantum corrections arise, and well-definedness is
*equivalent* to two families of polynomial identities: $$JN = I_3$$, and flatness
$$\sum_k (N_{ki}\,\partial_k N_{lj} - N_{kj}\,\partial_k N_{li}) = 0$$. Non-surjectivity
follows from maximal commutativity of $$\mathbb{C}[\hat{x}]$$ together with the collision.

Two corollaries. First, an explicit counterexample to the Poisson conjecture
$$\mathrm{PC}_3$$. Second, a *symplectic* Keller counterexample: the cotangent lift
$$\Phi(x, y) = (F(x),\, N(x)^{\mathsf{T}} y)$$ on $$\mathbb{C}^6$$ preserves the canonical
bracket, has $$\det J\Phi = 1$$ identically, and is not injective. A unimodular
counterexample on $$\mathbb{C}^3$$ is trivial from $$F$$ by a linear volume correction; the
point of $$\Phi$$ is that it is symplectic, which no map on odd-dimensional $$\mathbb{C}^3$$
can be.

## Certification

Every computational claim is certified twice, independently. The
[sympy script]({{ '/assets/dixmier3_check.py' | relative_url }}) verifies
$$\det J = -2$$, $$JN = I_3$$, the nine flatness identities, bracket preservation for
$$\Phi$$, and $$\det J\Phi = 1$$ in exact rational arithmetic. A
[Lean 4 / mathlib development](https://github.com/frozaken/dixmier3) proves the same
identities plus the collision, with no `sorry` and no axioms beyond the standard three
(`propext`, `Classical.choice`, `Quot.sound`).

## What remains

$$\mathrm{DC}_1$$ and $$\mathrm{DC}_2$$ remain open, and
$$\mathrm{DC}_2 \Rightarrow \mathrm{JC}_2$$: the two-variable Dixmier conjecture is at
least as strong as the plane Jacobian conjecture, which survives. Zheglov
([arXiv:2410.06959](https://arxiv.org/abs/2410.06959)) proposes a proof of
$$\mathrm{DC}_1$$. If it holds, no endomorphism of $$A_1$$ has a plane Keller
counterexample as its classical shadow, whereas in dimension three quantization succeeds
with no corrections. Locating the obstruction that appears only in low dimension is, I
believe, a fruitful new question about the plane Jacobian problem.

*Prepared with substantial assistance from AI systems (Claude Fable 5, Anthropic;
cross-checked with GPT-5.6-sol, OpenAI). All computational claims are machine-certified.
Corrections welcome: marcumail (at) gmail (dot) com.*
