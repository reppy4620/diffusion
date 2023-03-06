diffusion
===

The Implementation of DIffusion (like) models.

- DDPM
- Score-SDE
- Rectified Flow
- COnditional Flow Matching(Simplified version)

Rectified Flow is so simple compared to score-sde because it considers simple trajectries induced by ODE mapping tractable distribution (e.g. gaussian) to data distribution, but in my case, generated images are collapsed.