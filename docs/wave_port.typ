// [Written by Claude] The math behind EMerge's multi-mode WavePort.
// Build: typst compile docs/wave_port.typ   (figures: uv run --inexact python dev/waveport_figures.py)

#set document(title: "EMerge WavePort")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show link: underline

#align(center)[
  #text(17pt, weight: "bold")[Multi-mode wave ports in EMerge] \
  #v(2pt)
  #text(11pt)[Port eigenproblem, evanescent modes, mode tracking, the propagation constant over frequency, and Floquet ports]
]
#v(8pt)

= Overview

A `WavePort` terminates a guiding structure with a fixed set of $N$ port modes. Each mode $m$ is absorbed and
excited with its own propagation constant $gamma_m = j beta_m$, which is what a port needs when its modes have
different $beta$ (the even and odd mode of coupled lines, TE#sub[10] and TE#sub[30] of a waveguide). The
implementation rests on four pieces:

+ the port eigenproblem, solved at every frequency, *including evanescent modes* (@sec-eig);
+ a normalization and a projection that work for propagating and evanescent modes alike (@sec-norm);
+ the dense wave port boundary condition (WPBC), one rank-1 term per mode (@sec-wpbc);
+ a *fixed mode set*: the modes are chosen at the highest frequency and followed down in frequency by a
  predictor (an extrapolation of $beta^2$) and a field-overlap match (@sec-track, @sec-predict).

The predicted $beta^2$ is *only* used to steer the eigen solver and to identify the modes. The $beta_m$ and
the mode fields that enter the system matrix are always the eigen solution at that frequency, so the
accuracy of the extrapolation never limits the accuracy of the S-parameters. It only has to be good enough
to pick the right eigenvalue (@sec-accuracy).

The `FloquetPort` of periodic cells uses the same ideas with analytic modes: TE and TM Floquet harmonics,
including grating modes, each absorbed with its own $gamma$ by a dense projection term (@sec-floquet).

= The port eigenproblem <sec-eig>

On the port face, with fields $bold(E) = (bold(E)_t + hat(z) E_z) e^(-j beta z)$ ($hat(z)$ points into the
model), EMerge solves the Nédélec–Lagrange formulation in the scaled unknowns
$ bold(e)_t = beta bold(E)_t, quad e_z = -j E_z, $
which turns Maxwell's equations into the generalized eigenproblem
$ bold(A)(k_0) bold(x) = lambda bold(B)(k_0) bold(x), quad lambda = -beta^2. $ <eq-gep>

#figure(
  table(
    columns: 3,
    align: (left, left, left),
    stroke: 0.5pt,
    [*eigenvalue*], [*meaning*], [*treatment*],
    [$lambda < 0$ ($beta^2 > 0$)], [propagating mode], [kept],
    [$lambda > 0$ ($beta^2 < 0$)], [evanescent mode, $beta = -j alpha$], [kept (new)],
    [$lambda approx 0$], [spurious gradient (null-space) solutions], [removed: $|beta^2| < 10^(-4) k_0^2 epsilon_(r,max) mu_(r,max)$],
    [large residual], [unconverged Arnoldi vector], [removed: $norm(bold(A x) - lambda bold(B x)) > 10^(-6) (dots.c)$],
  ),
  caption: [The spectrum of @eq-gep. The spurious cluster sits at $beta^2 = 0$ to round-off (53 of them on a
    $10 times 10.16$ mm port at 8 GHz), while the physical evanescent modes sit where
    $beta^2 = k_0^2 - k_c^2$ puts them (TE#sub[01] $-67440$ vs. $-67520$ analytic, TE#sub[10] $-70521$ vs.
    $-70606$). The two groups are cleanly separated.],
)

The old solvers (`eig_boundary`) only returned $beta^2 >= 1$, so evanescent modes never appeared. The new
routine `SolveRoutine.eig_boundary_shifts` takes a list of target values of $beta^2$ ("shifts"): small ports
($< 500$ DoFs) are solved densely, larger ones by shift-invert Arnoldi, one factorization per cluster of
shifts. A shift must not sit on the spurious cluster at 0; since the shifts are predictions of physical
$beta^2$ values, they don't, except for a mode that is exactly at cutoff at a sample frequency (a known
limitation, @sec-limits).

*Branch of $beta$.* $beta = sqrt(-lambda)$ is taken with $"Im" beta <= 0$. Then an evanescent mode
$e^(-j beta z) = e^(-alpha z)$ decays along its direction of travel, like a lossy mode, and the outgoing
(reflected) evanescent wave $e^(+j beta z)$ decays away from the model, into the port.

= Normalization and modal orthogonality <sec-norm>

Every `WavePort` mode is scaled to
$ abs(1/2 integral.double_S (bold(e)_m times bold(h)_m) dot hat(n) dif S) = 1 $ <eq-norm>
_without_ a complex conjugate. For a lossless propagating mode with real transverse fields this is its power,
so the usual 1 W normalization is unchanged. An evanescent mode carries no power
($"Re" integral (bold(e) times bold(h)^*) dot hat(n) = 0$), but @eq-norm is still finite, so it can be
normalized at all. @eq-norm is also the natural normalization because of the *reciprocity (biorthogonality)
relation*
$ integral.double_S (bold(e)_m times bold(h)_n) dot hat(n) dif S = 0, quad m != n, $ <eq-biorth>
which follows from Lorentz reciprocity for any reciprocal (also lossy) cross-section, and holds for propagating
and evanescent modes alike. The conjugated power product is only orthogonal for lossless propagating modes.

= Wave port boundary condition and S-parameters <sec-wpbc>

With the mode profile $bold(w)_m = gamma_m bold(e)_(t m) + nabla_t e_(z m)$, which is proportional to
$bold(h)_(t m) times hat(n)$, the port operator reads
$ hat(n) times nabla times bold(E) = sum_(m=1)^N bold(w)_m frac(integral bold(E)_t dot bold(w)_m dif S, integral bold(e)_(t m) dot bold(w)_m dif S) - 2 a_m bold(w)_m. $ <eq-wpbc>
Because of @eq-biorth, $integral bold(e)_(t n) dot bold(w)_m dif S = 0$ for $n != m$: the sum is a
projection on the port modes, each mode is absorbed with its own $gamma_m$, and the incident amplitude $a_m$
excites exactly mode $m$. In the FEM, @eq-wpbc adds one dense rank-1 block per mode,
$ -frac(1, j omega mu_0 kappa_m) bold(G)_m bold(G)_m^T, quad bold(G)_m = integral.double bold(w)_m dot bold(N)_i dif S, quad kappa_m = frac(j, omega mu_0) integral.double bold(e)_(t m) dot bold(w)_m dif S, $
and the excitation vector $-2 bold(G)_m$. For an evanescent mode $gamma_m = alpha_m$ is real: the term then
absorbs the exponential tail of a mode that a discontinuity generates near the port, so a port can sit close
to the discontinuity (@sec-valid). The outgoing amplitude of mode $m$ is the same projection,
$ b_m = frac(integral (bold(E) - a_m bold(e)_m) dot bold(w)_m dif S, integral bold(e)_(t m) dot bold(w)_m dif S), quad S_(m n) = b_m / a_n . $
Between propagating modes $|S_(m n)|^2$ is a power ratio. Entries that involve an evanescent mode are
amplitude ratios in the normalization @eq-norm and may exceed 1; for that reason EMerge's power caps
(`mw_cap_sp_col`, `mw_cap_sp_single`) now only act on the propagating block.

= A fixed set of modes <sec-track>

*Selection.* The modes are fixed at the highest frequency $f_max$: the $N$ modes with the largest
$"Re" beta^2$ there, or, with `number_of_modes=None`, all modes that propagate at $f_max$. Evanescent modes are
included when fewer than $N$ modes propagate. If modes $N$ and $N+1$ are degenerate, a warning asks for both:
with only one, the port cannot absorb a field of that $beta$.

*Tracking.* From $f_max$ the modes are followed through the sweep frequencies, one step at a time. At each
new frequency:

+ predict $beta_m^2$ for every tracked mode (@sec-predict);
+ solve @eq-gep for the modes closest to the predictions (about $2N + 4$ candidates);
+ match tracked modes to candidates with a linear assignment (Hungarian) on the cost
  $ C_(m c) = -"CMAC"_(m c) + 1/2 frac(abs(beta_c^2 - beta_("pred",m)^2), k_0^2 epsilon_(r,max) mu_(r,max)), quad "CMAC"_(m c) = frac(abs(bold(u)_m^H bold(v)_c)^2, norm(bold(u)_m)^2 norm(bold(v)_c)^2), $
  where $bold(u)_m$, $bold(v)_c$ are the transverse field coefficients at the previous and the current
  frequency (same mesh, same DoFs);
+ rotate the phase of each matched mode so that $bold(u)_m^H bold(v)_c$ is real and positive.

Step 3 keeps every S-matrix index attached to the same physical mode through cutoff, where $beta^2$ changes
sign, and through *mode crossings*, where ordering by $beta$ would swap the labels. Step 4 makes the
modes, and therefore the S-parameters, smooth functions of frequency. A CMAC below 0.7 gives a warning
(too coarse a frequency step).

= Predicting $beta^2$ over frequency <sec-predict>

== Homogeneous cross-sections

If the port face is one isotropic material, the modes are pure TE, TM or TEM with frequency-independent
transverse profiles, and
$ beta^2(k_0^2) = epsilon_r mu_r k_0^2 - k_c^2 . $ <eq-homog>
$beta^2$ is exactly linear in $k_0^2$ with slope $epsilon_r mu_r$ (@fig-homog). Both $beta$ and the modes at
any frequency follow from one solve, which is what a homogeneous `WavePort` does (@sec-scaled).

#figure(image("figures/homogeneous_beta2.svg", width: 92%),
  caption: [$beta^2$ of the 12 + 6 tracked modes of the two homogeneous ports of the step in @sec-valid,
    8–30 GHz. Every line is straight with slope 1; the zero crossings are the cutoffs. Below its cutoff a mode
    stays in the port as an evanescent mode.]) <fig-homog>

This is also where the `ModalPort` bug sat: modes computed once were extrapolated with
$beta^2 = beta_0^2 + k_0^2 - k_(0,0)^2$, which leaves out $epsilon_r mu_r$, and their power normalization was
not updated (TE: $P prop beta slash omega$). A uniform guide with $epsilon_r = 2.2$ reflected $|S_(11)| = 0.41$
at 6 GHz. `get_beta` now uses @eq-homog, and a `ModalPort` reuses its modes only when they are exact at every
frequency (one material, only TEM modes). All other ports are solved again at every frequency.

== Inhomogeneous cross-sections: the slope $d beta^2 slash d k_0^2$

In the scaled unknowns, the Rayleigh quotient of the curl–curl operator becomes rational in $B = beta^2$ for a
fixed field, with $K = k_0^2$:
$ K = frac(a + B b, c + B d), quad cases(
  a = integral mu_r^(-1) |nabla_t times bold(e)_t|^2, & b = integral mu_r^(-1) |bold(e)_t + nabla_t e_z|^2,
  c = integral epsilon_r |bold(e)_t|^2, & d = integral epsilon_r |e_z|^2 .) $
A mode makes $G(K, B; bold(e)) = a + B b - K (c + B d)$ stationary in $bold(e)$. Along the dispersion curve the
change of the field therefore does not contribute (Hellmann–Feynman), and
$ s(K) equiv frac(d beta^2, d k_0^2) = frac(c + beta^2 d, b - k_0^2 d) = n_"eff" n_g = frac(c^2, v_p v_g) . $ <eq-slope>
(For lossy, reciprocal media the same holds with the non-conjugated bilinear forms, because the operator is
complex symmetric.) @eq-slope says how fast $beta^2$ can turn:

- *Homogeneous:* $s = epsilon_r mu_r$ for every mode, @eq-homog.
- *TE-type modes* ($e_z = 0$, so $d = 0$), and scalar, separable problems such as layered guides:
  $ s = frac(integral epsilon_r |bold(e)_t|^2, integral mu_r^(-1) |bold(e)_t|^2) in [min epsilon_r mu_r, max epsilon_r mu_r] $ <eq-bound>
  rigorously: the slope is an average of $epsilon_r mu_r$, weighted with where the field lives. At low
  frequency the field spreads over all materials, at high frequency it concentrates in the densest one, and
  $s -> max epsilon_r mu_r$.
- *Lossless propagating modes:* $s > 0$, since $v_g = P slash W > 0$ for a forward mode.
- *Hybrid modes* (quasi-TEM, LSE/LSM): the $e_z$ terms in $b - k_0^2 d$ are not sign definite, and I have no
  proof that @eq-bound holds. In every case computed here it does (@fig-slope): the slab guide stays in
  $[1.34, 3.94]$ against the bound $[1, 4]$, the microstrip quasi-TEM mode in $[3.31, 4.10]$ against
  $[1, 4.4]$, and the box modes of the microstrip in $[1.07, 3.28]$. The tracker therefore does not rely on
  @eq-bound, but it is a useful sanity check.

#figure(image("figures/inhomogeneous_neff2.svg", width: 100%),
  caption: [Tracked modes of two inhomogeneous ports (port solves only). Left: WR90 with a 3 mm slab of
    $epsilon_r = 4$ on the bottom wall; modes 4 and 5 cross near 26 GHz and keep their labels. Right: a
    shielded microstrip line ($epsilon_r = 4.4$): the quasi-TEM mode (1) and two box modes, which cross near
    49 GHz.])

#figure(image("figures/slope_bounds.svg", width: 100%),
  caption: [The slope @eq-slope, from the tracked $beta^2$, against $[min epsilon_r mu_r, max epsilon_r mu_r]$
    (shaded). Modes 1–4 of the slab guide are TE#sub[m0]-like with the same field across the slab, so they
    share one slope curve (the problem separates in $x$ and $y$).]) <fig-slope>

== The predictor and its error <sec-accuracy>

With the two nearest computed frequencies $K_1, K_2$, the prediction at $K$ is the secant
$ beta_"pred"^2 = beta^2(K_1) + frac(beta^2(K_1) - beta^2(K_2), K_1 - K_2) (K - K_1), $
and with one computed frequency the slope $epsilon_r mu_r$ of the face (exact if homogeneous). Two error
estimates follow from @eq-slope:

- *Guaranteed (first order).* If $s in [s_"lo", s_"hi"]$ on the step, the true $beta^2$ lies in a cone around
  the last point and
  $ abs(beta_"pred"^2 - beta^2) <= (s_"hi" - s_"lo") abs(Delta K), quad frac(abs(Delta beta^2), k_0^2) lt.tilde 2 (s_"hi" - s_"lo") frac(Delta f, f) . $
  This holds whenever @eq-bound does, without any smoothness assumption, and is zero for homogeneous ports.
- *Typical (second order).* For a smooth $s$ the secant error is $approx s'(K) Delta K^2$, so it scales as
  $Delta f^2$ (@fig-pred). In the slab guide at 0.1 GHz steps it is at most $5.5 dot 10^(-4) k_0^2$, while
  the guaranteed bound is $2 dot 3 dot 0.1 slash 30 = 0.02$.

#figure(image("figures/predictor_error.svg", width: 62%),
  caption: [Measured secant error of $beta^2$ against the frequency step, marching down in frequency as the
    tracker does. The slope of 2 in the log–log plot is the second-order behaviour.]) <fig-pred>

*How accurate does the prediction have to be?* The prediction is not used in the solve: $gamma_m$, $kappa_m$
and the mode fields in @eq-wpbc are the eigen solution at that frequency, to the accuracy of the 2D mesh. The
prediction has two jobs:

+ *Shift of the eigen solver.* Shift-invert finds the eigenvalues closest to the shift whatever its error. The
  only requirement is to stay away from the spurious cluster at $beta^2 = 0$.
+ *Identification.* The match must be unambiguous. That holds if the prediction error is below half the gap to
  the nearest other mode, _or_ if the fields tell the modes apart. Near crossings the gap goes to zero, so
  there the CMAC term decides. In the slab guide the lowest CMAC over 261 frequencies is 0.999, through the
  crossing. In the microstrip it drops to 0.78 at the box mode crossing (the solver returns mixtures of two
  nearly degenerate modes) and is back to 1.000 one step later, with the labels kept.

So a coarser predictor would still give exact S-parameters, and the step size is set by the identification,
not by the accuracy of $beta$.

== Homogeneous ports: one solve, scaled modes <sec-scaled>

A `WavePort` detects a homogeneous port face and then solves its modes only once, at $f_max$. The port is
scaled instead of solved when all of these hold:

- the face is one isotropic material,
- that material has no conductivity ($sigma$ adds $-j sigma slash (omega epsilon_0)$ to $epsilon_r$, so
  $epsilon_r$ changes with frequency),
- $epsilon_r$ and $mu_r$ are the same at $f_min$ and $f_max$ (no dispersion).

A constant loss tangent is allowed: it makes $epsilon_r$ complex but frequency independent. The walls are PEC
in the port eigenproblem at every frequency, also when they carry a surface impedance, so lossy walls do not
matter here.

At another frequency $beta$ follows from @eq-homog, with the branch $"Im" beta <= 0$. The fields follow from
the mode type:

- TE and TEM modes keep their fields;
- a TM mode has $bold(E)_t = -j beta slash k_c^2 nabla_t E_z$, so its $bold(E)_t$ is scaled by
  $beta slash beta_"ref"$;
- every mode is then normalized again (@sec-norm).

The scaled modes are the same fields the eigen solver would return, so the S-parameters do not change. On
the step of @sec-valid they agree with the per-frequency solve to $1.5 dot 10^(-5)$. The port mode time
drops from 27–29 s to 2 s of the sweep.

*Degenerate TE/TM pairs.* An eigen solver returns an arbitrary basis of a degenerate pair, for example
TE#sub[11]/TM#sub[11] of a rectangular guide, and a mix of TE and TM cannot be scaled with one factor. Such
a group is split first:

+ the TE combinations are those with $E_z = 0$ and the TM combinations those with $H_z = 0$;
+ each subspace gets a real basis (with $E_z$ rotated by $-j beta slash |beta|$, a TE or TM mode is real up
  to one phase);
+ modes of one type are made orthogonal in $integral.double (bold(e)_m times bold(h)_n) dot hat(bold(n)) dif S$.

A group whose modes are already pure keeps the basis of the solver. If a mode is neither TE, TM nor TEM, the
port falls back to solving and tracking.

== Why not interpolate inhomogeneous ports?

For inhomogeneous ports an interpolated $beta$ would also need interpolated mode fields. That requires a
consistent gauge and a treatment of degenerate subspaces, and gives up the exactness above for an error that
has to be controlled. The re-solve is cheap: a port solve costs 0.2–0.5 s per frequency.

= Validation <sec-valid>

H-plane step from WR90 (22.86 mm) to a 10 mm wide guide, both 10.16 mm high, 8–30 GHz, 45 frequencies, ports
with 12 and 6 modes (all modes that propagate at 30 GHz). The reference is a mode-matching solution of the
same step (TE#sub[n0] modes, 80 + 35 modes, converged to $4 dot 10^(-4)$), de-embedded to the junction.
Below 15.0 GHz *no* mode propagates at the narrow port: TE#sub[10] there is evanescent and $|S_(11)| = 1$.

#figure(image("figures/step_validation.svg", width: 100%),
  caption: [EMerge against mode matching. $|S_(11)|$ agrees to 0.004 and $angle S_(11)$ to 1–2°,
    $|S_(21)|$ to 0.003. With the narrow port 3 mm from the step instead of 20 mm the result is the same
    to $10^(-3)$: the evanescent modes are absorbed by the port, not reflected.]) <fig-step>

#figure(
  table(
    columns: 2,
    align: (left, left),
    stroke: 0.5pt,
    [*check*], [*result*],
    [$max |S_(11) - S_(11)^"MM"|$ (port 2 at 20 / 3 mm)], [0.027 / 0.026 (mean 0.009), largest at the 15 GHz cutoff],
    [power over propagating modes, $|sum |S_(m 1)|^2 - 1|$], [$2 dot 10^(-6)$],
    [uniform dielectric guide, `ModalPort`, $max |S_(11)|$], [0.41 before, $< 0.001$ after the fix],
    [evanescent transmission, 10 mm guide at 12 GHz], [$|S_(21)| = e^(-alpha L)$, no reflection],
  ),
  caption: [Results of `dev/waveport_step.py`, `dev/modal_port_beta_bug.py` and `tests/test_wave_port.py`.],
)

= Floquet ports <sec-floquet>

A `FloquetPort` terminates a periodic unit cell (`RectCell`, `HexCell`) above and below. Its modes are
known in closed form, so it needs no eigen solver and no tracking. It does need the rest of the machinery:
several modes with different $gamma$, a fixed mode set, and exact absorption of evanescent modes.

== Floquet modes

Let $bold(a)_1, bold(a)_2$ be the lattice vectors of the cell and $bold(b)_1, bold(b)_2$ the reciprocal
vectors, $bold(b)_i dot bold(a)_j = 2 pi delta_(i j)$. The periodic boundaries impose the phase
$e^(-j bold(k)_(t,00) dot bold(a))$ with $bold(k)_(t,00) = k_0 sin theta (cos phi, sin phi)$. Every field that
satisfies these boundaries is a sum of Floquet harmonics
$ bold(k)_(t,m n) = bold(k)_(t,00) + m bold(b)_1 + n bold(b)_2, quad
  k_(z,m n) = sqrt(epsilon_r k_0^2 - |bold(k)_(t,m n)|^2), quad "Im" k_z <= 0, $ <eq-floquet>
each as a TE and a TM plane wave. With $hat(bold(t)) = bold(k)_t slash |bold(k)_t|$ (the scan direction
$(cos phi, sin phi)$ when $bold(k)_t = 0$) the transverse fields are
$ bold(e)_"TE" = A_"TE" (hat(bold(z)) times hat(bold(t))) e^(-j bold(k)_t dot bold(r)), quad
  bold(e)_"TM" = -A_"TM" hat(bold(t)) e^(-j bold(k)_t dot bold(r)). $
The amplitudes give 1 W through the cell area $S$ for a propagating mode:
$ A_"TE"^2 = (2 k_0 Z_0) / (S |k_z|), quad A_"TM"^2 = (2 Z_0 |k_z|) / (S k_0 epsilon_r). $ <eq-floquet-amp>
For an evanescent mode the same expressions with $|k_z|$ normalize the amplitude, as for the `WavePort`
(@sec-norm). Plane waves have $hat(bold(n)) times nabla times bold(E) = gamma bold(E)_t$ with
$ gamma_"TE" = j k_z, quad gamma_"TM" = j epsilon_r k_0^2 slash k_z. $ <eq-floquet-gamma>
For an evanescent mode, $k_z = -j alpha$ gives $gamma_"TE" = alpha$ and $gamma_"TM" = -epsilon_r k_0^2 slash alpha$.

Over one cell the harmonics are orthogonal in the *conjugate* product:
$integral.double_S bold(e)_i dot bold(e)_j^* dif S = N_i delta_(i j)$. Two harmonics differ by
$e^(-j(m bold(b)_1 + n bold(b)_2) dot bold(r))$, which integrates to zero over a cell. The TE and TM modes of one
harmonic are orthogonal vectors. The non-conjugate product of the `WavePort` (@sec-norm) does *not* work
here: under an oblique scan $bold(e)_i dot bold(e)_i$ contains $e^(-2j bold(k)_t dot bold(r))$, which is
not constant over the cell.

== Port operator

The port imposes $hat(bold(n)) times nabla times bold(E) = Gamma bold(E)_t$ with
$ Gamma bold(E)_t = gamma_r bold(E)_t + sum_(i in cal(B)) (gamma_i - gamma_r) bold(e)_i
  (integral.double bold(E)_t dot bold(e)_i^* dif S) / N_i, quad gamma_r = j k_0 sqrt(epsilon_r). $ <eq-floquet-op>
On the span of the basis $cal(B)$, every mode is absorbed with its own $gamma_i$, which is exact. The rest of
the field sees a first-order absorbing boundary, $gamma_r$. If $cal(B)$ held every harmonic the mesh can
resolve, @eq-floquet-op would be the exact Dirichlet-to-Neumann map of the half-space. In the FEM system
@eq-floquet-op becomes
$ gamma_r M + sum_i (gamma_i - gamma_r) / N_i bold(G)_i bold(G)_i^H, quad
  G_(i,k) = integral bold(N)_k dot bold(e)_i dif S, $
a Robin mass matrix plus a dense Hermitian rank-1 term per basis mode on the port DoFs, the same block
pattern as the dense WPBC. A port mode $j$ is a combination $sum_i c_(j i) bold(e)_i$ of basis modes. Its
excitation is $-2 sum_i c_(j i) gamma_i bold(G)_i$. Its outgoing amplitude is the conjugate projection
$ b_j = (integral.double (bold(E) - a_j bold(e)_j) dot bold(e)_j^* dif S) / (integral.double |bold(e)_j|^2 dif S). $
With 1 W modes $|S_(i j)|^2$ is the diffraction efficiency of a propagating order.

== Mode set

As for the `WavePort` (@sec-track), the modes are fixed at the highest frequency, at the current scan angle:

- `number_of_modes=1` (the default): one S-matrix mode, the mix $cos psi thin bold(e)_"TE,00" + sin psi thin bold(e)_"TM,00"$ of
  the polarization angle $psi$. This is the old single-mode port. TE#sub[00] and TM#sub[00] are both
  still in $cal(B)$ and both absorbed exactly; before this change the port used one blended $gamma$.
- `number_of_modes=2`: TE#sub[00] and TM#sub[00] as separate S-matrix modes (port modes $p.1$ and $p.2$).
- More: also the grating modes $(m, n) != (0, 0)$, TE before TM, ordered by $|bold(k)_(t,m n)|^2$ at
  $f_max$. The specular $(0,0)$ harmonic always comes first.
- `None`: all modes that propagate at $f_max$.

A grating mode in the set that is below cutoff at a lower frequency stays in the S-matrix as an evanescent
mode ($k_z = -j alpha$), the same as for the `WavePort`.

== Validation

`dev/floquet_ports.py`, two Floquet ports on a cell:

+ *Uniform slab* ($epsilon_r = 4$, 6 mm), 10 GHz, 10 × 10 mm cell, $theta = 0 ... 70°$ at $phi = 0°$ and
  $30°$, TE#sub[00] and TM#sub[00]. The reference is the analytic slab response.
+ *Lamellar grating* (period 30 mm, 50 % bars of $epsilon_r = 4$, 6 mm thick), normal incidence, TE
  ($E_y$), 8.25–15.75 GHz. The $(plus.minus 1, 0)$ orders propagate above $c slash Lambda = 10$ GHz. The
  reference is RCWA (Moharam & Gaylord, 81 orders; it reproduces the analytic slab to $10^(-15)$ and
  conserves energy to $10^(-14)$).
+ *Hexagonal (brick) lattice* (20 × 12 mm, lattice vectors $(a, 0)$ and $(a slash 2, b)$) with an $epsilon_r = 6$
  post, 20 GHz, scan $(20°, 30°)$. Six modes propagate. The ports sit 10 mm or 4 mm from the post.

#figure(image("figures/floquet_slab.svg", width: 100%),
  caption: [Uniform slab: TE and TM against the analytic response at both scan planes. The largest
    complex error in $S_(11)$ and $S_(21)$ is $4 dot 10^(-4)$, cross-polarization $< 2 dot 10^(-5)$, and
    power is conserved to $1.5 dot 10^(-4)$.]) <fig-floquet-slab>

#figure(image("figures/floquet_grating.svg", width: 100%),
  caption: [Lamellar grating against RCWA. With the six modes of 16 GHz in the ports (`number_of_modes=None`),
    every efficiency agrees to 0.0015 and power is conserved to $3 dot 10^(-3)$ (worst near the 14.5 GHz
    resonance, $< 10^(-4)$ elsewhere). With only TE#sub[00]/TM#sub[00] the efficiencies are off by up to
    0.38 and 15–83 % of the power is lost, also *below* 10 GHz. There the $(plus.minus 1, 0)$ orders are
    evanescent but, 12 mm from the grating, still at $e^(-1.4)$ of their amplitude. Terminated with
    $gamma_r$ instead of their own $gamma$, they reflect.]) <fig-floquet-grating>

#figure(
  table(
    columns: 5,
    align: (center, center, center, center, center),
    stroke: 0.5pt,
    [*modes per port*], [*port distance*], [$max |sum |S|^2 - 1|$], [$S_(11)$ TE#sub[00]], [$S_(21)$ TE#sub[00]],
    [6 (`None`)], [10 mm], [$1.7 dot 10^(-2)$], [$-0.135 + 0.179 j$], [$-0.347 - 0.617 j$],
    [6 (`None`)], [4 mm], [$1.8 dot 10^(-1)$], [$-0.128 + 0.115 j$], [$-0.333 - 0.535 j$],
    [14], [10 mm], [$3.4 dot 10^(-4)$], [$-0.135 + 0.185 j$], [$-0.350 - 0.625 j$],
    [14], [4 mm], [$5.6 dot 10^(-3)$], [$-0.133 + 0.185 j$], [$-0.350 - 0.624 j$],
    [30], [10 mm], [$3.2 dot 10^(-4)$], [$-0.135 + 0.185 j$], [$-0.350 - 0.625 j$],
    [30], [4 mm], [$3.5 dot 10^(-4)$], [$-0.134 + 0.185 j$], [$-0.350 - 0.625 j$],
  ),
  caption: [Hexagonal lattice with a post, S-parameters de-embedded to the post. With only the propagating
    modes in the port, the strong evanescent harmonics near the post are terminated by $gamma_r$ and the
    result depends on the port distance. With 30 modes (24 of them evanescent) the port 4 mm from the post
    gives the same S-parameters as at 10 mm, to $5 dot 10^(-4)$.],
) <tab-floquet-hex>

The hexagonal runs also exposed an old bug: a scan angle set with `set_scanangle` *before* the ports (or
the periodic boundaries) were created never reached them. The ports then assumed normal incidence while the
walls were phased for the scan, and $|S_(11)|$ of the slab came out 10 % low at 20°. The cell now stores
the scan angle and applies it to boundaries and ports created later.

*Choosing the number of modes.* `None` is right when the ports are far from the structure compared with
the decay length $1 slash alpha$ of the first evanescent harmonics. Close to a strongly scattering
structure, add evanescent harmonics until the S-parameters stop changing (@tab-floquet-hex). Each mode
adds a right-hand side and its projections to the post-processing, but the dense block on the port face
has the same size for any number of modes.

= Limitations <sec-limits>

- A mode exactly at cutoff at a sample frequency ($|beta^2| < 10^(-4) k_0^2 epsilon_(r,max) mu_(r,max)$)
  cannot be told apart from the spurious cluster. Tracking then stops with an error; moving the frequency
  point slightly avoids it.
- Degenerate pairs are orthogonalized in the power product only if they propagate. A degenerate evanescent
  pair (TE#sub[11]/TM#sub[11] below cutoff) keeps the arbitrary basis of the eigen solver, and the tracker
  matches it by overlap.
- Evanescent modes have no characteristic impedance; their $Z_0$ is not defined.
- The bound @eq-bound is proven for homogeneous, TE-type and separable problems only.
- Floquet ports assume a homogeneous, non-magnetic medium at the port face (`er`, $mu_r = 1$) and a face
  normal to $z$. A grating mode exactly at cutoff (a Wood/Rayleigh anomaly, $k_z = 0$) has an infinite
  $gamma_"TM"$; it is clipped to $|k_z| = 10^(-6) k_0$ with a warning.
- Under an oblique scan the system is not reciprocal in the usual sense ($S_(i j) = S_(j i)$ holds for the
  reversed scan angle), so the reciprocity correction of the S-matrix should not be used with Floquet ports.
