# Foraging Paralysis as a Collapse Mechanism

**A Bifurcation Analysis of Honey Bee Colony Dynamics under Predation by *Vespa velutina***

Bachelor's thesis in mathematics, Technical University of Munich (TUM), 2026
Author: Anton Fritzler · Supervisor: Prof. Dr. Christina Kuttler

📄 [Read the thesis (PDF)](thesis.pdf) · 📓 [Walk through the computations](thesis_notebook.ipynb) · 🐍 [Library code](thesis_code.py)

<p align="center"><img src="figs/ladder.png" width="85%"></p>

## The short version

The Asian hornet *Vespa velutina* hunts honey bees by hovering in front of the hive entrance and catching
returning foragers in flight. Field observations by Requier et al. (2019) showed two separate effects.
Bees that fly are sometimes caught (**predation**). Far more striking, the colony largely **stops flying**:
foraging traffic drops by about an order of magnitude between zero and twenty hornets.

Mathematical models of colony collapse usually treat a stressor as extra mortality. In this thesis I put the
hornet into the colony model of Khoury, Barron and Myerscough (2013) through **both** channels and ask which one
does the damage.

The answer is clear-cut:

* **Paralysis alone can collapse a colony.** Three hornet loads organise the results (figure above). Above about
  **6 hornets** the food stores stop growing, above about **8** the colony loses its positive equilibrium and above
  about **10** collapse is proved for every initial state. Requier et al. routinely observed loads up to 20.
* **Predation alone cannot.** To remove the colony's equilibrium at full flight activity, forager mortality would
  have to triple. The fitted capture rates raise it by at most **0.24 %**. The gap is a factor of **851**.
* The hornet's two effects even work against each other: the more it paralyses the colony, the fewer bees
  leave the hive for it to catch.

In one sentence: **a hornet does not need to kill bees to kill a colony.**

## Research questions

| | Question | Answer |
|---|---|---|
| **Q1** | Can foraging paralysis alone drive a colony to collapse, without any increase in mortality? | Yes, below a foraging activity of $p_E = 0.346$ (about 9.7 hornets) collapse is proved. |
| **Q2** | Is there an explicit, interpretable threshold? | Yes. All thresholds share one closed form, a balance sheet of the colony's honey budget. |
| **Q3** | Which channel matters more, paralysis or predation? | Paralysis. Predation moves every threshold by about a hundredth of a hornet. |

## The model

The analysis works with a three-dimensional reduction (M) of the published delay model. Its state is the number of
hive bees $H$, foragers $F$ and the food stores $f$ (in grams), with $N = H + F$:

$$
\begin{aligned}
\dot H &= L\,s(f)\,\frac{H}{H+v} \;-\; H\Big(\alpha(f) - \sigma\frac{F}{N}\Big),\\
\dot F &= H\Big(\alpha(f) - \sigma\frac{F}{N}\Big) \;-\; m\,F,\\
\dot f &= c\,p\,F \;-\; \gamma_A N \;-\; \kappa\,L\,s(f)\,\frac{H}{H+v},
\end{aligned}
$$

with brood survival $s(f) = f^2/(f^2+b^2)$, food-dependent recruitment $\alpha(f) = \alpha_{\min} + \alpha_{\max}\,(1 - s(f))$
and the honey cost per reared bee $\kappa = \gamma_B/\phi$.

The hornet load $V$ enters at exactly two places:

| Channel | Where it acts | Calibration (Requier et al. 2019) |
|---|---|---|
| **Paralysis** | food intake $c \to c\,p(V)$ | $p(V) = e^{-\beta V}$, $\beta = 0.109$ |
| **Predation** | forager mortality $m_0 \to m_0 + \mu(V)$ | $\mu = \nu\,p\,\mathrm{HF}(p)$, $\mathrm{HF}(p) = h_0 e^{-\zeta p}$ |

![schema_bio](figs/schema_bio.png)

*The published model after Khoury et al. (2013), with the two hornet impacts in red.*

![calibration](figs/calibration.png)

*Calibration: (a) flight activity against hornet load, (b) homing failure per return flight, (c) the resulting extra mortality, which never exceeds a quarter of a percent of the baseline.*

## Main results

### Four thresholds on one ladder

Every threshold in foraging activity has the same form $p_X = \big(\gamma_A(1+X) + \kappa m\big)/c$ and differs only in $X$.
At the baseline mortality $m_0 = 0.154\ \mathrm{d}^{-1}$:

| Threshold | What happens below it | $p$ | hornets $V$ |
|---|---|---|---|
| $p_{\rm acc}$ accumulation edge | stores stop growing without bound | 0.518 | 6.04 |
| $p^*$ existence edge | the positive equilibrium disappears | 0.415 | 8.07 |
| $p_E$ collapse bound | collapse from every initial state (proved) | 0.346 | 9.73 |
| $p_\kappa$ break-even bound | a forager no longer earns her own keep plus her replacement | 0.319 | 10.47 |

If forager mortality is allowed to scale fully with flight activity, the loads shift upwards (the bar ends in the ladder
figure). The existence edge then sits near 17 hornets, still inside the range Requier et al. observed.

### The existence band and the two ways to die

![band](figs/band.png)

Positive equilibria exist exactly in the band $p^* < p < p_{\rm acc}$. They are locally stable (Routh–Hurwitz).

![endstates](figs/endstates.png)

*(a) Below the existence edge the colony dies with honey left in the comb. (b) Under heavy paralysis the stores run out first.*

Global collapse below $p_E$ follows from an energy capital $W = f + a_\star(H+F)$ that decreases along every solution
(budget identity plus Barbalat's lemma). An analogous energy function works for the full delay model.

### The food axis and bistability

![axis3d](figs/axis3d.png)

In blow-up coordinates the extinct state becomes a curve that attracts below the critical store level $f_{\rm crit} \approx 313$ g and repels above it. The branch of equilibria collides with it at the existence edge.

![basins](figs/basins.png)

Basins of attraction inside the band. $f_{\rm crit}$ decides the fate of tiny colonies only. Larger colonies survive with fewer stores.

### Paralysis against predation

![pm](figs/pm.png)

(a) The hornet trajectory in the plane of flight activity $p$ and forager mortality $m$ runs almost horizontally: it moves the colony along the activity axis, not the mortality axis. (b) The same trajectory with the mortality axis stretched by $10^4$. Predation peaks near 17 hornets and then falls again because fewer bees fly.

The gap between what predation would need and what it can deliver splits into two factors:

$$
\frac{m^* - m_0}{\max_V \mu} \;=\; \underbrace{\frac{m^* - m_0}{\nu h_0}}_{\approx\,51\ \text{(captures are rare)}} \cdot \underbrace{\zeta e}_{\approx\,17\ \text{(paralysis throttles them)}} \;\approx\; 851 .
$$

## Limitations

* The model has no winter. Above about six hornets it predicts a negative store balance over a season, which I read as
  a colony that goes into winter light. It does not predict the winter collapse itself.
* Thresholds in foraging activity are properties of the model. Thresholds in hornets additionally pass through one fitted
  exponential and move by about one hornet across the digitisation spread of the field data.
* Every threshold in $p$ scales with forage quality as $1/c$. The same hornet load can be survivable on a rich site and
  lethal on a poor one.

## Repository

| File | Content |
|---|---|
| [`thesis_notebook.ipynb`](thesis_notebook.ipynb) | Chapter-by-chapter walk-through: every central number is recomputed and compared with the thesis, every figure is produced. GitHub shows it with all outputs. |
| [`thesis_code.py`](thesis_code.py) | The library: model, equilibrium cascade, thresholds, stability, simulations with rationing, delay-model solver, figures and a verification report with 527 checks. |
| [`figs/`](figs) | All figures as PDF (for LaTeX) and PNG. |
| [`verification_report.txt`](verification_report.txt) | Output of the full verification against the printed values of the thesis. |

### Running the code

You need Python 3.9 or newer with `numpy`, `scipy`, `matplotlib` and Jupyter (all included in Anaconda).

```bash
pip install -r requirements.txt
```

Then open `thesis_notebook.ipynb` in Jupyter or VS Code and choose *Run All* (about one minute).
Alternatively, from a terminal:

```bash
python thesis_code.py --quick    # all checks and figures, about one minute
python thesis_code.py --list     # available check sections and figures
```

## References

* Khoury, D. S., Myerscough, M. R. and Barron, A. B. (2011). A quantitative model of honey bee colony population
  dynamics. *PLoS ONE* 6(4): e18491.
* Khoury, D. S., Barron, A. B. and Myerscough, M. R. (2013). Modelling food and population dynamics in honey bee
  colonies. *PLoS ONE* 8(5): e59084.
* Requier, F. et al. (2019). Predation of the invasive Asian hornet affects foraging activity and survival probability
  of honey bees in Western Europe. *Journal of Pest Science* 92: 567–578.

The full bibliography is in the thesis.

## Citation

```bibtex
@thesis{fritzler2026foraging,
  author = {Fritzler, Anton},
  title  = {Foraging Paralysis as a Collapse Mechanism: A Bifurcation Analysis of Honey Bee Colony
            Dynamics under Predation by Vespa velutina},
  type   = {Bachelor's thesis},
  school = {Technical University of Munich},
  year   = {2026}
}
```

## License

The code is released under the MIT License (see [`LICENSE`](LICENSE)). The thesis text and figures remain © Anton Fritzler.
