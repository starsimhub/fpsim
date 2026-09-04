# Documentation audit report

FPsim, branch `rc3.6-port`, audited 2026-09-03 against IDM documentation standards
(completeness, Diátaxis topic types, persona targeting, Python docstrings).

**Status: most recommendations have since been actioned** -- see "Actions taken" at the
end for what changed and what is still outstanding.

## 1. Summary

FPsim's documentation is structurally sound and better than average for a research
model: a real Quarto site, five executable tutorials, near-complete API reference
coverage, and unusually good honesty about the model's limits. It has one blocking
defect and one systemic gap. The blocking defect is that the v3.6.0 refactor left
stale API calls in a tutorial and two examples, which will fail the docs build. The
systemic gap is the absence of any how-to or explanation layer: everything is either
a tutorial or API reference, so a user who finishes the tutorials and gets stuck has
nowhere to go.

## 2. Strengths

- **The README's caveats section is genuinely excellent.** "FPsim is not a replacement
  for good data", "cannot predict exogenous events", and the prompt to check whether
  descriptive statistics would answer the question are exactly the guidance the
  model-user persona needs, and most modeling repos omit it entirely.
- **API reference coverage is essentially complete.** Every module except `version` is
  listed in the `quartodoc` config, with `parser: google` and `render_interlinks: true`
  correctly set.
- **`fpmod.py` came through the refactor well documented** — 31 of 32 public objects
  have docstrings and 15 carry `Args:`, the best ratio in the package. The rewrite did
  not degrade the docs.
- **Tutorials are executable `.qmd` with `error: false`**, so breakage surfaces as a
  build failure rather than silently rotting. This is the setup the standards ask for.
- All required repo-level files are present: LICENSE, CHANGELOG, CONTRIBUTING,
  CODE_OF_CONDUCT.

## 3. Weaknesses

### Blocking

- **Stale post-refactor API will fail the docs build.** `docs/tutorials/T5_new_method.qmd:124`
  calls `sim_split.connectors.fp`, which now raises `AttributeError` — FPmod moved to
  `sim.people.fp` in v3.6.0. The cell is executed at build time, and no `_freeze` cache
  is committed, so a clean build fails here. Same breakage in
  `examples/example_add_method.py:295,428` and `examples/example_dmpasc.py:174,386`.
  Note `sim.connectors.contraception` is still correct — only `.fp` moved, so this
  cannot be fixed with a blanket find-and-replace.

### Significant

- **No user guide layer (Diátaxis gap).** Documentation is tutorials + API reference
  only. There are no how-to topics and no explanation topics. Model-extenders and
  model-builders — the personas who need extension points, calibration guidance under
  sparse data, and architectural rationale — have nothing between a beginner tutorial
  and a raw API signature. For a model with a calibration system, nine locations, and
  a pluggable contraception module, this is the largest structural gap.
- **README has no quick usage example.** IDM standards require a minimal "hello world"
  in under 10 lines. A reader currently cannot see what FPsim code looks like without
  leaving the repo. Low effort, high impact.
- **Docstrings are broad but shallow.** 78% of public objects have a docstring, but only
  19% document `Args:` and only 7% carry an `Example:`. The standard asks for an example
  on every public class and non-trivial function. Weakest modules: `education.py` (9/15
  documented), `interventions.py` (14/25), `people.py` (12/18). `plotting.py` has 22/22
  docstrings but only one `Args:` — present but not useful.
- **Vale is not configured.** No `vale.ini` and no `.github/styles`, so the grammar and
  style gate the standards expect cannot run at all.

### Minor

- **`tutorials.md` is a hand-maintained duplicate of the sidebar.** It lists the same five
  tutorials already enumerated in `_quarto.yml`, so the two will drift. Per the TOC
  guidance it should instead be the Tutorials section's index page (first entry of the
  section), not a parallel list reachable only from the navbar.
- **T2–T5 have no `title:` in frontmatter**, so nav labels fall back to the first heading.
  The TOC guidance asks for explicit human-readable labels.
- **T1 front-loads installation instructions** (`git clone`, `pip install -e .`) before the
  lesson starts. Installation is a how-to, not a tutorial step; per the TOC guidance it
  belongs on the home page or its own topic. It also duplicates the README, with a
  different method (clone vs `pip install fpsim`).
- **README's "User guide" heading is misleading** — the section is a limitations and
  appropriate-use discussion, not a guide. Good content, wrong label.
- **README and `docs/index.md` both open repo-first** ("This repository contains the code
  for..."). The policy-maker persona guidance is explicit that this needs only one or two
  sentences of real-world relevance, not a reframed landing page — so this is a small,
  bounded fix rather than a rewrite.

## 4. Recommendations

Ranked by impact, then effort.

1. **Fix the stale `connectors.fp` references.** T5 line 124, `example_add_method.py`
   (2 sites), `example_dmpasc.py` (2 sites) → `sim.people.fp`. Leave
   `connectors.contraception` alone. Unblocks the docs build. **Effort: Low.**
2. **Add a build check for the examples.** The tutorials execute at build time but
   `examples/` does not, which is why two example files silently broke during the
   refactor. A smoke test that imports and runs each example would have caught this.
   **Effort: Low.**
3. **Add a quick usage example to the README** — a <10-line create-run-plot snippet
   directly under the intro. Required by IDM standards; the single highest-value README
   change. **Effort: Low.**
4. **Make `tutorials.md` the Tutorials section index** and give T2–T5 explicit `title:`
   frontmatter. Removes the drift risk between the two tutorial lists. **Effort: Low.**
5. **Move installation out of T1** into the home page or an Installation topic, and
   reconcile it with the README's `pip install fpsim`. **Effort: Low.**
6. **Configure Vale** with `vale.ini` and the IDM styles in `.github/styles`, and add it
   to CI. **Effort: Low.**
7. **Add `Example:` sections to public classes and non-trivial functions**, starting with
   the modules users touch first: `interventions.py`, `methods.py`, `analyzers.py`.
   Prioritize the 81% of objects missing `Args:` in those three. **Effort: High**, but
   splits cleanly per module.
8. **Create a user guide layer.** Start with two subject-matter guides that match how
   FPsim is actually used — a Calibration guide (explanation parent + how-tos for
   `calibrate_all.py`, adding a location, interpreting mismatch) and a Contraception
   guide (method definitions, `method_mix`, adding a method). Per IDM TOC conventions
   these are grouped by subject with the explanation topic as parent, not split into
   separate how-to and explanation trees. **Effort: High.**
9. **Rename the README's "User guide" heading** to something accurate such as "When to
   use FPsim" or "Scope and limitations", and add one or two sentences on policy
   relevance at the very top. **Effort: Low.**

## 5. Completeness checklist

| Component | Status | Notes |
|---|---|---|
| README.md | Partial | No quick usage example; "User guide" heading mislabeled |
| LICENSE | Present | MIT |
| Changelog | Present | `CHANGELOG.md`, 3.6.0 entry current |
| Contributing guide | Present | `CONTRIBUTING.md` |
| Code of conduct | Present | `CODE_OF_CONDUCT.md` + `docs/conduct.md` |
| Folder-level READMEs | Present | `fpsim/locations`, `fpsim/data_processing`, `docs`, `.github/workflows` |
| Docs site config | Present | `docs/_quarto.yml`; quartodoc correctly configured |
| Hello-world tutorial | Partial | T1 exists but opens with installation rather than the lesson |
| Advanced tutorial(s) | Present | T2–T5 cover features, interventions, eligibility, new methods |
| User guide | **Missing** | No how-to or explanation topics anywhere |
| API reference | Present | All modules but `version` listed; auto-generated |
| Diataxis coverage | Partial | Tutorials + reference only; two of four types absent |
| Persona targeting | Partial | Model-user well served; extender/builder underserved |
| Docstring quality | Partial | 78% coverage, 19% `Args:`, 7% `Example:` |
| Style linting (Vale) | **Missing** | No `vale.ini` or `.github/styles` |


## 6. Actions taken (2026-09-03)

Done:

1. **Stale API fixed** -- T5 and four example files moved off `sim.connectors.fp`; the docs
   site now renders clean end to end, which it did not before.
2. **README** -- added the missing quick-start, led with what FPsim does rather than "this
   repository contains", renamed the mislabeled "User guide" section.
3. **Installation** -- split out of T1 into its own how-to.
4. **Tutorials** -- `tutorials/index.qmd` is now the section landing page; T2-T5 have
   explicit titles.
5. **User guide** -- added Calibration and Contraception guides, each an explanation parent
   with how-tos beneath, closing the Diátaxis gap. The recalibration how-to is promoted
   from `fpsim/locations/CALIBRATION.md`, which users could not see.
6. **Docstrings** -- Args and runnable Examples added to the user-facing classes in
   `methods.py`, `analyzers.py` and `interventions.py`. Example coverage on those modules
   went from 8 to 20 objects. Package-wide: coverage 79%, Args 22%, Example 10%.
7. **Examples** -- all eight scripts fixed and now covered by `tests/test_examples.py`.

Deliberately not done:

- **Vale** -- `idm_standards` ships no `vale.ini` or style bundle, so this cannot be
  satisfied by copying a template. Choosing a voice bundle (Microsoft, Google, custom) is a
  team decision, and dropping one in would flag existing prose wholesale. Needs a decision
  before it can be configured.

Still outstanding:

- **Docstring depth** -- Args and Example coverage remain low package-wide (22% and 10%).
  The three user-facing modules are done; `plotting.py` (22 docstrings, 1 Args) and
  `experiment.py` are the next most valuable.
- **Broken analyzers** -- `track_as` (public), `track_parity` and `track_postpartum` assign
  a plain dict to `self.results`, which Starsim 3.6.0 locks, so they raise on init. They
  need porting to `ss.Results`.
- **Docs build needs network** -- `quartodoc interlinks` fetches `objects.inv` from
  scipy/numpy at build time and fails hard if unreachable; one render here failed on a
  connection timeout and only succeeded on retry. Worth caching the inventories, given how
  much the persona guidance emphasizes low-connectivity users.
