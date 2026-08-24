# SWS_SBI

For data and framework documentation, please see [pre-print under construction].

Data used in this framework, was first published by:
Resta, F., Allegra Mascaro, A. L., & Pavone, F. (2020). Study of Slow Waves (SWs) propagation through wide-field calcium imaging of the right cortical hemisphere of GCaMP6f mice [Dataset]. EBRAINS. https://doi.org/10.25493/3E6Y-E8G

---

Simulation-based inference (SBI) for a delayed Wilson–Cowan model of mouse cortex,
fitted to widefield calcium imaging recordings.

The project has three stages:

1. **Isocortex** — build the pixel → brain-region mapping from the Allen CCF atlas.
2. **Pipeline** — preprocess the raw widefield recordings and reduce them to one
   median time series per region.
3. **SBI** — fit the Wilson–Cowan model to those time series with sequential
   neural posterior estimation, then evaluate and plot the results.

---

## ⚠️ Run everything from the project root

Every script resolves its paths relative to `./` (`PROJECT_ROOT = Path("./")`,
`config.ROOT = Path("./")`). Running a script from inside its own folder will
create or look for `data/` in the wrong place.

```bash
cd /path/to/SWS-SBI
python src/isocortex/pixel_brain_mapping.py   # correct
```

```bash
cd src/isocortex && python pixel_brain_mapping.py   # wrong
```

Note the two different invocation styles, they are not interchangeable:

| Stage | How to run | Why |
|---|---|---|
| `src/isocortex/*`, `src/pipeline/*` | `python src/<folder>/<script>.py` | plain imports, resolved via the script's own directory |
| `src/sbi/*` | `python -m src.sbi.<module>` | relative imports (`from .sbi_utils import config`, `from ...fitness import ...`) require package context |

---

## ⚠️ Two separate environments are required

`allensdk` pins old versions of numpy/pandas/scipy and conflicts with the
`torch` + `jax` + `sbi` stack. Keep them apart:

**Environment A — `isocortex` and `pipeline`**

```
allensdk, pynrrd, scikit-image, scipy, numpy, pandas, plotly,
h5py, oasis (OASIS deconvolution), imageio[ffmpeg], Pillow, openpyxl
```

**Environment B — `sbi`**

```
torch, jax[cuda|cpu], sbi, numpy, pandas, matplotlib, openpyxl
```

Example:

```bash
conda create -n sws-isocortex python=3.10 && conda activate sws-isocortex
pip install allensdk pynrrd scikit-image plotly h5py oasis-deconv imageio[ffmpeg] Pillow openpyxl

conda create -n sws-sbi python=3.11 && conda activate sws-sbi
pip install torch sbi "jax[cpu]" matplotlib pandas openpyxl
```

---

## Input data

The recordings and connectivity tables are already in the repository under
`data/`:

```
data/
├── raw_dataset/
│   ├── 1/{1,2,3}.h5          # dataset key: video_mouse0_trial{0,1,2}, shape (1000, 100, 100)
│   └── 2/{1,2,3}.h5          # dataset key: video_mouse1_trial{0,1,2}
└── tables/
    ├── atlas.xlsx            # sheets: "connectivity", "distances"
    └── local.xlsx            # sheets: "connectivity", "distances"
```

The Allen CCF reference space is **not** in the repository. It is downloaded
automatically by the AllenSDK the first time `pixel_brain_mapping.py` runs, so
that first run needs an internet connection and takes a while. It lands in:

```
data/allen_data/25_micron_resolution/
├── manifest.json
├── structures.json
└── ccf_2017/annotation_25.nrrd
```

---

## Recreating the main experiment

Run these three in order. Each step consumes the previous step's output.

### 1. Pixel → region mapping (environment A)

```bash
python src/isocortex/pixel_brain_mapping.py
```

Flattens the Allen CCF annotation volume to a dorsal view, crops it to the
imaging field, separates the regions with a Sobel edge filter, and downsamples to
the 100×100 imaging grid — once per mouse (mouse 2 is shifted up by `crop_factor`
rows to correct for alignment).

Writes:

```
data/pixel_brain_mappings/pixel_brain_map_mouse1.npy   # (100, 100) int region ids
data/pixel_brain_mappings/pixel_brain_map_mouse2.npy
data/pixel_brain_mappings/id_acronym_lookup.csv        # region id -> acronym
data/pixel_brain_mappings/unique_colormap.csv
data/allen_data/25_micron_resolution/ccf_2017/structure_<id>.nrrd   # mask cache
```

On the first run this also downloads the Allen CCF reference space into
`data/allen_data/25_micron_resolution/`; later runs reuse it.

Set `show_2d=True` in `main()` for interactive hover plots of each intermediate
flat map. Pass `save=False` for a dry run that prints the paths it would write.

### 2. Preprocessing pipeline (environment A)

```bash
python src/pipeline/generate_pipeline.py
```

Loads the six recordings (2 mice × 3 trials) and applies the pipeline defined at
the bottom of `main()`. The default is pipeline **2**:

```python
["mask", "dFF", "oasis", "bandpass", "minmax_norm"]
```

i.e. brain-contour masking → ΔF/F with an airPLS baseline → OASIS denoising and
deconvolution → 0.5–4 Hz bandpass → min–max normalisation. Other valid steps:
`sub_background`, `detrend`, `zscore_norm`, `denoise` (PCA), `fft_deconv`.

It then collapses each region to the median over its pixels.

Writes:

```
data/pixel_brain_mappings/contour_mask{1,2}.npy   # brain outline per mouse
data/pipelines/2.npy                              # (2, 3, 1000, 100, 100) pixel-space
data/median_pipelines/2.npy                       # (2, 3, n_regions, 1000) region-space
videos/<i>_<StepLabel>.mp4                        # one per step, if save_video=True
videos/<i>_Median.mp4                             # region-median reconstruction
```

Toggle `generate_pipeline`, `save_video`, `generate_median_pipeline` and
`save_median_video` in `main()` to re-run only part of this.

**`data/median_pipelines/<PIPELINE>.npy` is the observation the SBI stage fits.**

### 3. Inference (environment B)

```bash
python -m src.sbi.run_sbi
```

Runs `NUM_ROUNDS` (25) rounds of SNPE with `SAMPLES_PER_ROUND` (10 000)
simulations each, over the 14 Wilson–Cowan parameters in
`config.PARAMETER_SPACE`. Each simulation runs the JAX model (`src/wc_model.py`),
selects the regions present in the mouse mapping, and reduces the output to the
summary vector defined in `src/sbi/sbi_utils/summary.py` (FC, delayed FC at
τ ∈ {2, 3, 5}, per-region PSD over 0.5–4 Hz, region-averaged PSD, value
histogram). After each round the posterior is truncated with a density
thresholder and used as the proposal for the next.

Writes one pickle per round:

```
data/sbi/pkls/posterior_round_<r>_mouse<m>_trial<t>.pkl
```

Key knobs in `src/sbi/sbi_utils/config.py`: `MOUSE_IDX`, `TRIAL_IDX`, `PIPELINE`,
`NUM_ROUNDS`, `SAMPLES_PER_ROUND`, `LEARNING_RATE`, `PARAMETER_SPACE`,
`DFC_DELAYS`, `N_BINS`, and the frequency grid.

> `src/wc_model.py` uses `N = 40` regions and only filters the connectivity
> tables down to the mapped regions when `N == 12`. Keep `N` consistent with the
> tables you are using.

---

## Producing the plots

### Posterior pairplot + MAP estimate (environment B)

```bash
python -m src.sbi.pairplot
```

Overlays prior and posterior samples for a given round. Change the round in the
`__main__` block (`round=1` by default; use `round=25` for the final posterior).
Prints the MAP estimate as a ready-to-paste `theta` dict.

→ `figures/pairplots/round_<round>.png`

### Posterior-predictive MSE (environment B)

Two steps — compute, then plot.

```bash
python -m src.sbi.mse        # draws 100 posterior samples per round, simulates each
python -m src.sbi.mse_plot   # renders the curves
```

`mse.py` compares each simulation against the recording on all five summary
statistics and saves them per round. The `theta` dict in its `__main__` block is
the "manual" reference parameter set, drawn as a ★ on the plots; pass
`manual_theta=None` to skip it. Results are re-saved after every round, so the
plotting step works on a partial run.

→ `data/mse/mse_mouse<m>_trial<t>.npz`
→ `figures/predictive_checks/{fc_mse, dfc_mse, psds_mse, avg_psd_mse, hist_mse}.pdf`

`mse_plot.main()` options: `sample_style="band"` (mean ± std) or `"scatter"`
(every sample as a dot), `log_y`, `scatter_jitter`, and `latex_table=True` to
print a LaTeX table of the same numbers.

### Empirical vs. simulated comparison figures (environment B)

```bash
python -m src.sbi.plot_individual
```

Simulates one parameter vector (the `theta` dict in `__main__` — paste the MAP
estimate from `pairplot` here) and saves side-by-side figures against the
recording.

→ `figures/summary_statistics_and_timeseries/` (set by `out_dir` in `__main__`;
the default when `out_dir=None` is `data/compare/figures_mouse<m>_trial<t>/`)

```
timeseries_empirical.png      timeseries_simulated.png
fc_empirical.png              fc_simulated.png              fc_delta.png
dfc_tau{2,3,5}_empirical.png  dfc_tau{2,3,5}_simulated.png  dfc_tau{2,3,5}_delta.png
psd_empirical.png             psd_simulated.png             psd_delta.png
psd_average.png               value_distribution.png
```

### Per-pixel time series through the pipeline (environment A)

Two steps — dump the intermediates, then plot.

```bash
python src/pipeline/generate_pipeline_in_steps.py
python src/pipeline/pixel_timeseries.py
```

`generate_pipeline_in_steps.py` re-runs the preprocessing from raw data once per
prefix of the pipeline (`["mask"]`, `["mask", "dFF"]`, `["mask", "dFF", "oasis"]`,
…) and saves each cumulative result as its own array. It writes no videos and no
region medians — it exists purely to give the plotting script a snapshot after
every step.

→ `data/pipeline_in_steps/{0,1,2,3,4}.npy`   — each `(2, 3, 1000, 100, 100)`

`pixel_timeseries.py` then picks a reference pixel valid in both the contour mask
and the region map, and plots its trace at each of those stages. The `TITLES`
list at the top of the file labels the panels and must stay aligned with the
pipeline numbers above.

→ `figures/pixel_timeseries/{0,1,2,3,4}.png`

> Note that this recomputes ΔF/F and OASIS from scratch for every prefix, so it
> is roughly 5× the cost of a single `generate_pipeline.py` run.

### Connectivity and distance matrices (environment A)

```bash
python src/isocortex/plot_matrices.py
```

Heatmaps of the atlas connectivity and distance tables, with the paired value in
the hover text. By default both figures open in a browser. To write them instead,
call `main()` with `connectivity_html=` / `distance_html=` set to a path — `.html`
for interactive output, any image extension for a static one (needs `kaleido`).

### 3D isocortex surfaces (environment A)

```bash
python src/isocortex/visualize_isocortex.py
```

Renders every isocortical region as a 3D mesh inside a translucent brain outline,
with the four regions absent from the connectome (`SSp-un`, `AUDpo`, `VISli`,
`VISpor`) highlighted in red. Opens interactively; the surface stack is several
GB in memory. Call `fig.write_html(...)` on the returned figure to save it.

---

## Where everything is saved

```
data/
├── pixel_brain_mappings/     pixel→region maps, contour masks, id/colour lookups
├── allen_data/               Allen CCF (downloaded on first run) + .nrrd masks
├── pipelines/                <n>.npy  — full pixel-space preprocessed data
├── pipeline_in_steps/        <n>.npy  — one snapshot per cumulative pipeline step
├── median_pipelines/         <n>.npy  — per-region median traces (SBI target)
├── sbi/pkls/                 posterior_round_<r>_mouse<m>_trial<t>.pkl
├── mse/                      mse_mouse<m>_trial<t>.npz
└── compare/                  default output dir for plot_individual figures

figures/
├── pairplots/                round_<r>.png
├── predictive_checks/        *_mse.pdf
├── pixel_timeseries/         <step>.png
└── summary_statistics_and_timeseries/   empirical vs. simulated figures

videos/
├── <i>_<Step>.mp4            one grid video per pipeline step
└── <i>_Median.mp4            region-median reconstruction
```

---

## Source layout

```
src/
├── fitness.py                summary statistics: FC, delayed FC, PSD, histogram
├── wc_model.py               delayed Wilson–Cowan model (JAX), constants, table loading
├── isocortex/
│   ├── pixel_brain_mapping.py    builds the pixel→region maps  [step 1]
│   ├── plot_matrices.py          connectivity / distance heatmaps
│   ├── visualize_isocortex.py    3D region surfaces
│   └── utils/{allensdk,plotting}.py
├── pipeline/
│   ├── generate_pipeline.py      runs preprocessing + region medians  [step 2]
│   ├── generate_pipeline_in_steps.py   saves a snapshot after each step
│   ├── mouse_dataset.py          dataset container and all processing steps
│   ├── airPLS.py                 baseline estimation for ΔF/F
│   ├── pixel_timeseries.py       per-pixel traces per step
│   └── video.py                  grid video writer
└── sbi/
    ├── run_sbi.py                sequential SNPE  [step 3]
    ├── mse.py / mse_plot.py      posterior-predictive MSE + figures
    ├── pairplot.py               prior/posterior pairplot + MAP
    ├── plot_individual.py        empirical vs. simulated figures
    └── sbi_utils/
        ├── config.py             paths, parameter space, all hyperparameters
        ├── data.py               region mapping + observed summary loading
        ├── prior.py              box-uniform prior
        ├── simulator.py          batched JAX simulator ↔ torch bridge
        ├── summary.py            summary vector assembly
        └── inference.py          SNPE rounds + posterior saving
```

