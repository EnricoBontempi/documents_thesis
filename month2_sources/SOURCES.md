# Sources of the Month 2 report

All files in `data/` are byte copies of tables published in `EnricoBontempi/official_thesis`,
except the two subsets marked *filtered* (rows selected, no value changed).
No new training, inference or TEST evaluation was performed for the report.

| Local copy (`month2_sources/data/...`) | Commit in official_thesis | Original path |
|---|---|---|
| `free_s7/{global_by_setting,per_condition,training_costs_by_run,training_costs_by_attempt,eval_costs_by_task,run_matrix,coverage_degradation,selection_shift}.csv` | `2879c1b5270be30fb10375f0b300a50cd37d5413` | `results/month2/s7_c149_comparison/package/tables/` |
| `free_s7/{baselines_c149_macro,swap_neural_by_run,swap_neural_family_p_seed_dispersion,swap_neural_by_condition}.csv` | `2879c1b5…` | `results/month2/s7_c149_comparison/residuals/tables/` |
| `fields/fields_val_frame1000_best.png`, `fields/field_summary.json` | `2879c1b5…` | `results/month2/s7_c149_comparison/fields/` |
| `k_sweep/by_seed.csv`, `k_sweep/checks.json` | `be18ad007e66fbf2565ad3d5486c6513fc5c3e4c` | `results/month2/temperature_k_sweep/` |
| `k_sweep/per_condition_val_best_primary.csv` (*filtered*: role VAL, checkpoint best/fitted, primary layouts) | `be18ad00…` | `results/month2/temperature_k_sweep/per_condition.csv` |
| `joint/{joint_by_run_replica_K,joint_by_seed_K,joint_across_seeds_K,assisted_checkpoint_epochs}.csv` | `d20232ef04873e94708995d119bedc2f6eca2f50` | `results/month2/joint_step23/package/tables/` |
| `assisted_v2/{gain_vs_K0_by_run,global_by_setting,completeness,training_costs_by_run,eval_costs_by_task,free_S7_reference_vs_assisted_K0}.csv` | `d20232ef…` | `results/month2/temperature_assisted_c149/package_v2/tables/` |
| `assisted_v2/per_condition_val_best.csv` (*filtered*: role VAL, checkpoint best) | `d20232ef…` | `results/month2/temperature_assisted_c149/package_v2/tables/per_condition.csv` |
| `data_quantity/global_by_setting.csv` | `4f0e2637758de0766d0bb0891fe8ff97eee978c9` | `results/month2/…/o5/package/tables/` (data-quantity package of report 06) |
| `anchors/month2_temperature_anchors_nested_v1.json` (file SHA256 `a9c997c9…`) | `d20232ef…` | `configs/observations/` |

## Figures and tables of `main.tex`

| Report item | Built from |
|---|---|
| Fig. anchor layouts (`figures/month2/anchor_layouts.pdf`) | anchor manifest; background = truth panel cropped from `fields_val_frame1000_best.png` |
| Fig. data quantity | `data_quantity/global_by_setting.csv` |
| Fig. free comparison, Tab. free summary, Tab. per seed | `free_s7/global_by_setting.csv`, `free_s7/run_matrix.csv`, `free_s7/baselines_c149_macro.csv` |
| Fig. fields (`fields_level_structure.png`) | crops of `fields/fields_val_frame1000_best.png` (scales: `field_summary.json`) |
| Fig. VAL conditions, Tab. free per condition | `free_s7/per_condition.csv` |
| Fig. RMSE vs K, Tab. joint | `joint/joint_by_seed_K.csv` (free, mean, median, assisted); `k_sweep/by_seed.csv` (AO, oracle) |
| Fig. K=1 by layout | `joint/joint_by_run_replica_K.csv` |
| Fig. assisted gain | `assisted_v2/gain_vs_K0_by_run.csv` |
| Tab. regional | `k_sweep/by_seed.csv`, `assisted_v2/global_by_setting.csv` |
| Tab. per condition vs K | `k_sweep/per_condition_val_best_primary.csv`, `assisted_v2/per_condition_val_best.csv` |
| Tab. baselines vs K | `k_sweep/by_seed.csv` |
| Tab. costs, Tab. run matrix | `free_s7/training_costs_by_run.csv`, `assisted_v2/training_costs_by_run.csv`, `assisted_v2/completeness.csv`, `joint/assisted_checkpoint_epochs.csv` |

Values quoted in the text that are not in these tables (dense diagnostic, 38-frame nine-site
diagnostic, leave-one-condition-out selection gap, GLU batch-tie diagnostic, evaluation gates)
come from the project reports `docs/month2/03`, `07` (`82bff35`), `08` (`7cb7bfd`), `06` (`4f0e263`),
`10` (`2879c1b`), `11` (`be18ad0`) and `12` (`d20232e`).

Regenerate with `python month2_sources/make_figures.py` (pandas, matplotlib, pillow).
