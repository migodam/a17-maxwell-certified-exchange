# Scoring, candidate coverage and path search

**The full-dictionary anchor score is accurate in object-aggregated common-checkpoint comparisons. The largest missed gains are outside the online short-pool.**

|Geometry / material|Dictionary anchor top 1|Pool best opportunity|Pool anchor top 1|
|---|---:|---:|---:|
|Smooth / Gaussian|.995995|.968926|.964089|
|Near-contact / Gaussian|.981000|.352819|.345634|
|Layered / voxel|.993711|.919810|.915063|
|Asymmetric / voxel|.988332|.658924|.638713|

Each ratio sums gains over the same object's nine common receiver checkpoints before division by summed best nonnegative opportunity. It is not a lower bound at every checkpoint. For example, smooth-middle-k16 mandatory anchor top-one has negative true gain of −4.77416e−7, while the teacher best is +2.44314e−7. Directed deployment rejects negative moves; a no-op is available.

Full-dictionary scoring is offline diagnosis, not online twelve-candidate cost. Three-action teacher versus verified path gain capture is .897960, .433679, .851805 and .695099 for smooth, near-contact, layered and asymmetric geometries. That descriptive path comparison is not score-only regret.

## Controlled direction/evaluation comparison

Identical initial-pool endpoints are crossed by direction v/d and linear/quadratic evaluation. All ranking choices execute actual endpoint displacement d.

|Reference|v-linear|v-quadratic|d-linear|d-quadratic|
|---|---:|---:|---:|---:|
|Complete|24|18|29|1|
|Anchor|26|20|30|2|

Entries count negative complete gains out of 36 mandatory top-one choices. No-op is excluded only in this diagnostic, included in deployment. It resolves finite-displacement effects without assigning old entire-path ratios to curvature alone.

Sources: analysis/same_checkpoint_score_regret.csv, direction_curvature_rank_choices.csv, object_summed_ratios.csv and immutable move records.

