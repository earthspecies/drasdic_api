# DRASDIC
Domain Randomization for Animal Sound Detection in Context

# Setup

Readme assumes everything is run with `uv`. In particular, requirements are in `pyproject.toml`

## Inference API

Download model and cfg file: 

```
mkdir weights; cd weights
gsutil -m cp \
  "gs://fewshot/drasdic_weights/main_model/random_9/args.yaml" \
  "gs://fewshot/drasdic_weights/main_model/random_9/model_80000.pt" \
  .
cd ..
```

In `args.yaml`, change `previous_checkpoint_fp` from `null` to `/absolute/path/to/weights/model_80000.pt`.

Download example audio file and selection table:

```
gsutil -m cp \
  "gs://fewshot/evaluation/formatted/marmoset/selection_tables/20160907_Twin1_marmoset1.txt" \
  "gs://fewshot/evaluation/formatted/marmoset/audio/20160907_Twin1_marmoset1.wav" \
  .
```

For example usage, see `uv run demo.py`. For API details, see `drasdic/inference/interface.py`.

## Data Generator

TODO

## Evaluation (FASD13)

TODO
