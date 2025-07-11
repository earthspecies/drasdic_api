# DRASDIC

This is the public repo for DRASDIC: Domain Randomization for Animal Sound Detection in Context. 

## Quick links:

- Preprint (LINK TO COME)
- [Appendix](https://github.com/user-attachments/files/21130954/appendix_7_8.pdf)
- Model weights (LINK TO COME)
- [Fewshot Animal Sound Detection 13 (FASD13) evaluation dataset](https://zenodo.org/records/15843741)

![figures_v1 003(1)](https://github.com/user-attachments/assets/f8cac62a-4721-4383-bac2-3a10dafb87b1)

## Setup

We manage packages with `uv`. In particular, requirements are in `pyproject.toml`

# Inference API

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

# Evaluation: Fewshot Animal Sound Detection 13 (FASD13)

[Obtain the dataset here!](https://zenodo.org/records/15843741)

[Appendix with more details](https://github.com/user-attachments/files/21130954/appendix_7_8.pdf)

<img width="2400" height="1200" alt="example_audio_compact(6)" src="https://github.com/user-attachments/assets/9ddd9508-faac-4ad7-90a9-e14e38128d55" />

Fewshot Bioacoustic Sound Event Detection (FSBSED) describes the task of detecting animal sounds in recordings based on only a handful of examples. It is of interest to researchers in ecology, animal behavior, and machine learning.

A collection of public FSBSED datasets was previously provided in [Nolasco et al., 2023](https://www.sciencedirect.com/science/article/pii/S157495412300287X) and [Liang et al., 2024](https://ieeexplore.ieee.org/document/10714948?signout=success), but were designated as datasets for model training and validation. We complement these with Fewshot Animal Sound Detection 13 (FASD13), a public benchmark to be used for model evaluation. FASD13 consists of 13 bioacoustics datasets, each of which includes between 2 and 12 audio files. Eleven of these datasets were used from previous studies; they were chosen for their taxonomic diversity, varied recording conditions, and quality of their annotations. Two (CC and JS) are presented here for the first time. All datasets were developed alongside studies of ecology or animal behavior, and represent a range of realistic problems encountered in bioacoustics data. 

We follow the data format in [Nolasco et al., 2023](https://www.sciencedirect.com/science/article/pii/S157495412300287X): Each audio file comes with annotations of the onsets and offsets of positive sound events, i.e. sounds coming from a predetermined category (such as a species label or call type). An N-shot detection system is provided with the audio up through the Nth positive event, and must predict the onsets and offsets of positive events in the rest of the recording. Evaluation of N-shot detection systems is described in loc. cit.

**FASD13 Summary** 

| Dataset | Full Name       | N files | Dur (hr) | N events | Recording type | Location            | Taxa                                           | Detection target        |
|---------|------------------|---------|----------|----------|----------------|---------------------|------------------------------------------------|--------------------------|
| AS     | AnuraSet         | 12      | 0.20     | 162      | T. PAM         | Brazil              | Anura                                          | Species                  |
| CC     | Carrion Crow     | 10      | 10.00    | 2200     | On-body        | Spain               | Corvus corone + Clamator glandarius           | Species + Life Stage     |
| GS     | Gunshot          | 7       | 38.33    | 85       | T. PAM         | Gabon               | Homo sapiens                                   | Production Mechanism     |
| HA     | Hawaiian Birds   | 12      | 1.10     | 628      | T. PAM         | Hawaii, USA         | Aves                                           | Species                  |
| HG     | Hainan Gibbon    | 9       | 72.00    | 483      | T. PAM         | Hainan, China       | Nomascus hainanus                              | Species                  |
| HW     | Humpback Whale   | 10      | 2.79     | 1565     | U. PAM         | North Pacific Ocean | Megaptera novaeangliae                         | Species                  |
| JS     | Jumping Spider   | 4       | 0.23     | 924      | Substrate      | Laboratory          | Habronattus                                    | Sound Type               |
| KD     | Katydid          | 12      | 2.00     | 883      | T. PAM         | Panamá              | Tettigoniidae                                  | Species                  |
| MS     | Marmoset         | 10      | 1.67     | 1369     | Laboratory     | Laboratory          | Callithrix jacchus                             | Vocalization Type        |
| PM     | Powdermill       | 4       | 6.42     | 2032     | T. PAM         | Pennsylvania, USA   | Passeriformes                                  | Species                  |
| RG     | Ruffed Grouse    | 2       | 1.50     | 34       | T. PAM         | Pennsylvania, USA   | Bonasa umbellus                                | Species                  |
| RS    | Rana Sierrae     | 7       | 1.87     | 552      | U. PAM         | California, USA     | Rana sierrae                                   | Species                  |
| RW    | Right Whale      | 10      | 5.00     | 398      | U. PAM         | Gulf of St. Lawrence| Eubalaena glacialis                            | Species                  |
