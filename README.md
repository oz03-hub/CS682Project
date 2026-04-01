# BERT 2 Funnel Transformer Knowledge Distillation

Task-specific knowledge distillation from a fine-tuned BERT teacher to a custom Funnel Transformer student, skipping language-model pre-training entirely. Evaluated on three sentiment classification datasets.


## Architecture

### Teacher `cs682/models/teacher.py`

- **Base model:** `google/bert_uncased_L-8_H-512_A-8` (8-layer, 512-hidden BERT)
- Linear classification head on top of the final `[CLS]` token
- Exposes intermediate `[CLS]` hidden states at configurable layer indices for layer-wise distillation

### Student `cs682/models/student.py`

A custom encoder-only Funnel Transformer with a fixed **3-block** architecture:

| Block | Pooling | Sequence length |
|-------|---------|-----------------|
| 0     | none    | T               |
| 1     | ×2      | T / 2           |
| 2     | ×2      | T / 4           |

- The `[CLS]` token is kept separate and never pooled
- The first layer of each pooling block attends over the full (un-pooled) keys/values to minimise context loss
- Initialized from BERT's token and positional embeddings via `FunnelTransformer.from_bert()`

## Distillation Objective

The combined loss used in `StudentTrain.ipynb`:

```
L = α · L_task  +  β · L_logit  +  γ · L_layer
```

| Term | Description |
|------|-------------|
| `L_task` | Cross-entropy with ground-truth labels |
| `L_logit` | KL divergence between soft teacher/student outputs (temperature-scaled) |
| `L_layer` | MSE between teacher and student `[CLS]` hidden states at mapped block boundaries |

## Datasets

Raw CSV files live under `cs682/data/`.

## Repository Structure

```
.
├── TeacherTrain.ipynb      # Fine-tune BERT teacher on a chosen dataset
├── StudentTrain.ipynb      # Distil teacher into the Funnel Transformer student
├── models/                 # Trained models live here
└── cs682/
    ├── models/
    │   ├── teacher.py      # BERTTeacher module
    │   └── student.py      # FunnelTransformer module
    ├── data/
    │   ├── loader.py       # IMDBDataset, YelpDataset, AmazonDataset
    │   ├── imdb/
    │   ├── yelp/
    │   └── amazon/
    └── evaluator.py        # Accuracy, precision, recall evaluation
```

## Workflow
Same as CS682 HWs.

1. Clone the repository.
2. Ensure `.csv` files are present in `cs682/data/imdb` | `amazon` | `yelp`. If the files are missing for some reason, run `bash cs682/data.prepare.sh`, to unzip tar files.
3. After ensuring the data is present, upload the entire directory to Google Drive and start working from there.
4. You will need to uncomment initial cells just like the HW.
