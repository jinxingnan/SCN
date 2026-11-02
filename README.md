# SCN: Structurally Constrained Navigation

SCN combines language model representations with knowledge graph embeddings for multi-hop question answering. Its core components are entity soft-token projection, language-model encoding, navigation-vector projection, graph-constrained action scoring with a learnable STOP action, and constrained beam search. The implementation includes multi-positive contrastive objectives and training, inference, and evaluation entry points.

## Repository structure

```text
scn/                  Model, projections, search, losses, and evaluation utilities
embeddings/           KGE loading and candidate-vector fusion
data/                 JSONL data loaders
configs/              WebQSP and CWQ configuration presets
train_gln.py          Training entry point
run_inference.py      Beam-search inference
compute_metrics.py    Answer selection and evaluation metrics
```

## Setup

Install PyTorch for the target hardware, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root. In a configuration under `configs/`, set `llm_model_name`, `train_file`, `kge_embeddings_dir`, `kge_mapping_dir`, and `output_dir`. Relative paths are resolved from the working directory. Match `llm_hidden_size` and `kge_dim` to the model and embedding dimensions.

The KGE directories contain `entity_embeddings.pt`, `relation_embeddings.pt`, and the mappings `entity_to_id.tsv.gz` and `relation_to_id.tsv.gz`. Mapping files have tab-separated `id` and `label` columns with a header; IDs index the embedding rows. The vocabulary must cover the seed entities and all entities and relations in each input subgraph.

JSONL records contain `id`, `question`, `q_entity` (seed labels), `a_entity` (answer labels, with `answer` as a fallback), and `graph` (a list of `[head, relation, tail]` triples). Training also uses `ground_truth_paths`, a list of paths represented as lists of triples; inference accepts records without path annotations.

## Training

The presets use `val_file: null` to split the training records according to `train_ratio`. To use a separate development set, set `val_file` to its path. Reserve the test set for evaluation.

```bash
python train_gln.py --config configs/webqsp_optimized_v2.yaml
# CWQ:
python train_gln.py --config configs/cwq_optimized_v2.yaml
```

## Inference and evaluation

Use a checkpoint directory containing `config.yaml`, `best_model.pt`, and `lora_best/` when LoRA is enabled:

```bash
python run_inference.py \
  --checkpoint_dir output/YOUR_CHECKPOINT \
  --eval_file datasets/RoG-webqsp-test.jsonl \
  --beam_size 6 \
  --max_hops 3 \
  --inference_temperature 0.05 \
  --output_file results/webqsp_inference.json
```

For CWQ, select its evaluation file and use `--max_hops 4`. Set temperature with `--inference_temperature`; when calling the Python model interface directly, set `model.temperature`. Compute answer and path metrics from the saved predictions:

```bash
python compute_metrics.py \
  --inference_file results/webqsp_inference.json \
  --threshold_delta 0.8 \
  --output_file results/webqsp_metrics.json
```

`threshold_delta` controls the score margin for multi-answer selection; select it on development data. Each entry point provides further options through `--help`.
