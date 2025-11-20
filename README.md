# pillar-pretrain

This repository contains the pretraining code for the Pillar-0 model.

## Installation

We rely on [uv](https://github.com/astral-sh/uv) for dependency management. Install it (once) and sync dependencies inside the repo:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
uv add flash-attn --no-build-isolation
source .venv/bin/activate

# Install rad-vision-engine
git clone https://github.com/yalalab/rad-vision-engine ../rad-vision-engine
cd ../rad-vision-engine && git checkout release
cd ../pillar-pretrain
uv pip install -e ../rad-vision-engine
```

## Data + Text Cache
### Vision cache generation for Merlin-Abd-CT
```bash
uv run vision-engine process --config ../rad-vision-engine/configs/ct_abdomen.yaml --input-series-csv data/merlin/accessions.csv --output data/merlin/merlin_cache_1.5mm --workers 128
```

### Text cache generation
```bash
mkdir -p data/text_cache_qwen3_embedding_8b
uv run python scripts/cache_merlin_embeddings.py --model-name Qwen/Qwen3-Embedding-8B --json-file data/merlin/merlin_abd_ct/train.json --cache-dir data/text_cache_qwen3_embedding_8b/train --batch-size 32 --num-processes 8
uv run python scripts/cache_merlin_embeddings.py --model-name Qwen/Qwen3-Embedding-8B --json-file data/merlin/merlin_abd_ct/valid.json --cache-dir data/text_cache_qwen3_embedding_8b/val --batch-size 32 --num-processes 8
```

### Using pre-generated cache

You can download the cache from our hugginface dataset repo.
Update the paths inside `src/miniclip/data_configs/pillar0_merlin_abd_ct_384.yaml` so they point at your Merlin ABD-CT manifest, RVE cache, and text cache root. The default configuration expects a text cache at `./data/text_cache_qwen3_embedding_8b/`.

## Launch Training

All training happens through the provided script:

```bash
bash train_unimodal_pillar0_merlin_abd_ct_1node.sh
```

# Citation
If you use this code in your research, please cite the following paper:

```
@article{pillar0,
  title   = {Pillar-0: A New Frontier for Radiology Foundation Models},
  author  = {Agrawal, Kumar Krishna and Liu, Longchao and Lian, Long and Nercessian, Michael and Harguindeguy, Natalia and Wu, Yufu and Mikhael, Peter and Lin, Gigin and Sequist, Lecia V. and Fintelmann, Florian and Darrell, Trevor and Bai, Yutong and Chung, Maggie and Yala, Adam},
  year    = {2025}
}
```
