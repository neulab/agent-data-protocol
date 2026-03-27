# LLaVA Plus

- **Original Paper**: [LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents](https://arxiv.org/abs/2311.05437)
- **Original Repository**: [LLaVA-VL/LLaVA-Plus-Codebase](https://github.com/LLaVA-VL/LLaVA-Plus-Codebase)
- **HuggingFace Dataset**: [LLaVA-VL/llava-plus-data](https://huggingface.co/datasets/LLaVA-VL/llava-plus-data)
- **License**: See original repository
- **Size**: ~117k samples across 6 image sources

## Description

LLaVA Plus extends LLaVA with tool-use capabilities. The dataset consists of multi-turn conversations where an AI agent uses vision tools (segmentation, inpainting, detection, etc.) on images from 6 source datasets.

## Image Sources

The dataset references images from 6 external sources that must be downloaded separately:

| Source | Examples | Unique Images | Download Size | Auth Required |
|--------|----------|--------------|---------------|---------------|
| coco | 68,468 | ~41,815 | ~10 GB | No |
| vg (Visual Genome) | 22,814 | ~16,567 | ~4 GB | No |
| instruct-pix2pix | 11,317 | ~6,745 | ~420 GB streamed | No |
| hiertext | 6,398 | ~6,398 | ~3.6 GB | No |
| journeydb | 4,451 | ~4,451 | ~3 TB streamed | Yes (HF) |
| infoseek (OVEN) | 3,591 | ~3,585 | ~243 GB streamed | Yes (HF) |

Images are saved to `$DATA_DIR/llava_plus/images/{source}/` and referenced in standardized data as `images/{source}/{filename}`.

## Downloading Images

### All sources at once (SLURM)

```bash
sbatch slurm_download_llava_plus_images.sh
```

### Specific sources

```bash
sbatch slurm_download_llava_plus_images.sh coco vg hiertext
```

### Prerequisites

1. `full_raw.jsonl` must exist at `$DATA_DIR/llava_plus/full_raw.jsonl` (run extract stage first)
2. For journeydb: `huggingface-cli login` and accept terms at https://huggingface.co/datasets/JourneyDB/JourneyDB
3. For infoseek: `huggingface-cli login` and accept terms at https://huggingface.co/datasets/ychenNLP/oven

### Manual image list generation

If needed, generate image lists from `full_raw.jsonl`:

```bash
cat full_raw.jsonl | jq -r 'select(.data_source == "coco") | .image' | sort -u > coco_images_needed.txt
cat full_raw.jsonl | jq -r 'select(.data_source == "vg") | .image' | sort -u > vg_images_needed.txt
cat full_raw.jsonl | jq -r 'select(.data_source == "hiertext") | .image' | sort -u > hiertext_images_needed.txt
cat full_raw.jsonl | jq -r 'select(.data_source == "journeydb") | .image' | sort -u > journeydb_images_needed.txt
cat full_raw.jsonl | jq -r 'select(.data_source == "instruct-pix2pix") | .image' | sort -u > instruct-pix2pix_images_needed.txt
cat full_raw.jsonl | jq -r 'select(.data_source == "infoseek") | .image' | sort -u > infoseek_images_needed.txt
```

## Download Scripts

Each source has its own download script in this directory:

| Script | Source | Method |
|--------|--------|--------|
| `download_coco.py` | COCO train2017/val2017 | Direct URL with fallback |
| `download_vg.py` | Visual Genome VG_100K/VG_100K_2 | Direct URL with fallback |
| `download_hiertext.py` | Open Images OCR (S3) | Stream 3 tar archives |
| `download_journeydb.py` | JourneyDB (HuggingFace) | Stream 200 .tgz shards |
| `download_infoseek.py` | OVEN (HuggingFace) | Stream 8 tar shards |
| `download_instruct_pix2pix.py` | Berkeley server | Download+extract 30 zip shards |

All scripts use `download_utils.py` for shared functionality (exponential backoff, progress, error reporting).
