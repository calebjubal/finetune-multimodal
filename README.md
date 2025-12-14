# finetune-multimodal

Fine-tuning a **vision-language model (Qwen2-VL-7B-Instruct)** with **LoRA** to read synthetic metro/MRT map images and produce the route between two stations that uses the **minimum number of line transfers**.

The model takes an image of a transit map plus a `source` and `target` station, and returns a structured JSON answer:

```json
{"transfer_count": 1, "route": "Forest,Delta"}
```

---

## Task

Given:
- An MRT/metro map image (colored lines, stations, interchange points)
- A `source` station and a `target` station

Predict:
- `transfer_count` — the minimum number of line changes needed
- `route` — the ordered list of stations from source to target

This is a **multimodal reasoning** problem: the model must visually parse the map (lines, stations, interchanges) and perform graph-like path reasoning, then emit a strictly-formatted JSON object.

---

## Approach

| Component | Choice |
|---|---|
| Base model | `Qwen/Qwen2-VL-7B-Instruct` |
| Fine-tuning method | LoRA (PEFT) — parameter-efficient |
| Trainer | `SFTTrainer` from TRL |
| LoRA config | `r=16`, `alpha=32`, dropout `0.05`, targets `q_proj, k_proj, v_proj, o_proj` |
| Epochs | 3 |
| Batch size | 2 per device × 4 grad-accum (effective 8) |
| Learning rate | `2e-4` |
| Precision / placement | `torch_dtype="auto"`, `device_map="auto"` |

The data is reformatted into chat-style multimodal messages (`image` + `text` → `assistant` JSON) before supervised fine-tuning.

### Inference & output validation

`solution.py` runs the fine-tuned adapter over the test set and enforces strict output rules:
- Extracts the first valid JSON object from the model's text output.
- Validates that the route **starts at `source`** and **ends at `target`**, and that `transfer_count >= 0`.
- Falls back to a safe default (`-1`, `"Forest,Delta"`) when the model fails to produce a valid prediction.
- Writes predictions to `submission.csv` (commas inside the route field are auto-quoted to satisfy CSV formatting).

---

## Dataset

Synthetic MRT route-tracing dataset (~622 MB of PNG images). **Images are not committed** to this repo (see `.gitignore`); only the label/metadata CSVs are included.

| Split | Rows | Images |
|---|---|---|
| Train | 1,000 | 1,000 |
| Test | 3,500 | 3,500 |

**`train.csv`** columns:
`id, image_path, task_aux, transfer_count, difficulty, station_count, line_count, interchange_count`

**`test.csv`** columns:
`id, image_path, task_aux`

`task_aux` encodes the query, e.g. `source=Forest;target=Delta`. Each map varies in `difficulty` (`easy`/`hard`), `station_count` (10–57+), `line_count`, and `interchange_count`.

To run training/inference, place the images under:
```
dataset/public/train_images/
dataset/public/test_images/
```

---

## Repo structure

```
finetune-multimodal/
├── solution.py            # Full pipeline: data formatting, LoRA SFT training, inference, submission
├── main.py                # Entry-point stub
├── pyproject.toml         # Project metadata (Python >= 3.12)
├── .python-version        # 3.12
├── sample_submission.csv  # Expected submission format
└── dataset/public/
    ├── train.csv          # Train labels/metadata (images gitignored)
    └── test.csv           # Test queries (images gitignored)
```

---

## Setup & usage

```bash
# Dependencies (install into your environment)
pip install torch transformers peft trl datasets pillow qwen-vl-utils accelerate

# Place dataset images under dataset/public/{train_images,test_images}/

# Run the full train + inference pipeline (GPU recommended)
python solution.py
```

Outputs:
- LoRA adapter → `working/qwen-mrt-lora-final/`
- Predictions → `submission.csv`

---

## Known limitation

The CSV labels contain `transfer_count` but **not** the ground-truth station-by-station route. During training, the assistant target uses a placeholder route (`source,target`), so SFT primarily teaches the **output format and transfer-count behavior**, relying on the base model's pretrained visual/reasoning ability for actual path tracing. A stronger version would supervise full routes (e.g. from a graph solver over the map) to directly train path tracing.
