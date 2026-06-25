# Surveillance GTA Dataset Generator

Generate GTA-style JSON benchmark samples from surveillance video frames.

This repo now follows the GTA JSON structure exactly at the sample level:

```json
{
  "0": {
    "tools": [],
    "files": [],
    "dialogs": [],
    "gt_answer": {
      "whitelist": [["answer"]],
      "blacklist": null
    }
  }
}
```

Each video becomes one dataset sample. The sample uses chronological keyframes from that video instead of the static image inputs used by the original GTA dataset.

## What this version does

- Extracts representative keyframes from raw surveillance videos into `frames/<video_name>/`.
- Loads the reference GTA JSON file (`GTA dataset.json`) as style examples.
- Loads tool metadata (`toolmeta.JSON`) and passes it into every generation request.
- Passes the selected video frames to the vision model in chronological order.
- Keeps prompts in separate editable text files under `prompts/`.
- Restricts generation to surveillance-observation questions only.
- Uses the OpenAI Batch API for full runs to reduce generation cost.
- Post-processes and validates the model response so the final output keeps the exact GTA top-level structure.

## Project structure

```text
Surveillance_Bench-main/
├── GTA dataset.json                         # GTA reference examples
├── README.md
├── extract.py                               # Raw video -> keyframes
├── prompt.py                                # Batch-first GTA JSON generation
├── requirements.txt
├── toolmeta.JSON                            # Tool metadata passed to the model
├── prompts/
│   ├── surveillance_system_prompt.txt       # Global rules and safety constraints
│   └── surveillance_user_prompt.txt         # Template with GTA/tool/frame placeholders
├── videos/                                  # Put raw .mp4/.avi/.mov/.mkv videos here
├── frames/                                  # Created by extract.py
│   └── <video_name>/
│       ├── keyframe_001_t_00000.000s.jpg
│       ├── keyframe_002_t_00001.000s.jpg
│       └── keyframes.json
├── runs/
│   └── surveillance_batch/
│       ├── requests.jsonl                   # Batch input file
│       ├── manifest.json                    # Maps Batch custom_id -> video/frame paths
│       ├── batch_id.txt
│       ├── batch_output.jsonl
│       └── report.json
└── surveillance_gta_dataset.json            # Final output
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your API key with an environment variable:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Alternatively, create a local file named `openai_key.txt` containing only the API key. Do not commit that file.

You can also choose a model:

```bash
export OPENAI_MODEL="gpt-4o"
```

## Step 1: Extract frames from videos

Put raw videos in `videos/`, then run:

```bash
python extract.py --videos-dir videos --output-root frames --max-frames 80 --sample-fps 2
```

This creates one frame folder per video:

```text
frames/video_001/*.jpg
frames/video_002/*.jpg
```

To extract one video only:

```bash
python extract.py videos/video_001.mp4 -o frames/video_001 --max-frames 80 --sample-fps 2
```

Useful extraction options:

```bash
python extract.py --help
```

Important options:

- `--max-frames`: hard cap on saved frames per video. Default is `80`.
- `--sample-fps`: low-rate FPS used while scanning the video.
- `--contact-sheet`: also creates `contact_sheet.jpg` for quick visual inspection.

## Step 2: Edit prompts if needed

The prompts are separate text files:

```text
prompts/surveillance_system_prompt.txt
prompts/surveillance_user_prompt.txt
```

Modify these files to change question style or surveillance constraints. The code automatically injects:

- `$GTA_EXAMPLES_JSON`
- `$TOOL_METADATA_JSON`
- `$FRAME_MANIFEST_JSON`

Do not remove those placeholders from `surveillance_user_prompt.txt` unless you also update `prompt.py`.

## Step 3: Full run with Batch API

Use Batch API for full dataset generation.

```bash
python prompt.py run-batch \
  --frames-dir frames \
  --gta-file "GTA dataset.json" \
  --tools-file toolmeta.JSON \
  --output-dir runs/surveillance_batch \
  --output-file surveillance_gta_dataset.json \
  --max-frames-per-video 24 \
  --num-gta-examples 5 \
  --batch-image-max-side 1024 \
  --image-detail low
```

What happens:

1. `prompt.py` builds `runs/surveillance_batch/requests.jsonl`.
2. It uploads the JSONL file for Batch processing.
3. It creates a Batch job and saves the Batch id to `runs/surveillance_batch/batch_id.txt`.
4. It polls the Batch job until it reaches a terminal status.
5. It downloads Batch output and writes `surveillance_gta_dataset.json`.
6. It writes a validation report to `runs/surveillance_batch/report.json`.

## Alternative: Prepare, submit, check, collect manually

This is useful when you do not want to keep one terminal process open.

Prepare the Batch JSONL file:

```bash
python prompt.py prepare-batch \
  --frames-dir frames \
  --gta-file "GTA dataset.json" \
  --tools-file toolmeta.JSON \
  --output-dir runs/surveillance_batch \
  --max-frames-per-video 24
```

Submit it:

```bash
python prompt.py submit-batch \
  --input-jsonl runs/surveillance_batch/requests.jsonl \
  --batch-id-file runs/surveillance_batch/batch_id.txt
```

Check status:

```bash
python prompt.py status --batch-id "batch_id_here"
```

Collect final results after the Batch job completes:

```bash
python prompt.py collect \
  --batch-id "batch_id_here" \
  --manifest runs/surveillance_batch/manifest.json \
  --tools-file toolmeta.JSON \
  --output-file surveillance_gta_dataset.json
```

## Small debug run without Batch

Use synchronous mode only for quick testing on one or two videos:

```bash
python prompt.py run-sync \
  --frames-dir frames \
  --gta-file "GTA dataset.json" \
  --tools-file toolmeta.JSON \
  --output-file surveillance_gta_dataset_debug.json \
  --limit 1
```

For full dataset generation, use `run-batch`.

## Validate the final JSON

```bash
python prompt.py validate \
  --dataset surveillance_gta_dataset.json \
  --tools-file toolmeta.JSON \
  --manifest runs/surveillance_batch/manifest.json
```

Validation checks:

- Dataset is a JSON object keyed by sample id.
- Each sample has exactly the GTA top-level keys: `tools`, `files`, `dialogs`, `gt_answer`.
- `files` match the frame manifest.
- The first dialog turn is user.
- The final dialog turn is assistant content.
- Tool names used in dialogs exist in `toolmeta.JSON`.
- `gt_answer.blacklist` is `null` and `whitelist` is nested like GTA.

## Output details

The final `surveillance_gta_dataset.json` is keyed by video index:

```json
{
  "0": {
    "tools": [
      {
        "name": "SceneDescriber",
        "description": "A useful tool that returns a brief description of the input image.",
        "inputs": [],
        "outputs": []
      }
    ],
    "files": [
      {
        "type": "image",
        "path": "frames/video_001/keyframe_001_t_00000.000s.jpg",
        "url": ""
      }
    ],
    "dialogs": [
      {
        "role": "user",
        "content": "Which direction does the person in the red jacket move between the first and last visible frames?"
      },
      {
        "role": "assistant",
        "tool_calls": [
          {
            "type": "function",
            "function": {
              "name": "SceneDescriber",
              "arguments": {
                "image": "frames/video_001/keyframe_001_t_00000.000s.jpg"
              }
            }
          }
        ],
        "thought": "I need to describe the initial frame before comparing it with later frames."
      },
      {
        "role": "tool",
        "name": "SceneDescriber",
        "content": {
          "type": "text",
          "content": "The person in the red jacket is near the left side of the entrance."
        }
      },
      {
        "role": "assistant",
        "content": "The person in the red jacket moves from left to right."
      }
    ],
    "gt_answer": {
      "whitelist": [["The person in the red jacket moves from left to right."]],
      "blacklist": null
    }
  }
}
```

`prompt.py` overrides the model's `files` field with the real frame paths from the manifest, then normalizes the sample so the top-level keys match GTA exactly.

## Cost and size controls

Batch request files include image payloads, so keep them small enough to upload.

Recommended defaults:

```bash
--max-frames-per-video 24
--batch-image-max-side 1024
--batch-jpeg-quality 85
--image-detail low
```

If `requests.jsonl` is close to 200 MB, reduce one or more of:

```bash
--max-frames-per-video 12
--batch-image-max-side 768
--num-gta-examples 3
```

`--num-gta-examples all` is supported, but it repeats the entire GTA JSON context in every request and can be expensive. The default style-reference setting is `5` examples.

## Notes on surveillance prompts

The default prompts intentionally avoid unsafe surveillance tasks. They ask for benign observations such as counts, motion, object placement, entry/exit, visible text, queue state, and scene changes. They also instruct the model not to identify real people, match faces, infer protected attributes, read private personal data, or accuse someone of a crime.
