import os
import json
import base64
import random
from pathlib import Path
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-5"

FRAMES_DIR = "frames"
TOOLS_FILE = "toolsmeta.json"
GTA_FILE = "GTA_dataset.json"

OUTPUT_FILE = "surveillance_gta_dataset.json"


NUM_EXAMPLES = 5 # Number of random GTA dataset examples included in the prompt as references


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_tools():
    with open(TOOLS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_gta_examples():
    with open(GTA_FILE, "r", encoding="utf-8") as f:
        gta = json.load(f)

    keys = list(gta.keys())

    if len(keys) <= NUM_EXAMPLES:
        selected = [gta[k] for k in keys]
    else:
        selected = [gta[k] for k in random.sample(keys, NUM_EXAMPLES)]

    return selected


def get_video_folders():
    folders = []

    for item in os.listdir(FRAMES_DIR):
        full = os.path.join(FRAMES_DIR, item)

        if os.path.isdir(full):
            folders.append(full)

    folders.sort()
    return folders


def get_keyframes(folder):
    frames = list(Path(folder).glob("*.jpg"))

    frames = sorted(frames)

    return [str(x) for x in frames]


# Prompt
def build_prompt(tools_meta, gta_examples):

    return f"""
You are generating a GTA-style benchmark sample.

You will receive:
1. Surveillance video keyframes in chronological order.
2. Available tools.
3. GTA benchmark examples.


TOOLS:
{json.dumps(tools_meta, indent=2)}

GTA EXAMPLES:
{json.dumps(gta_examples, indent=2)}


TASK:
Generate ONE new GTA benchmark sample.

Requirements:

1. Create a realistic surveillance question.

2. The question must require reasoning across multiple keyframes.

3. Use only tools from the provided tool list.

4. Generate a complete GTA trajectory:
   - user
   - assistant
   - tool
   - assistant
   - tool
   ...
   - assistant final answer

5. Assistant messages should contain:
   - thought
   - tool_calls

6. Tool outputs should be realistic.

7. The final answer must be consistent with the trajectory.

8. Include:
   - tools
   - files
   - dialogs
   - gt_answer

9. Follow GTA style exactly.

10. Output ONLY valid JSON.

Output format:

{{
    "tools": [...],
    "files": [...],
    "dialogs": [...],
    "gt_answer": {{
        "whitelist": [[ "...answer..." ]],
        "blacklist": null
    }}
}}
"""

# Generate Sample
def generate_sample(video_folder, tools_meta, gta_examples):

    frame_paths = get_keyframes(video_folder)

    prompt = build_prompt(
        tools_meta,
        gta_examples
    )

    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]

    files_section = []

    for frame in frame_paths:

        files_section.append(
            {
                "type": "image",
                "path": frame,
                "url": ""
            }
        )

        img_b64 = encode_image(frame)

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }
            }
        )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        response_format={
            "type": "json_object"
        }
    )

    sample = json.loads(
        response.choices[0].message.content
    )

    sample["files"] = files_section

    return sample


# Main
def main():

    tools_meta = load_tools()

    gta_examples = load_gta_examples()

    folders = get_video_folders()

    dataset = {}

    for idx, folder in enumerate(folders):

        print(f"Processing {folder}")

        try:

            sample = generate_sample(
                folder,
                tools_meta,
                gta_examples
            )

            dataset[str(idx)] = sample

            with open(
                OUTPUT_FILE,
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(
                    dataset,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

        except Exception as e:

            print(
                f"Failed {folder}: {e}"
            )

    print(
        f"Saved to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()