# Surveillance_Bench
<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Pipeline Banner">
</p>

<h1 align="center"> Surveillance GTA Dataset Generator</h1>

<p align="center">
Generate structured GTA-style reasoning datasets from surveillance video frames using LLMs and tool-based reasoning.
</p>

---

## Project Idea

This project converts surveillance video frames into **structured multimodal reasoning datasets**.  
Each video is transformed into one sample containing a question, multi-step tool-based reasoning trajectory, and a final answer generated using an LLM.

---

##  Pipeline Overview

```text id="pipeline1"
Raw Videos
    ↓
extract.py (frame extraction)
    ↓
frames/video_xxx/*.jpg
    ↓
prompt.py (LLM + tool reasoning)
    ↓
GTA-style JSON dataset
    ↓
surveillance_gta_dataset.json
```
## Project Structure
```
Project/
├── videos/    
├── extract.py                  # Extract frames from videos
├── prompt.py                   # Generate GTA dataset using GPT
├── config.py                   # OpenAI API key
├── toolsmeta.json              # Tool definitions
├── GTA_dataset.json            # Sample examples
├── frames/                     # Input video frames
│   ├── video_001/
│   ├── video_002/
│   └── ...
├── surveillance_gta_dataset.json  # Output dataset
```
## How to Run

### 1. Install dependencies
Make sure you have Python installed with required packages:

```bash
pip install openai
```
## 2. Set OpenAi API Key
OPENAI_API_KEY = ""

## 2. Prepare Input Data

```
videos/                      # raw videos (optional)
frames/                      # will be created after extraction
toolsmeta.json              # tool definitions
GTA_dataset.json            # few-shot examples
```
## 4. Run the Pipeline

1. extract.py
2. prompt.py

## 4. Output Format

Each video generates one sample:

```json
{
  "tools": [],
  "files": [],
  "dialogs": [],
  "gt_answer": {
    "whitelist": [["answer"]],
    "blacklist": null
  }
}
