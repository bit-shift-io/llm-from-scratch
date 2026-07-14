# LLM FROM SCRATCH

Code in this repo is a result of following the tutorial here:
`https://github.com/angelos-p/llm-from-scratch/tree/main`

## Requirements

- **Apple Silicon / Linux / Windows:** Python 3.9–3.13 (PyTorch has no 3.14 wheels yet)
- **Intel Macs (x86_64):** Python **3.12**. PyTorch stopped shipping Intel-macOS
  builds after 2.2.2, which requires Python ≤ 3.12 and NumPy 1.x. `requirements.txt`
  selects the right torch/numpy versions automatically via environment markers.

## Installation

Clone the repo and set up a virtual environment:

```bash
git clone <repo-url>
cd llm-from-scratch

# Create and activate a virtual environment.
# On an Intel Mac use python3.12; elsewhere python3.13 (or lower) is fine:
python3.12 -m venv .venv
source .venv/bin/activate       # on Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Running

```bash
python model.py
```
