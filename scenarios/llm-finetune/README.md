# LLM Fine-Tuning Scenario — Financial Sentiment & Advisory

This scenario demonstrates multi-party LLM fine-tuning inside a DEPA Confidential Clean Room.  Three financial data providers contribute de-identified datasets to fine-tune a language model for financial question-answering, without any party seeing the others' raw data.

## Scenario Overview

| Property | Value |
|----------|-------|
| **Task** | Instruction-tuned financial Q&A |
| **Base Model** | microsoft/phi-2 (2.7B params) |
| **Method** | LoRA (rank 16) with optional DP-LoRA |
| **Framework** | HuggingFace (SFTTrainer + PEFT) |
| **TDPs** | 3 — market data provider, fintech provider, bank |
| **Privacy** | Differential Privacy via Opacus (epsilon=1.0) |
| **Data Format** | JSONL (Alpaca instruction format) |
| **Output Format** | Safetensors (adapter or merged) |

## Important: No Network Access Inside the CCR

The Confidential Clean Room has **no internet access** at runtime.  All models and datasets must be pre-downloaded by their respective owners before being encrypted and uploaded.

```
 TDP 1 (Market Data)     TDP 2 (FinTech)      TDP 3 (Bank)         Modeller
 ┌─────────────────┐     ┌────────────────┐    ┌──────────────┐    ┌─────────────────┐
 │ prepare_market   │     │ prepare_fintech│    │ prepare_bank │    │ download_model   │
 │ _data.py         │     │ _data.py       │    │ _data.py     │    │ .py              │
 │       ↓          │     │       ↓        │    │       ↓      │    │       ↓          │
 │ encrypt + upload │     │ encrypt + upload│    │ encrypt +   │    │ encrypt + upload │
 └────────┬─────────┘     └───────┬────────┘    │ upload      │    └────────┬─────────┘
          │                       │              └──────┬──────┘             │
          └───────────────────────┼─────────────────────┼───────────────────┘
                                  ↓                     ↓
                     ┌─────────────────────────────────────────────┐
                     │   Confidential Clean Room (NO internet)     │
                     │   encfs decrypts → pytrain LLM_Finetune    │
                     │   /mnt/remote/<tdp>/   /mnt/remote/model/  │
                     └─────────────────────────────────────────────┘
```

## Parties and Trust Boundaries

| Party | Role | Script | What they run |
|-------|------|--------|---------------|
| **Market Data Corp** (TDP 1) | Data provider | `preprocess.sh` | `prepare_market_data.py` — downloads `financial_phrasebank`, converts to JSONL |
| **FinTech Solutions** (TDP 2) | Data provider | `preprocess.sh` | `prepare_fintech_data.py` — downloads financial instruction data, converts to JSONL |
| **National Bank** (TDP 3) | Data provider | `preprocess.sh` | `prepare_bank_data.py` — generates compliance Q&A JSONL |
| **Modeller** | Model provider | `save-model.sh` | `download_model.py` — downloads HF model to local safetensors |

Each party runs their step independently on their own machine.  They do not share raw data with each other.

## Quick Start (Local — CPU Sanity Test)

For machines without a GPU.  Validates the entire pipeline end-to-end.

### Step 1: Build scenario containers

```bash
cd scenarios/llm-finetune
bash ci/build.sh
```

### Step 2: Modeller downloads the base model

```bash
cd deployment/local
HF_MODEL_ID=HuggingFaceTB/SmolLM2-135M MODEL_LOCAL_NAME=smollm2-135m bash save-model.sh
```

### Step 3: Data providers prepare their datasets

```bash
MAX_SAMPLES=50 bash preprocess.sh
```

### Step 4: Build the LLM training container

```bash
cd ../../../../                    # repo root
docker build -f ci/Dockerfile.llm-train-cpu src -t depa-training-llm-cpu:latest
```

### Step 5: Run the CPU training

```bash
cd scenarios/llm-finetune/deployment/local
bash train-cpu.sh
```

### Step 6: Check outputs

Outputs in `modeller/output/`:
- `adapter_model.safetensors` — LoRA adapter weights
- `adapter_config.json` — PEFT config
- `audit.jsonl` — HMAC-signed audit log
- `metrics_export.json` — Training metrics
- `mlruns/` — MLflow experiment data

## Quick Start (Local — GPU)

### Step 1: Build scenario containers

```bash
cd scenarios/llm-finetune
bash ci/build.sh
```

### Step 2: Modeller downloads the base model

```bash
cd deployment/local
bash save-model.sh                # default: microsoft/phi-2
```

### Step 3: Data providers prepare their datasets

```bash
bash preprocess.sh
```

### Step 4: Build the training container (from repo root)

```bash
cd ../../../../ && ./ci/build.sh
```

### Step 5: Run training

```bash
cd scenarios/llm-finetune/deployment/local
bash train.sh
```

## Configuration

Two configs are provided:

| Config | Use case | Model | Steps | DP |
|--------|----------|-------|-------|----|
| `llm_finetune_config.yaml` | GPU production | phi-2 (2.7B) | full 3 epochs | Yes (epsilon=1.0) |
| `llm_finetune_config_cpu.yaml` | CPU sanity test | SmolLM2-135M | 20 steps | No |

All configuration is schema-validated. No arbitrary code execution is permitted inside the enclave.

## Supported Frameworks

| Framework | Status | Notes |
|-----------|--------|-------|
| HuggingFace (SFTTrainer) | Primary | Uses `trl` + `peft` |
| Axolotl | Supported | Wraps axolotl Python API |
| Torchtune | Supported | Invokes torchtune recipes |
| Raw PyTorch | Supported | Hand-written training loop with PEFT |
