# LLM Fine-tuning on Medical QnA datasets

This hypothetical scenario involves three training data providers (TDPs), MedQA, ChatDoctor and MedQuad, and a TDC who wishes to fine-tune a base LLM using datasets from these TDPs. The repository contains sample datasets and a model. The model and datasets are for illustrative purposes only.

The end-to-end fine-tuning pipeline consists of the following phases. 

1. Data pre-processing and de-identification
2. Data packaging, encryption and upload
3. Model packaging, encryption and upload 
4. Encryption key import with key release policies
5. Deployment and execution of CCR
6. Model decryption 

## HuggingFace tokens

In ```scenarios/llm-finetune/config/pipeline_config.json```, set your ```HF_READ_TOKEN``` and ```HF_READ_TOKEN``` tokens.

Likewise, set the ```HF_READ_TOKEN``` variable in ```scenarios/llm-finetune/src/load_base_model.py```

## Build container images

Build container images required for this sample as follows. 

```bash
cd scenarios/llm-finetune
./ci/build.sh

```

This script builds the following container images. 

- ```preprocess-medqa, preprocess-chatdoctor, preprocess-medquad```: Containers that pre-process and de-identify datasets. 
- ```ccr-model-save```: Container that saves the model to be trained in Pytorch format (DEVELOPER NOTE: Need to change to ONNX format.)

## Data pre-processing and de-identification

The folders ```scenarios/llm-finetune/data``` contains three sample training datasets. Acting as TDPs for these datasets, run the following scripts to de-identify the datasets. 

```bash
cd scenarios/llm-finetune/deployment/docker
./preprocess.sh
```

This script performs pre-processing and de-identification of these datasets before sharing with the TDC.

## Prepare model for training

Next, acting as a TDC, save a sample model using the following script. 

```bash
./save-model.sh
```

This script will save the model as ```scenarios/llm-finetune/data/modeller/model/model.pth.```

## Deploy locally

Assuming you have cleartext access to all the de-identified datasets, you can train the model as follows. 

```bash
./train.sh
```
The script joins the datasets and trains the model using a pipeline configuration defined in [pipeline_config.json](./config/pipeline_config.json). If all goes well, you should see output similar to the following output, and the trained model will be saved under the folder `/tmp/output`. 

```
docker-train-1  | {"device": "cpu", "saved_model_dir": "/mnt/model/", "model_name": "facebook/opt-350m", "trained_model_output_path": "/mnt/remote/output/model.pth", "input_dataset_path": "/tmp/medqa_chatdoctor_medquad_joined.csv", "HF_READ_TOKEN": "", "HF_WRITE_TOKEN": "", "SAVE_MODEL_REPO_NAME": "Sarang-Galada/medqa-dp-llm-finetune-test_06Feb25", "MAX_TOKENS": 512, "BATCH_SIZE": 4, "NUM_EPOCHS": 1, "Q_PRECISION": "load_in_4bit", "LEARNING_RATE": 5e-5, "EPSILON": 7.5, "DELTA": 1e-5, "MAX_GRAD_NORM": 1.0, "MAX_PHYSICAL_BATCH_SIZE": 4, "LORA_RANK": 8, "LORA_ALPHA": 32, "LORA_TARGET_MODULES": null, "LORA_DROPOUT": 0.05, "LORA_BIAS": null, "LOG_FREQ_STEPS": 50}
docker-train-1  | Epoch [1/5], Loss: 0.0084
docker-train-1  | Epoch [2/5], Loss: 0.4231
docker-train-1  | Epoch [3/5], Loss: 0.0008
docker-train-1  | Epoch [4/5], Loss: 0.0138
docker-train-1  | Epoch [5/5], Loss: 0.0489
```