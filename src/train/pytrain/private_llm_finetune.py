# 2023, The DEPA CCR DP Training Reference Implementation
# authors shyam@ispirt.in, sridhar.avs@ispirt.in
#
# Licensed TBD
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Key references / Attributions: https://depa.world/training/reference-implementation
# Key frameworks used : DEPA CCR,Opacus, PyTorch,ONNX, onnx2pytorch


### NEW IMPORTS ###

# For parsing configs and args
import argparse
import json
import os
# For standard ML procedures
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
# For loading readymade models and datasets
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, get_scheduler
# For parameter-efficient training
import peft
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
# For differential privacy
import opacus
from opacus import PrivacyEngine  # For differential privacy
from opacus.utils.batch_memory_manager import BatchMemoryManager  # For large batch sizes
# For monitoring training progress
from tqdm import tqdm
# For accessing HuggingFace Hub
from huggingface_hub import login, HfApi, HfFolder, Repository

###################


### OLD IMPORTS ###

# # torch related imports
# from typing import Optional
# import torch
# from torchvision import datasets, transforms

# from tqdm import tqdm
# import torch.utils.data as data
# from torch.utils.data import DataLoader
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

# # sklearn,pandas,numpy related imports
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score
# import numpy as np
# import pandas as pd

# # opacus related imports
# from opacus.accountants import create_accountant
# from opacus import PrivacyEngine

# # onnx related imports
# import onnx
# from onnx2pytorch import ConvertModel

# other imports
# import os
# import json
# import argparse
# from pathlib import Path

###################

from .task_base import TaskBase

logger = {
    "epochs_per_report": 1,
    "metrics": [
        "tdp_config",
        "tdc_config",
        "model_architecture",
        "model_hyperparameters",
        "model_config",
        "accuracy",
        "precision",
        "recall",
    ],
    "ccr_pbt_logger_file": "/mnt/remote/output/ccr_depa_trg_model_logger.json",
}

# def compute_delta(ccr_context):
#     return 1 / ccr_context["sample_size"]


# class CustomDataset(Dataset):
#     """
#     Class to convert dataset columns to tensors
#     """

#     def __init__(self, features, target):
#         self.features = torch.tensor(features, dtype=torch.float32)
#         self.target = torch.tensor(target.values, dtype=torch.float32)

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         return self.features[idx], self.target[idx]


class PrivateLLMFineTune(TaskBase):
    """
    Args:
    config:training configuration 

    Methods:
    load_data:loads data from HuggingFace repo, tokenizes and prepares dataloaders for training
    load_model:loads model object from model config
    load_optimizer:loads model optimizer from model config
    apply_lora:applies lora to the model using peft
    make_dprivate:make model,dataloader and optimizer private
    train:differentially private llm finetuning
    save_model_ft:saves and pushes model to HuggingFace repo
    execute:mega function which includes all the above functions

    """

    def init(self, config):
        # self.DEVICE = torch.device(config["DEVICE"] if torch.cuda.is_available() else "cpu")
        self.device = torch.device(config["device"])

        self.config = config
        self.model = None
        self.model_config = None
        self.tokenizer = None
        self.dataset = None
        self.train_loader = None
        # self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.privacy_engine = None
        self.loss_fn = CrossEntropyLoss()
        self.model_ft = None

        login(config["HF_READ_TOKEN"])

        print("*** STARTING FINE-TUNING ***")
        print(f"Fine-tuning {config["MODEL_NAME"]} on {config["DATASET_NAME"]}...")

    # def ccr_logger_function(ccr_tracking_object, ccr_model):
    #     """
    #     Function to implement logging for audit/model cert
    #     """
    #     file_path = ccr_tracking_object["ccr_pbt_logger_file"]
    #     with open(file_path, "w") as file:
    #         file.write("Model Architecture\n")
    #         string = str(ccr_model.model)
    #         file.write(string)
    #         for c in ccr_model.logger_list:
    #             file.write(c)


    def load_model(self):
        # Load pretrained model configurations and tokenizer
        local_dir = self.config["saved_model_dir"] + self.config["model_name"].replace("/", "_")
        
        # Load configuration and tokenizer
        if not local_dir.exists():
            raise ValueError(f"Directory not found: {local_dir}")

        # Load configuration and tokenizer
        self.model_config = AutoConfig.from_pretrained(local_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        
        # Create model instance
        self.model = AutoModelForCausalLM.from_config(self.model_config)
        
        if format.lower() == "pytorch":
            # Load PyTorch weights
            state_dict = torch.load(local_dir / "model.pth")
            self.model.load_state_dict(state_dict)

        # Ensure padding token is set as EOS
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def load_data(self):

        if format == "csv":
            # Load from CSV
            df = pd.read_csv(self.config["input_dataset_path"])
            self.dataset = Dataset.from_pandas(df)

        def tokenize_function(examples):
            return self.tokenizer(examples, padding="max_length", truncation=True, max_length=self.config["MAX_TOKENS"])

        tokenized_inputs = [tokenize_function(input_text) for input_text in self.dataset["input"]]
        tokenized_outputs = [tokenize_function(output_text) for output_text in self.dataset["output"]]

        # Create the dataset with tokenized inputs and corresponding outputs
        train_dataset = [(torch.tensor(t["input_ids"]), torch.tensor(t["attention_mask"]), torch.tensor(output.input_ids)) 
                        for t, output in zip(tokenized_inputs, tokenized_outputs) if (len(t["input_ids"])>0 and len(output)>0)]

        # Create the DataLoader
        self.train_loader = DataLoader(train_dataset, batch_size=self.config["BATCH_SIZE"], shuffle=True)


    def load_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["LEARNING_RATE"])
        self.scheduler = get_scheduler(
            name="linear",  # You can also use "cosine" or other schedules
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.config["NUM_EPOCHS"] * len(self.train_loader)  # Total number of training steps,
        )


    def apply_lora(self):
        lora_config = LoraConfig(
            r=self.config["LORA_RANK"], # 8, # Rank of the low-rank matrices
            lora_alpha=self.config["LORA_ALPHA"], # 32, # Scaling factor for the LoRA updates
            target_modules=self.config["LORA_TARGET_MODULES"], # ["q_proj", "v_proj"], # Modules to apply LoRA to  ### Modify as per model architecture
            lora_dropout=self.config["LORA_DROPOUT"], # 0.05, # Dropout probability applied to the LoRA updates for regularization
            bias=self.config["LORA_BIAS"], # "none", # Whether to include bias parameters in the LoRA layers
            task_type="CAUSAL_LM" # Type of task - eg. causal modelling, seq2seq
        )
    
        # Obtain the parameter-efficient LoRA model
        self.model = get_peft_model(self.model, lora_config)


    def make_dprivate(self):
        self.privacy_engine = PrivacyEngine() # secure_mode=True requires torchcsprng to be installed

        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            epochs=self.config["NUM_EPOCHS"],
            target_delta=self.config["DELTA"],  # Privacy budget
            target_epsilon=self.config["EPSILON"],  # Probability of privacy breach
            max_grad_norm=self.config["MAX_GRAD_NORM"], # threshold for clipping the norm of per-sample gradients
        )

    def train(self):
        # 8. Training loop with BatchMemoryManager
        self.model.train()
        for epoch in range(1, self.config["NUM_EPOCHS"] + 1):
            losses = []

            # Use BatchMemoryManager for managing memory
            with BatchMemoryManager(
                data_loader=self.train_loader,
                max_physical_batch_size=self.config["MAX_PHYSICAL_BATCH_SIZE"],
                optimizer=self.optimizer
            ) as memory_safe_loader:

                # Training step
                for step, batch in enumerate(tqdm(memory_safe_loader, desc=f"Epoch {epoch}/{self.config["NUM_EPOCHS"]}")):
                    # Move batch to DEVICE
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)

                    # Skip empty batches
                    if input_ids.size(0) == 0:
                        continue

                    # Forward pass
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits  # Model predictions
            
                    # Compute loss
                    # Shift logits and labels for causal language modeling
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
                    # Backward pass and optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            
                    # Record loss
                    losses.append(loss.item())

                    # Log progress every 50 steps
                    if step > 0 and step % 50 == 0:
                        train_loss = np.mean(losses)
                        epsilon = self.privacy_engine.get_epsilon(self.config["DELTA"])

                        print(
                            f"Epoch: {epoch} | Step: {step} | "
                            f"Train loss: {train_loss:.3f} | "
                            f"ɛ: {epsilon:.2f}"
                        )

            # Epoch summary
            train_loss = np.mean(losses)
            epsilon = self.privacy_engine.get_epsilon(self.config["DELTA"])
            print(f"Epoch {epoch} completed. Average loss: {train_loss:.4f}, ɛ: {epsilon:.2f}")


        # 9. Unwrap the DP fine-tuned model - our model is wrapped by a PEFT wrapper and Opacus wrapper

        ## Step 1: Check if the model is wrapped in GradSampleModule (Opacus wrapper)
        if isinstance(self.model, opacus.grad_sample.GradSampleModule):
            unwrapped_model = self.model._module  # Access the underlying model from GradSampleModule
        else:
            unwrapped_model = self.model  # If not wrapped, use the model as-is
        ## Step 2: For LoRA/PEFT models, unwrap further
        if isinstance(unwrapped_model, peft.PeftModelForCausalLM):
            self.model_ft = unwrapped_model.base_model  # Extract the base model under the PEFT wrapper
        else:
            self.model_ft = unwrapped_model  # If not a PEFT model, use as-is
        
        # Set model for inference by freezing parameters
        self.model_ft.eval()
        print("model unwrapped!")


    def save_ft_model(self):
        # 10. Push to HuggingFace Hub
        self.model_ft.push_to_hub(self.config["SAVE_MODEL_REPO_NAME"], token=self.config["HF_WRITE_TOKEN"])
        self.tokenizer.push_to_hub(self.config["SAVE_MODEL_REPO_NAME"], token=self.config["HF_WRITE_TOKEN"])
        print(f"Model and tokenizer pushed to Hugging Face Hub: https://huggingface.co/{self.config["SAVE_MODEL_REPO_NAME"]}")


    def execute(self, config):
        try:
            # --- START OF FINE-TUNING CODE ---
            self.init(config)
            self.load_model()
            self.load_data()
            self.load_optimizer_and_scheduler()
            self.apply_lora()
            self.make_dprivate()
            self.train()
            self.save_ft_model()
            print("Fine-tuning complete!")
            # --- END OF FINE-TUNING CODE ---

        except Exception as e:
            print(f"An error occurred during fine-tuning: {e}")
            exit(1)
        
