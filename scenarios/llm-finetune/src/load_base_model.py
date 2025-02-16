from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
import torch
# import os
from pathlib import Path

# Class for handling language model loading and saving
class ModelHandler:
    def __init__(self, model_name, save_dir="saved_models"):
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def load_from_huggingface(self):
        """Load model and tokenizer from HuggingFace"""
        # Load pretrained model configurations and tokenizer
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)


        ### TO DO
        # Create a quantization configuration for 4-bit precision
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        
        # Load pretrained model based on configurations. Force CPU device
        device = torch.device('cpu')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            config=self.config,
            quantization_config=quantization_config,
            device_map={"": device}
        )
        
        # Prepare model for low-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Ensure padding token is set as EOS
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        return self.model, self.tokenizer
    
    def save_locally(self, format="pytorch"):
        """Save model and tokenizer locally in specified format"""
        if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
            raise ValueError("Model and tokenizer must be loaded first")
            
        # Create model-specific directory
        model_save_dir = self.save_dir / self.model_name.replace('/', '_')
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "pytorch":
            # Save in PyTorch format
            torch.save(self.model.state_dict(), 
                      model_save_dir / "model.pth")
            self.tokenizer.save_pretrained(model_save_dir)
            self.config.save_pretrained(model_save_dir)
            
        # elif format.lower() == "onnx":
        #     # Save in ONNX format
        #     dummy_input = torch.zeros(1, 512, dtype=torch.long)
        #     torch.onnx.export(self.model, 
        #                     dummy_input,
        #                     model_save_dir / "model.onnx",
        #                     export_params=True,
        #                     opset_version=11)
        #     self.tokenizer.save_pretrained(model_save_dir)
        #     self.config.save_pretrained(model_save_dir)
            
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        print(f"Model and tokenizer saved in {format} format at {model_save_dir}")
        return str(model_save_dir)
    
    def load_locally(self, local_dir=None, format="pytorch"):
        """Load model and tokenizer from local directory"""
        if local_dir is None:
            local_dir = self.save_dir / self.model_name.replace('/', '_')
        else:
            local_dir = Path(local_dir)
            
        if not local_dir.exists():
            raise ValueError(f"Directory not found: {local_dir}")
            
        # Load configuration and tokenizer
        self.config = AutoConfig.from_pretrained(local_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        
        # Create model instance
        self.model = AutoModelForCausalLM.from_config(self.config)
        
        if format.lower() == "pytorch":
            # Load PyTorch weights
            state_dict = torch.load(local_dir / "model.pth")
            self.model.load_state_dict(state_dict)
            
        # elif format.lower() == "onnx":
        #     # For ONNX, we need to use ONNX Runtime
        #     import onnxruntime as ort
        #     self.model = ort.InferenceSession(str(local_dir / "model.onnx"))
            
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        # Ensure padding token is set as EOS
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        return self.model, self.tokenizer

# Example usage:
MODEL_NAME = "meta-llama/Llama-3.2-1b"
SAVE_DIR = "/mnt/model"
handler = ModelHandler(model_name=MODEL_NAME, save_dir=SAVE_DIR)

# Load from HuggingFace
model, tokenizer = handler.load_from_huggingface()

# Save locally in PyTorch format
save_path = handler.save_locally(format="pytorch")

# # Load from local storage
# model, tokenizer = handler.load_locally(format="pytorch")