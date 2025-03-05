import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig
from trl import DPOTrainer
from typing import Tuple

class RL_DPOTrainer:
    def __init__(self, model_name: str, dataset_name: str, output_dir: str, wandb_token: str):
        """
        Initializes the RL_DPOTrainer class, loads the model and tokenizer, and sets up WandB logging.

        Args:
            model_name (str): The Hugging Face model identifier.
            dataset_name (str): The dataset name used for training.
            output_dir (str): Path to save trained models.
            wandb_token (str): Weights & Biases authentication token.
        """
          
        self.model_name = model_name
        self.output_dir = output_dir
        self.wandb_token = wandb_token

        # Initialize WandB
        print("Logging to WandB")
        wandb.login(key=self.wandb_token)

        # Load model & tokenizer
        print("Loading the models")
        self.model, self.peft_config, self.tokenizer = self.load_model()


    def load_model(self) -> Tuple[AutoModelForCausalLM, LoraConfig, AutoTokenizer]:
        """
        Loads the model, tokenizer, and PEFT LoRA configuration for fine-tuning.

        Returns:
            Tuple[AutoModelForCausalLM, LoraConfig, AutoTokenizer]: The model, LoRA configuration, and tokenizer.
        """

        print(f"Loading models: {self.model_name}")
        # Load the model to fine-tune
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # Load the model to fine-tune
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, load_in_4bit=True)
        model.config.use_cache = False

        # Enable LoRA for efficient fine-tuning
        peft_config = LoraConfig(
            task_type="CAUSAL_LM", 
            r=16, 
            lora_alpha=32, 
            lora_dropout=0.05, 
            bias="none", target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
            )

        return model, peft_config, tokenizer
    

    def train(self, dataset: Dataset) -> None:
        """
        Trains the model using Direct Preference Optimization (DPO).

        Args:
            dataset: The dataset to train on.
        """

        training_args = TrainingArguments(
            output_dir=self.output_dir,  # Path to save trained models
            per_device_train_batch_size=4,  # Number of training examples per device (GPU/TPU)
            gradient_accumulation_steps=4,  # Accumulate gradients over multiple steps to simulate larger batch size
            gradient_checkpointing=True,  # Enable checkpointing to save memory by recomputing activations during backpropagation
            learning_rate=5e-5,  # Initial learning rate for optimizer
            lr_scheduler_type="cosine",  # Learning rate schedule type (cosine decay for smooth reduction)
            max_steps=200,  # Maximum number of training steps (set low for testing purposes)  
            save_strategy="no",  # Disable checkpoint saving (useful for short training runs)
            logging_steps=1,  # Log training progress every step
            optim="paged_adamw_32bit",  # Optimizer type (AdamW with paged memory for efficiency)
            warmup_steps=100,  # Number of warm-up steps before scheduler starts reducing LR
            bf16=True,  # Use BF16 (bfloat16) precision for reduced memory usage  # Enable BF16 for memory efficiency
            report_to="wandb",  # Enable logging to Weights & Biases (WandB) for experiment tracking
        )

        # Initialize the Direct Preference Optimization (DPO) trainer for reinforcement learning
        self.trainer = DPOTrainer(
            self.model,
            args=training_args,  # Training arguments defined in TrainingArguments
            train_dataset=dataset,  # Training dataset containing prompt-response pairs
            tokenizer=self.tokenizer,  # Tokenizer used for processing input text
            peft_config=self.peft_config,  # Configuration for LoRA-based PEFT fine-tuning
            beta=0.1,  # Regularization strength for DPO optimization
            max_prompt_length=1024,  # Maximum allowed length for the input prompt
            max_length=1536,  # Maximum sequence length including prompt and generated output
        )

        self.trainer.train()


    def save_model(self) -> None:
        """
        Saves the trained model and tokenizer to the output directory.
        """

        print(f"Saving the models to {self.output_dir}")
        self.trainer.model.save_pretrained(self.output_dir)
        self.trainer.tokenizer.save_pretrained(self.output_dir)
        print("Models saved")