from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Dict, Any

class DataHandler:
    def __init__(self, dataset_name: str, tokenizer: AutoTokenizer, split: str = 'train'):
        """
        Handles dataset loading and transformation into ChatML format.

        Args:
            dataset_name (str): The name of the dataset to load from Hugging Face Datasets.
            tokenizer (PreTrainedTokenizer): The tokenizer to process text data.
            split (str, optional): The dataset split to load (default is 'train').
        """

        self.dataset_name = dataset_name
        self.tokenizer = tokenizer

        print(f"Loading dataset: {self.dataset_name}")
        self.dataset = load_dataset(self.dataset_name)[split]

    def transform_to_chatml_format(self, example: Dict[str, Any]) -> Dict[str, str]:
        """
        Converts a dataset example into the ChatML format, which includes system, user, and response messages.

        Args:
            example (Dict[str, Any]): A single example from the dataset.

        Returns:
            Dict[str, str]: A dictionary containing the formatted prompt, chosen response, and rejected response.
        """

        if len(example['system']) > 0:
          message = {"role": "system", "content": example['system']}
          system = self.tokenizer.apply_chat_template([message], tokenize=False)
        else:
            system = ""

        # Format the instruction
        message = {"role": "user", "content": example['question']}
        prompt = self.tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

        # Format the chosen answer
        chosen = example['chosen'] + "<|im_end|>\n"

        # Format the rejected answer
        rejected = example['rejected'] + "<|im_end|>\n"

        return {
            "prompt": system + prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
    
    
    def map_dataset(self) -> None:
        """
        Applies the ChatML formatting function to the entire dataset.
        """
        
        print(f"Formating the dataset")
        self.dataset = self.dataset.map(
            self.transform_to_chatml_format,
            remove_columns=self.dataset.column_names
        )


    def sample_data(self, num_samples: int = 10) -> Dataset:
        """
        Returns a shuffled subset of the dataset for quick inspection or testing.

        Args:
            num_samples (int, optional): The number of samples to return (default is 10).

        Returns:
            Dataset: A subset of the dataset with the specified number of samples.
        """

        return self.dataset.shuffle(seed=42).select(range(num_samples))