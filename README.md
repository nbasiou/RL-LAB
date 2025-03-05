# RL-DPO
**Align LLM with human preferences using Reinforcement Learning (RL)**

This repository implements the alignment of a fine-tuned LLM using RL Direct Preference Optimization in Google Colab. 
The code supports the following:

- Setup the development environent
- Preparation of the dataset: the dataset [Intel/orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs) is used.
- Alignment of the LLM with TRL-DPOTrainer: The original model [teknium/OpenHermes-2.5-Mistral-7B](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B) is used.
- Testing the aligned LLM

## Dependencies
Install all the necessary package requirements.

````python
pip install -r requirements.txt
````

## Notes
Model training requires an A100 GPU. 


