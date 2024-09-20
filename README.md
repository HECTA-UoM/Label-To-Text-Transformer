# Label-To-Text-Transformer (LT3)
Repository for [Generating Medical Instructions with Conditional Transformer](https://arxiv.org/abs/2310.19727).

Accepted by [NeurIPS 2023 Workshop on Synthetic Data Generation with Generative AI](https://www.syntheticdata4ml.vanderschaar-lab.com), SyntheticData4ML Workshop.

### Please use up-to-date Github repository:
[LT3-with-Sam-Github](https://github.com/SamySam0/LT3)

### Huggingface page:
[LT3-for-inference-deployment](https://huggingface.co/SamySam0/LT3)

## Our Work
### Abstract
Access to real-world medical instructions is essential for medical research and healthcare quality improvement. However, access to real medical instructions is often limited due to the sensitive nature of the information expressed. Additionally, manually labelling these instructions for training and fine-tuning Natural Language Processing (NLP) models can be tedious and expensive. We introduce a novel task-specific model architecture, Label-To-Text-Transformer (LT3), tailored to generate synthetic medical instructions based on provided labels, such as a vocabulary list of medications and their attributes. LT3 is trained on a vast corpus of medical instructions extracted from the MIMIC-III database, allowing the model to produce valuable synthetic medical instructions. We evaluate LT3's performance by contrasting it with a state-of-the-art Pre-trained Language Model (PLM), T5, analysing the quality and diversity of generated texts. We deploy the generated synthetic data to train the SpacyNER model for the Named Entity Recognition (NER) task over the n2c2-2018 dataset. The experiments show that the model trained on synthetic data can achieve a 96-98\% F1 score at Label Recognition on Drug, Frequency, Route, Strength, and Form. 

### If you use this repository, please cite:
```
@InProceedings{LT3,
      title={Generating Medical Instructions with Conditional Transformer}, 
      author={Samuel Belkadi and Nicolo Micheletti and Lifeng Han and Warren Del-Pinto and Goran Nenadic},
      year={2023},
      booktitle={SyntheticData4ML Workshop at NeurIPS 2023}
}
```
