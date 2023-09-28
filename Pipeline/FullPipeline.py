from datetime import datetime
import time, os

import torch
import torch.nn as nn
import torch.optim as optim

from EncDecTransformer.Models import Transformer
from EncDecTransformer.As_BeamTranslator import Translator as As_BeamTranslator
from Pipeline.Trainer import Trainer
from Pipeline.Generator import Generator
from Pipeline.Evaluator import Evaluator
from Modules.LoggingHelp import write_training_logs, save_generated_outputs, save_all_models_logs, save_best_model_logs


class FullPipeline:

  def __init__(self, run_name, logging, save_model):
    # Directories
    self.logging = logging
    self.save_model = save_model
    if self.logging:
      self.creation_time = datetime.now().strftime("%m-%d-%H-%M")
      self.run_name = f"{run_name} - {self.creation_time}"
      self.run_path = f"drive/MyDrive/SynDa_Health/LT3/Runs/{self.run_name}/"
      os.makedirs(self.run_path)


  def train(self, hyperparams, tokenized_dataset):
    # Transformer
    transformer = Transformer(
      keyword_max_length = hyperparams["keyword_max_length"],
      description_max_length = hyperparams["description_max_length"],
      vocab_size = hyperparams["vocab_size"],
      pad_idx = hyperparams["pad_idx"],
      d_model = hyperparams["d_model"],
      d_v = hyperparams["d_v"],
      d_hid = hyperparams["d_hid"],
      n_head = hyperparams["n_head"],
      n_layers = hyperparams["n_layers"],
      dropout = hyperparams["dropout"],
    )

    # Optimizer & Criterion (loss function)
    optimizer = optim.AdamW(transformer.parameters(), lr=hyperparams["learning_rate"], weight_decay=hyperparams["weight_decay"])
    criterion = nn.CrossEntropyLoss(ignore_index=hyperparams["pad_idx"])

    # Training
    print("Model training is starting...")
    start_time = time.time()

    model_checkpoints, training_loss_logs = Trainer(
      batch_size = hyperparams["batch_size"], n_epochs = hyperparams["n_epochs"], checkpoint = hyperparams["batch_per_checkpoint"], pad_idx=hyperparams["pad_idx"],
      vocab_size = hyperparams["vocab_size"], src_tokens=tokenized_dataset["train"]["input_ids"], tgt_tokens=tokenized_dataset["train"]["labels"]
    ).train(model=transformer, criterion=criterion, optimizer=optimizer)
    
    total_training_time = time.time() - start_time
    print("Model training has terminated successfully.\n")

    # Logging
    if self.logging:
      write_training_logs(total_training_time, hyperparams, self.run_path + "1-training_logs.txt")
      print("Training logs saved.\n")
    
    return model_checkpoints, total_training_time, training_loss_logs


  def generate_outputs(self, transformer, translator_params, device, tokenized_dataset, tokenizer):
    # Translator
    translator = As_BeamTranslator(
      transformer=transformer, 
      pad_idx=translator_params["pad_idx"], sos_idx=translator_params["sos_idx"], eos_idx=translator_params["eos_idx"], 
      max_output_length=translator_params["max_seq_length"], 
      beam_size=translator_params["beam_size"], maximal_step_probability_difference=translator_params["maximal_step_probability_difference"],
      nrp_length=translator_params["nrp_length"], alpha=translator_params["alpha"], tree_length_product=translator_params["tree_length_product"]
    ).to(device)

    # Get predictions from model
    predictions = Generator(tokenized_dataset, tokenizer).generate(translator=translator, device=device, nb_output_multiplier=translator_params["nb_output_multiplier"])
    return predictions

  
  def prepare_evaluation_set(self, tokenized_dataset, predictions):
    srcs, preds, refs = [], [], []
    for sample, prediction in zip(tokenized_dataset["validation"], predictions):
      for pred in prediction:
        srcs.append(sample["keywords"])
        preds.append(pred)
        refs.append(sample["descriptions"])
    return srcs, preds, refs
  
  def evaluate(self, sources, predictions, references):
    # Multi-reference automatic evaluation
    return Evaluator().eval(sources, predictions, references)

  
  def select_best_model(self, model_checkpoints, tokenized_dataset, nb_checkpoints, n_epochs, total_time,
                        translator_params, device, tokenizer):
    models = {}
    print("Selecting best model checkpoint...")
    for idx, model in enumerate(model_checkpoints):
      print(f"Evaluating checkpoint {idx+1} out of {nb_checkpoints}...")
      # Calculate training time & epoch number
      training_time, epochs_passed = (total_time / nb_checkpoints) * (idx+1), (n_epochs / nb_checkpoints) * (idx+1)
      # Generate validation outputs
      mult_predictions = self.generate_outputs(model, translator_params, device, tokenized_dataset, tokenizer)
      # Evaluation
      sources, predictions, references = self.prepare_evaluation_set(tokenized_dataset, mult_predictions)
      eval_results = self.evaluate(sources=sources, predictions=predictions, references=references)
      print("Checkpoint evaluation results:", eval_results)
      # Save results for model
      models[idx] = (eval_results, mult_predictions, training_time, epochs_passed)
    
    best_model_idx = max(models, key=lambda model: (0.35*models[model][0]["BLEU"] + 0.2*models[model][0]["ROUGE-1"] + 0.2*models[model][0]["ROUGE-2"] + 100*0.25*models[model][0]["BERTScore"]))
    best_model = model_checkpoints[best_model_idx]
    print("Best model checkpoint found.\n")
    return best_model, models[best_model_idx], models


  def run(self, hyperparams, tokenized_dataset, translator_params, device, tokenizer):
    
    hyperparams["batch_per_checkpoint"] = int((len(tokenized_dataset["train"]) / hyperparams["batch_size"]) / hyperparams["checkpoint_per_epoch"])
    assert hyperparams["batch_per_checkpoint"] >= 1, "checkpoint_per_epoch value is too high. Must be < len(training_data) / batch_size (than one by batch)"
    
    # Training
    model_checkpoints, total_training_time, training_loss_logs = self.train(hyperparams, tokenized_dataset)
    # Select best checkpoint
    trained_transformer, best_model_eval_logs, all_models_eval_logs = self.select_best_model(model_checkpoints, tokenized_dataset, len(model_checkpoints), hyperparams["n_epochs"], total_training_time, translator_params, device, tokenizer)
    # Extract logs
    eval_results, mult_predictions, training_time, epochs_passed = best_model_eval_logs
    print("Multi-ref evaluation results for best model:\n", eval_results, "\n")

    # Logging
    if self.logging:
      # Save best model logs
      nb_params, nb_trainable_params = sum(p.numel() for p in trained_transformer.parameters()), sum(p.numel() for p in trained_transformer.parameters() if p.requires_grad)
      save_best_model_logs(self.run_path + "1-training_logs.txt", translator_params, training_time, epochs_passed, hyperparams["n_epochs"], eval_results, nb_params, nb_trainable_params)
      # Save training loss logs + all evaluation logs
      save_all_models_logs(self.run_path + "3-training_and_eval_logs.txt", hyperparams["n_epochs"], training_loss_logs, all_models_eval_logs, eval_results)
      # Save best model
      if self.save_model:
        torch.save(trained_transformer.state_dict(), self.run_path + "2-trained_model.pt")
      # Save generated outputs
      save_generated_outputs({tokenized_dataset["validation"]["keywords"][i]: mult_predictions[i] for i in range(len(mult_predictions))}, self.run_path + "4-validation_generations.txt")
      print("Model saved.\nLogs saved.\nGenerations saved.\n")

    # Return trained model
    print("Pipeline terminated successfully.\n")
    return trained_transformer
    