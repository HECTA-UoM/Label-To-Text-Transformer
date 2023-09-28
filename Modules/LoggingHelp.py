import pandas as pd
import numpy as np
import time, json


def write_training_logs(training_time, hp, path):

    with open(path, 'w') as f:

        description = f"Model parameters:\n{', '.join([f'{key}={value}' for key, value in hp.items()])}."
        f.write(description)
        f.write("\n\n")

        training_time = time.gmtime(training_time)
        f.write(f"Training time: {time.strftime('%H:%M:%S', training_time)}.\n")
    

def save_generated_outputs(src_to_preds, path):
    with open(path, 'w') as f:
        f.write(json.dumps(src_to_preds, indent=2, separators=(',', ': ')).replace('],', '],\n'))


def save_best_model_logs(path, translator_params, training_time, epochs_passed, total_epochs, scores, nb_params, nb_trainiable_params):
    # Prediction scores
    scores = ', '.join([f'{key}={round(value, 2)}' for key, value in scores.items()])
    training_time = time.strftime('%H:%M:%S', time.gmtime(training_time))

    with open(path, 'a') as f:
        f.write(f"\n\nTranslator parameters:\n{', '.join([f'{key}={value}' for key, value in translator_params.items()])}.\n")
    
        f.write("\n\nBest model (at checkpoint) logs:\n\n")
        f.write(f"Evaluation scores: {scores}.\n")
        f.write(f"Best model training time: {training_time}.\n")
        f.write(f"Number of epochs: {epochs_passed} out of {total_epochs}.\n")
        f.write(f"Number of parameters: {nb_params} (for which {(nb_params/nb_trainiable_params)*100}% are trainable).")

def save_all_models_logs(path, nb_epochs, training_loss_logs, all_models_eval_logs, eval_results):
    with open(path, 'w') as f:
        # Save nb of epochs
        f.write(f"Number of epochs: {nb_epochs}.\n\n")
        # Save training loss logs
        f.write("Training loss logs:\n")
        f.write(f"[{', '.join([str(l) for l in training_loss_logs])}]\n\n")
        # Save all models evaluation logs
        f.write("Evaluation logs:\n")
        ordered_idx = sorted(list(all_models_eval_logs.keys()), reverse=False)
        for idx in ordered_idx:
            eval_res, _, training_time, epoch_passed = all_models_eval_logs[idx]
            training_time = time.strftime('%H:%M:%S', time.gmtime(training_time))
            f.write(f"Results: {eval_res} - (Epoch: {round(epoch_passed, 3)}; Training time: {training_time})\n")
        # Save best evaluation results
        f.write(f"Best evaluation result: {eval_results}\n")
