import evaluate
from bert_score import score


class Evaluator:

  def __init__(self):
    self.bleu = evaluate.load("bleu", quiet=True)
    self.rouge = evaluate.load('rouge', quiet=True)

  def untrained_similarity_eval(self, predictions, references): # Multi-reference
    bleu_eval = self.bleu.compute(predictions=predictions, references=references)
    rouge_eval = self.rouge.compute(predictions=predictions, references=references)
    return bleu_eval, rouge_eval

  def trained_similarity_eval(self, predictions, references):
    P_mul, R_mul, F_mul = score(predictions, references, lang="en", rescale_with_baseline=True)
    return float(F_mul.mean())

  def diversity_eval(self, predictions, references):
    return 


  def eval(self, sources, predictions, references):

    bleu, rouge = self.untrained_similarity_eval(predictions, references)
    bertscore = self.trained_similarity_eval(predictions, references)
    # diversity = self.diversity_eval(predictions, references)

    scores = {
      "BLEU": round(bleu['bleu']*100,2),
      "ROUGE-1": round(rouge['rouge1']*100,2),
      "ROUGE-2": round(rouge['rouge2']*100,2),
      "ROUGE-L": round(rouge['rougeL']*100,2),
      "BERTScore": round(bertscore, 2),
    }

    return scores
    
