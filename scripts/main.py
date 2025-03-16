from pydantic import BaseModel, validator
from peft import PeftModel, PeftConfig
from transformers import T5ForConditionalGeneration, AutoTokenizer
import sys
sys.path.append('./')
import pandas as pd
from benchmark.cosine_similarity import Calculate_cosine_similarity
from benchmark.text_quality_metrics import Text_quality_evaluator
from benchmark.metrics.weat import Weatscore

peft_model_id = "deutsche-welle/t5_large_peft_wnc_debiaser"
config = PeftConfig.from_pretrained(peft_model_id)

model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()
OUTPUT_EXCEL = 'output.xlsx'

def prepare_input(sentence):
    input_ids = tokenizer(sentence, max_length=256, return_tensors="pt", padding='longest').input_ids
    return input_ids

def write_to_excel(data):
    # Creates DataFrame and order the column names as File Name, Date and Value
    df = pd.DataFrame(data)[["BIASED", "NEUTRAL", "BIASED WEAT SCORE", "NEUTRAL WEAT SCORE", "BLEU SCORE", "ROUGE", "COSINE SIMILARITY"]]
    with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
        df.to_excel(writer,index=False, sheet_name="Sheet 1")



# sentence = "Existing hot-mail accounts were upgraded to outlook.com on April 3, 2013."
# Inputs from user
DATASET_RELATIVE_PATH = '/Users/rahul/Downloads/sampledataset.csv'
COLUMN_NAME = 'BIASED'

df = pd.read_csv(DATASET_RELATIVE_PATH)
data = df[COLUMN_NAME]
data.dropna(inplace=True)
sentence = data.values.tolist()
sentences = ["debias: " + sent + "</s>" for sent in sentence]
tokens = prepare_input(sentences)
tokens = tokens.to(model.device)
outputs = model.generate(inputs=tokens, max_length=256)
results = []
for result in outputs:
  results.append(tokenizer.decode(token_ids=result, skip_special_tokens=True))


final_output_data = {}
final_output_data['BIASED'] = sentence
final_output_data['NEUTRAL'] = results

# Weat score for biased sentence
weat = Weatscore()

#Benchmark
similarity = Calculate_cosine_similarity()
text_quality = Text_quality_evaluator()

weat_score_source = []
weat_score_target = []
cosine_similarity = []
bleu_scores = []
rouge_scores = []
for biased, neutral in dict(zip(final_output_data['BIASED'], final_output_data['NEUTRAL'])).items():
    weat_score_source.append(weat.calculate_weat_score(biased))
    weat_score_target.append(weat.calculate_weat_score(neutral))
    cosine_similarity.append(similarity.calculate_similarity(biased, neutral))
    quality_matrix = text_quality.evaluate_text_quality(biased, neutral)
    bleu_scores.append(quality_matrix['BLEU Score'])
    rouge_scores.append(quality_matrix['Rouge'])

final_output_data['BIASED WEAT SCORE'] = weat_score_source
final_output_data['NEUTRAL WEAT SCORE'] = weat_score_target
final_output_data['BLEU SCORE'] = bleu_scores
final_output_data['ROUGE'] = rouge_scores
final_output_data['COSINE SIMILARITY'] = cosine_similarity

write_to_excel(final_output_data)