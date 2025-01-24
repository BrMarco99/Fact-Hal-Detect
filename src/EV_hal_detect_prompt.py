import os
import pandas as pd
from tqdm import tqdm
import argparse
import jsonlines

# Argument parsing
parser = argparse.ArgumentParser(description="Relevance checking script for QA.")
parser.add_argument("--model", type=str, default='llama3', help="Model to be used for LLMChat")
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--prompt", type=str)
parser.add_argument("--save_freq", type=int, default=500, help="Frequency of saving checkpoints")
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
args = parser.parse_args()

if not os.path.exists("output"):
    os.makedirs("output")

output_path = "output"
source_path = "../"
fileInputName = os.path.join(source_path, "files/TruthfulQA", args.input)
filePromptName = os.path.join(source_path, "prompts", args.prompt)
fileOutputName = os.path.join(output_path, args.output)
df = pd.read_json(fileInputName, lines=True)


print("\n\n\n")
print("USING MODEL: ", args.model)
print ("OPENING: ", fileInputName)
print ("PROMPT:", filePromptName)
print("\n\n\n")


IDs = df['ID'].tolist()
types = df['Type'].tolist()
categories = df['Category'].tolist()
questions = df['Question'].tolist()
answers = df['Answer'].tolist()
sources = df['Source'].tolist()
evidences = df['Evidences'].tolist()
real_fact_labels = df['Factuality_ground_label'].tolist()

# Read main instruction
with open(filePromptName, 'r', encoding="utf-8") as f:
    main_instruction = f.read()



print ("Writing in:", fileOutputName)    
directory = os.path.dirname(fileOutputName)
if not os.path.exists(directory):
    os.makedirs(directory)
    
# Resume functionality
i = 0    
if args.resume:
    try:
        with jsonlines.open(fileOutputName, 'r') as f:
            for line in f:
                i = i + 1
            f.close()    
        print("\n\n\n Checkpoint obtained: ", i, "samples \n\n\n")        
    except FileNotFoundError:
        print("No checkpoint file found, starting from scratch.")
    

################### LLM ############################
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
transformers.logging.set_verbosity_error()
import torch

TOKEN = os.environ['HUGGINGFACE_TOKEN']  # HuggingFace token
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = 'meta-llama/Meta-Llama-3-8B'  
QUANTIZATION_CONFIGS = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type='nf4', 
    bnb_4bit_compute_dtype=torch.bfloat16
)
MODEL_CONFIGS = {
    'torch_dtype': torch.bfloat16,
    #'attn_implementation': 'eager',
    'device_map': DEVICE,
    'token': TOKEN
}

TOKENIZER_CONFIGS = {'token': TOKEN}
tokenizer = AutoTokenizer.from_pretrained(MODEL, **TOKENIZER_CONFIGS)
model = AutoModelForCausalLM.from_pretrained(MODEL, 
                                           **MODEL_CONFIGS, 
                                           #quantization_config=QUANTIZATION_CONFIGS
                                           )

stop_list = [ ".", "\n", "\n\n", "#Question#"]
stop_token_ids = [tokenizer(x, return_tensors='pt', add_special_tokens=False)['input_ids']  for x in stop_list]
stop_token_ids_filtered = []
for x, tokens in zip(stop_list, stop_token_ids):
    if len(x) != len(tokens[0]):
        stop_token_ids_filtered.append(tokens[0][1:])
    else:
        stop_token_ids_filtered.append(tokens[0])

#stop_token_ids_filtered = stop_token_ids_filtered + [torch.tensor([11])] # Comma
stop_token_ids = [torch.LongTensor(x).to(model.device) for x in stop_token_ids_filtered  if len(x) >0]
#print(stop_token_ids)
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        #print(f"Input_ids : {input_ids}")
        
        for stop_ids in stop_token_ids:
            #print(f"Processing input: {model.tokenizer.convert_ids_to_tokens(input_ids[0][-len(stop_ids):])}")
            #print(f"Comparing: [{input_ids[0][-len(stop_ids):]}] with [{stop_ids}]")
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])
####################################################


results = []
# Process judgements
for _ in tqdm(range(len(questions))):
    
    #prepare evidences
    idx = 0
    knowledge_text = ""
    if evidences[i] is not None and evidences[i]:  #all evidences should contain something
        for evid in evidences[i]:
            text = evid["text"]
            knowledge_text = knowledge_text + "#Knowledge-" + str(idx) + "#:" + text +"\n" 
            idx = idx + 1
        
        prompt = main_instruction.format(question=questions[i], answer=answers[i], knowledge = knowledge_text)
        input_encodings = tokenizer(prompt, 
                                  return_tensors='pt'
                                  ).to(DEVICE)
        input_len = input_encodings['input_ids'].shape[-1]
        output = model.generate(**input_encodings, 
                        do_sample=False, 
                        max_new_tokens=5,
                        stopping_criteria=stopping_criteria)
        current_output = tokenizer.decode(output.squeeze()[input_len:])
        print(i, ':', current_output)
        model_factuality_label = None
        
        if "TRUE" in current_output:
           model_factuality_label = True
        elif "FALSE" in current_output:
           model_factuality_label = False
        else:
           model_factuality_label = None
                
        output_dict = {
        "ID": IDs[i],
        "Type": types[i],
        "Category": categories[i],
        "Question": questions[i],
        "Answer": answers[i],
        "Source": sources[i],
        "Evidences": evidences[i],
        "Factuality_ground_label": real_fact_labels[i],
        "Model_factuality_judgement": current_output,
        "Model_factuality_label": model_factuality_label
        }     
        
        results.append(output_dict)
        
    else:  #evidence empty: that is not supposed to happen  
      print("ERROR: Evidence empty. ID:", IDs[i])
      break
    

    # Save intermediate results
    if (i + 1) % args.save_freq == 0 or i == len(questions) - 1:
        with jsonlines.open(fileOutputName, 'a') as writer:
            for result in results:
                writer.write(result)
            writer.close()
            print("File written")
            results = []
            
    i = i + 1        