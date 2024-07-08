import os
import setGPU
import pandas as pd
from tqdm import tqdm
import argparse
import json
import jsonlines
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("ciao")
def tensor_to_list(tensor):
    return tensor.detach().cpu().tolist()

def make_serializable(data):
    if isinstance(data, torch.FloatTensor) or isinstance(data, torch.Tensor):
        return tensor_to_list(data)
    elif isinstance(data, (int, float, str, bool)) or data is None:
        return data
    elif isinstance(data, dict):
        return {key: make_serializable(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return [make_serializable(element) for element in data]
    else:
        return str(data)


# Argument parsing
parser = argparse.ArgumentParser(description="Relevance checking script for QA.")
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--save_freq", type=int, default=5, help="Frequency of saving checkpoints")
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
args = parser.parse_args()

fileInputName = f'/content/drive/MyDrive/Tesi/Codice/FINAL/files/{args.input}'
fileOutputName = f'/content/drive/MyDrive/Tesi/Codice/FINAL/files/{args.output}'
df = pd.read_json(fileInputName, lines=True)


print("\n\n\n")
print ("OPENING: ", fileInputName)
print ("Writing in:", fileOutputName)    
directory = os.path.dirname(fileOutputName)
if not os.path.exists(directory):
    os.makedirs(directory)
print("\n\n\n")


IDs = df['ID'].tolist()
types = df['Type'].tolist()
categories = df['Category'].tolist()
questions = df['Question'].tolist()
answers = df['Answer'].tolist()
sources = df['Source'].tolist()
real_fact_labels = df['Factuality_ground_label'].tolist()
evidences = df['Evidences'].tolist()



# Example model and tokenizer initialization (replace with your specific model)
#model_name = "berkeley-nest/Starling-LM-7B-alpha"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model = AutoModelForCausalLM.from_pretrained(model_name, bnb_4bit_compute_dtype=torch.bfloat16, load_in_4bit=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)


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

# Process judgements
results = []
for _ in tqdm(range(len(questions)-i)):
    
    #prepare evidences
    idx = 0
    knowledge_text = ""
    if evidences[i] is not None and evidences[i]:  #all evidences should contain something
        for evid in evidences[i]:
            text = evid["text"]
            knowledge_text = knowledge_text + "#Knowledge-" + str(idx) + "#:" + text +"\n" 
            idx = idx + 1
        
        input_text =  knowledge_text + " " + questions[i] + " " + answers[i]
        print(i,":", input_text) 
        
        inputs = tokenizer(input_text, return_tensors="pt")
        #labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        # Generate output
        with torch.no_grad():
            output = model(**inputs, output_hidden_states = True,output_attentions = True)
        
        output_dict = {
            "ID": IDs[i],
            "Type": types[i],
            "Category": categories[i],
            "Question": questions[i],
            "Answer": answers[i],
            "Source": sources[i],
            "Factuality_ground_label": real_fact_labels[i],
            #"Evidences": evidences[i],
            "input_text": input_text,
            #"loss": output.loss,    più tardi si potrebbe aggiungere, la commento solo per ridurre le dimensioni del file finale
            #"logits": output.logits.tolist(),
            #"past_key_values": [[pkv.tolist() for pkv in layer] for layer in output.past_key_values],
            "hidden_states": [hs[:, -1, :].tolist()[0] for hs in output.hidden_states] if output.hidden_states is not None else None,   #take only for last token for memory issues
            #"attentions": [att.tolist() for att in output.attentions] if output.attentions is not None else None,
        }
        
        serializable_output = make_serializable(output_dict)
        results.append(serializable_output)  
        
    else:  #evidence empty: that is not supposed to happen  
        print("ERROR: Evidence empty. ID:", IDs[i])
        break
    
    # Save intermediate results
    if (i + 1) % args.save_freq == 0 or i == len(questions) - 1:
        with jsonlines.open(fileOutputName, 'a') as writer:
            for result in results:
                writer.write(result)
                print("Done")
            writer.close() 
        print("File written")
        results = []
    
    i = i + 1            
 