import os
import setGPU
import pandas as pd
from tqdm import tqdm
import argparse
import json
import jsonlines
from architectures import LLMCompletion

print("ciaao")
# Argument parsing
parser = argparse.ArgumentParser(description="Relevance checking script for QA.")
parser.add_argument("--model", type=str, default='Starling-LM-7B-alpha', help="Model to be used for LLMChat")
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--prompt", type=str)
parser.add_argument("--save_freq", type=int, default=10, help="Frequency of saving checkpoints")
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
args = parser.parse_args()

fileInputName = f'/content/drive/MyDrive/Tesi/Codice/FINAL/files/{args.input}'
filePromptName = f'/content/drive/MyDrive/Tesi/Codice/FINAL/prompts/{args.prompt}'
fileOutputName = f'/content/drive/MyDrive/Tesi/Codice/FINAL/files/{args.output}'
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

# Initialize LLMChat model
llm = LLMCompletion(args.model)



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
    



results = []
# Process judgements
for _ in tqdm(range(len(questions)-i)):
    
    #prepare evidences
    idx = 0
    knowledge_text = ""
    if evidences[i] is not None and evidences[i]:  #all evidences should contain something
        for evid in evidences[i]:
            text = evid["text"]
            knowledge_text = knowledge_text + "#Knowledge-" + str(idx) + "#:" + text +"\n" 
            idx = idx + 1
        
        prompt = main_instruction.format(question=questions[i], answer=answers[i], knowledge = knowledge_text)
        current_output = llm(prompt)
        print(i, ':', prompt, current_output)
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
                print("Done")
            writer.close()
            print("File written")
            results = []
            
    i = i + 1        