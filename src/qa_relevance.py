import os
import setGPU
import pandas as pd
from tqdm import tqdm
import argparse
import json
import jsonlines
from architectures import LLMCompletion

print("go")
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
#print ("PROMPT:", filePromptName)
print("\n\n\n")


IDs = df['ID'].tolist()
types = df['Type'].tolist()
categories = df['Category'].tolist()
questions = df['Question'].tolist()
answers = df['Answer'].tolist()
real_fact_labels = df['Factuality_ground_label'].tolist()
sources = df['Source'].tolist()
manual_labels = df['manual_checkworthy'].tolist()


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
        
    prompt = main_instruction.format(question=questions[i], answer=answers[i])
    current_output = llm(prompt)
    print(i, ':', current_output)
    model_checkworthy_label = None
    
    if current_output != "":
        if "NONE" in current_output:
            model_checkworthy_label = False
        else:
            model_checkworthy_label = True
            
    output_dict = {
    "ID": IDs[i],
    "Type": types[i],
    "Category": categories[i],
    "Question": questions[i],
    "Answer": answers[i],
    "Factuality_ground_label": real_fact_labels[i],
    "Source": sources[i],
    "manual_checkworthy": manual_labels,
    "model_checkworthiness_judgement": current_output,
    "model_checkworthiness": model_checkworthy_label,
    
    }     
    
    results.append(output_dict)
    
    
    

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
