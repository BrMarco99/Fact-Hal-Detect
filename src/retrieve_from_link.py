# given a claim, return a list of related evidence
import json
import os
from typing import List, Tuple
import time
from tqdm import tqdm
from transformers import pipeline
from sentence_transformers import CrossEncoder
import spacy
import numpy as np
from copy import deepcopy
import torch
from collections import Counter
import requests

import bs4
from typing import List, Dict, Any
import argparse
import pandas as pd
import jsonlines

def is_tag_visible(element: bs4.element) -> bool:
    """Determines if an HTML element is visible.

    Args:
        element: A BeautifulSoup element to check the visiblity of.
    returns:
        Whether the element is visible.
    """
    if element.parent.name in [
        "style",
        "script",
        "head",
        "title",
        "meta",
        "[document]",
    ] or isinstance(element, bs4.element.Comment):
        return False
    return True


def scrape_url(url: str, timeout: float = 5) -> Tuple[str, str]:
    """Scrapes a URL for all text information.

    Args:
        url: URL of webpage to scrape.
        timeout: Timeout of the requests call.
    Returns:
        web_text: The visible text of the scraped URL.
        url: URL input.
    """
    # Scrape the URL
    try:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:100.0) Gecko/20100101 Firefox/100.0"}
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as _:
        return None, url

    # Extract out all text from the tags
    try:
        soup = bs4.BeautifulSoup(response.text, "html.parser")
        texts = soup.findAll(text=True)
        # Filter out invisible text from the page.
        visible_text = filter(is_tag_visible, texts)
    except Exception as _:
        return None, url

    # Returns all the text concatenated as a string.
    web_text = " ".join(t.strip() for t in visible_text).strip()
    # Clean up spacing.
    web_text = " ".join(web_text.split())
    return web_text, url

def chunk_text(
    text: str,
    tokenizer,
    sentences_per_passage: int = 5,
    filter_sentence_len: int = 250,
    sliding_distance: int = 2,
) -> List[str]:
    """Chunks text into passages using a sliding window.

    Args:
        text: Text to chunk into passages.
        sentences_per_passage: Number of sentences for each passage.
        filter_sentence_len: Maximum number of chars of each sentence before being filtered.
        sliding_distance: Sliding distance over the text. Allows the passages to have
            overlap. The sliding distance cannot be greater than the window size.
    Returns:
        passages: Chunked passages from the text.
    """
    if not sliding_distance or sliding_distance > sentences_per_passage:
        sliding_distance = sentences_per_passage
    assert sentences_per_passage > 0 and sliding_distance > 0

    passages = []
    try:
        doc = tokenizer(text[:500000])  # Take 500k chars to not break tokenization.
        sents = [
            s.text.replace("\n", " ")
            for s in doc.sents
            if len(s.text) <= filter_sentence_len  # Long sents are usually metadata.
        ]
        for idx in range(0, len(sents), sliding_distance):
            passages.append((" ".join(sents[idx : idx + sentences_per_passage]), idx, idx + sentences_per_passage-1))
    except UnicodeEncodeError as _:  # Sometimes run into Unicode error when tokenizing.
        print("Unicode error when using Spacy. Skipping text.")

    return passages

def get_relevant_snippets(query, source_link, tokenizer, passage_ranker, timeout=10, max_passages=5, sentences_per_passage=5):
    
    #get the text from the URL
    web_text, _ = scrape_url(source_link)
    retrieved_passages = list()
    #divide the text in passages
    if web_text != None:
        passages = chunk_text(text=web_text, tokenizer=tokenizer, sentences_per_passage=sentences_per_passage)
        if passages:
            # Score the passages by relevance to the query using a cross-encoder.
            scores = passage_ranker.predict([(query, p[0]) for p in passages]).tolist()
            passage_scores = list(zip(passages, scores))
    
            # Take the top passages_per_search passages for the current search result.
            passage_scores.sort(key=lambda x: x[1], reverse=True)
    
            relevant_items = list()
            #sliding window and passage selection
            for passage_item, score in passage_scores:
                overlap = False
                if len(relevant_items) > 0:                
                    for item in relevant_items:
                        if passage_item[1] >= item[1] and passage_item[1] <= item[2]:
                            overlap = True
                            break
                        if passage_item[2] >= item[1] and passage_item[2] <= item[2]:
                            overlap = True
                            break
    
                # Only consider top non-overlapping relevant passages to maximise for information 
                if not overlap:
                    relevant_items.append(deepcopy(passage_item))
                    retrieved_passages.append(
                        {
                            "text": passage_item[0],
                            "url": source_link,
                            "sents_per_passage": sentences_per_passage,
                            "retrieval_score": score,  # Cross-encoder score as retr score
                        }
                    )
                if len(relevant_items) >= max_passages:
                    break
                
    return retrieved_passages


# Argument parsing
parser = argparse.ArgumentParser(description="Relevance checking script for QA.")
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--save_freq", type=int, default=500, help="Frequency of saving checkpoints")
parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
args = parser.parse_args()

files_path = "../files"
output_path = "output"
fileInputName = os.path.join(files_path, "TruthfulQA", args.input)
fileOutputName = os.path.join(output_path, args.output)
df = pd.read_json(fileInputName, lines=True)
  
print ("OPENING: ", fileInputName)
#print ("PROMPT:", filePromptName)
print("\n\n\n") 

print ("Writing in:", fileOutputName)    
directory = os.path.dirname(fileOutputName)
if not os.path.exists(directory):
    os.makedirs(directory)

IDs = df['ID'].tolist()
types = df['Type'].tolist()
categories = df['Category'].tolist()
questions = df['Question'].tolist()
answers = df['Answer'].tolist()
sources = df['Source'].tolist()
real_fact_labels = df['Factuality_ground_label'].tolist()
    
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


tokenizer = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer"])
passage_ranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)   
results = []

for _ in tqdm(range(len(questions)-i)):
        
    evidences = get_relevant_snippets(answers[i], sources[i], tokenizer, passage_ranker, timeout=10, max_passages=5, sentences_per_passage=5)
    output_dict = {
    "ID": IDs[i],
    "Type": types[i],
    "Category": categories[i],
    "Question": questions[i],
    "Answer": answers[i],
    "Source": sources[i],
    "Factuality_ground_label": real_fact_labels[i],
    "Evidences": evidences
    }     
    
    results.append(output_dict)
    
    # Save intermediate results
    if (i + 1) % args.save_freq == 0 or i == len(questions) - 1:
        with jsonlines.open(fileOutputName, 'a') as writer:
            for result in results:
                writer.write(result)
            writer.close()
            print("File written")
            results = []
            
    i = i + 1    