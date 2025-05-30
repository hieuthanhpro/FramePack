import os
import pandas as pd
import numpy as np
import json
import re


def read_text_file_as_chunks(file_path, chunk_length=3000):
    with open(file_path, "r", encoding="utf8") as file:
        data = file.read()
        # Replace multiple spaces with a single space
        data = re.sub(r'\s+', ' ', data)
        
        # Strip leading and trailing spaces
        data = re.sub(r'^\s+|\s+$', '', data, flags=re.MULTILINE)
        
        # Remove indents (leading spaces)
        data = re.sub(r'^\s+', '', data, flags=re.MULTILINE)
        
        # Replace multiple newlines with a single newline
        data = re.sub(r'\n{2,}', '\n', data)
        chunks = []
        data = " ".join(" ".join(data.split("\n")).strip().split("\t")).split(" ")
        print(len(data))
        # data = re.split(";|\n|\t", data)
        # data = [d.strip() for d in data if d.strip() != ""]
        it = 0
        for it in range(0, len(data), chunk_length):
            chunks.append(" ".join(data[it : it + chunk_length]))
        if it < len(data):
            chunks.append(" ".join(data[it:]))
        return chunks
    

def iterate_text_as_chunks(file_path, chunk_length=500, overlapped_window_sz=100, mode="file"):
    data = None
    if mode == "file":
        with open(file_path, "r", encoding="utf8") as file:
            data = file.read()
    else:
        data = file_path
    # Replace multiple spaces with a single space
    data = re.sub(r'\s+', ' ', data)
    
    # Strip leading and trailing spaces
    data = re.sub(r'^\s+|\s+$', '', data, flags=re.MULTILINE)
    
    # Remove indents (leading spaces)
    data = re.sub(r'^\s+', '', data, flags=re.MULTILINE)
    
    # Replace multiple newlines with a single newline
    data = re.sub(r'\n{2,}', '\n', data)
    chunks = []
    data = " ".join(" ".join(data.split("\n")).strip().split("\t")).split(" ")

    it = 0
    for it in range(0, len(data), overlapped_window_sz):
        chunks.append(" ".join(data[it: it + chunk_length]))
    if it < len(data):
        chunks.append(" ".join(data[it:]))
    return chunks
    

    

def create_json_from_dict(data_dict={}, filename=''):
    data = json.dumps(data_dict, indent=4)
    # remove all parents dir
    filename = filename.split('/')[-1].split(".")[0] + ".json"
    with open("jsons/" + filename, "w") as file:
        file.write(data)


def read_json_from_file(filepath=''):
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f'[EXCEPTION] {filepath} doesn\' exists!')
        return None
    except:
        print(f'[EXCEPTION] Error reading JSON: {filepath}')
        return None

if __name__ == "__main__":
    file_path = os.path.join(
        os.path.dirname(__file__), r"books\alice_in_wonderland.txt"
    )
    chunks = read_text_file_as_chunks(file_path)
    print(len(chunks))
    print(chunks[len(chunks) - 1], len(chunks[len(chunks) - 1]))
