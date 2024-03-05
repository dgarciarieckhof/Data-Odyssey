import os
import re
import numpy as np
from tqdm import tqdm
import pdfplumber

# Get text from pdf files
path_in = os.path.join('data','inputs','raw')
path_out = os.path.join('data','outputs')
files = os.listdir(path_in)

notas = {}
for file in tqdm(files):
    pdf_path = os.path.join(path_in,file)
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
            # Extract sentences from text
            text_ = [' '.join(t.split('\n')) for t in text.split('.')]
            text_ = [t.strip() for t in text_]
            text_ = [t.lower() for t in text_ if len(t) > 20]
    notas[file] = text_