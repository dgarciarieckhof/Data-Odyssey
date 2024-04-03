import os
import re
import nltk
import chardet
import pdfplumber
import unicodedata
import numpy as np
import pandas as pd
from tqdm import tqdm

# Get text from pdf files
path_in = os.path.join('data','inputs','raw')
path_out = os.path.join('data','inputs','processed')
files = os.listdir(path_in)

notas = {}
for file in tqdm(files):
    pdf_path = os.path.join(path_in,file)
    filedt = file[17:-7]+"-01"
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
            # Extract sentences from text
        text_ = re.split(r'[.]\n|[:]\n|\n\d[.]',text)
        text_ = [re.sub(r'\n',' ',t) for t in text_]
        text_ = [t.strip() for t in text_]
    notas[filedt] = text_

# Clean text from pdf files
ids = []
notas_inf = []
for idx, val in tqdm(notas.items()):
    ids.append(idx)
    notas_inf.append(val)

notas_df = pd.DataFrame(columns=['ID','NOTA'])
notas_df['ID'] = ids
notas_df['ID'] = pd.to_datetime(notas_df['ID'],format='%Y-%m-%d').dt.strftime('%Y-%m')
notas_df['NOTA'] = notas_inf
notas_df = notas_df.explode('NOTA')

# Remove numerators at the beggining of a sentence
notas_df['NOTA'] = notas_df['NOTA'].apply(lambda x: re.sub(r'^(\d+\.\s|\w+\.\s)', '', x, flags=re.MULTILINE))

# Remove numerator within the text
notas_df['NOTA'] = notas_df['NOTA'].apply(lambda x: re.sub(r' i\)|i\)', '', x))
notas_df['NOTA'] = notas_df['NOTA'].apply(lambda x: re.sub(r' ii\)|ii\)', '', x))
notas_df['NOTA'] = notas_df['NOTA'].apply(lambda x: re.sub(r' iii\)|iii\)', '', x))
notas_df['NOTA'] = notas_df['NOTA'].apply(lambda x: re.sub(r' iv\)|iv\)', '', x))
notas_df['NOTA'] = notas_df['NOTA'].apply(lambda x: re.sub(r' v\)iv\)', '', x))

# Remove accents
def f_remove_accents(old):
    """
    Removes common accent characters, lower form.
    Uses: regex.
    """
    new = old.lower()
    new = re.sub(r'[àáâãäå]', 'a', new)
    new = re.sub(r'[èéêë]', 'e', new)
    new = re.sub(r'[ìíîï]', 'i', new)
    new = re.sub(r'[òóôõö]', 'o', new)
    new = re.sub(r'[ùúûü]', 'u', new)
    new = re.sub(r'[ñ]', 'ni', new)
    return new

notas_df[f'NOTA_TK'] = notas_df['NOTA'].apply(lambda x: f_remove_accents(x))

# Remove currency symbols 
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r's/\.', '', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'us\$', '', x))

# Remove footnotes numerals
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r' 1\/|1\/', '', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r' 2\/|2\/', '', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r' 3\/|3\/', '', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r' 4\/|4\/', '', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r' 5\/5\/', '', x))

# Split the sentences into sub-sentences to avoid lengthy sentences
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: x.split('.'))
notas_df = notas_df.explode('NOTA_TK')

# Remove sentences with mainly numbers
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'\s+', ' ', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: x.strip())
notas_df = notas_df[notas_df['NOTA_TK'].apply(lambda x: len(x)) != 0]

def numbers_presence(text):
    # Split the sentence into words
    words = text.split()
    # Count the total number of words
    total_words = len(words)
    # Count the number of words containing numbers
    words_with_numbers = sum(1 for word in words if re.search(r'\d', word))
    # Calculate the percentage
    if total_words == 0:
        return 0
    percentage = (words_with_numbers / total_words) * 100
    return percentage    

notas_df['FLAG'] = notas_df['NOTA_TK'].apply(lambda x: numbers_presence(x))
notas_df = notas_df[notas_df['FLAG']<22]
del notas_df['FLAG']

# Remove commas, change commas between two numbers for points
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r"(\d+),(\d+)", r"\1comma\2", x, flags=re.IGNORECASE))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'comma', r'.', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'\s+', ' ', x))

# Remove sentences with less than 15 words
notas_df['FLAG'] = notas_df['NOTA_TK'].apply(lambda x: x.split())
notas_df['FLAG'] = notas_df['FLAG'].apply(lambda x: sum(1 for word in x if not re.search(r'\d', word)))
notas_df = notas_df[notas_df['FLAG']>=15].reset_index(drop=True)
del notas_df['FLAG']

# Remove numbers and words that are together like 2001banco
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'(\d+)([a-zA-Z]+)|([a-zA-Z]+)(\d+)', r'\1\3 \2\4', x))

# Mask dates to avoid learning specific or spurious patterns
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'\b\d{1,2}\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4}\b',' [DATE] ',x,flags=re.IGNORECASE))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre|anio)\s\d{4}\b',' [DATE] ',x,flags=re.IGNORECASE))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'\b(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre|anio)\s+de\s+\d{4}\b',' [DATE] ',x,flags=re.IGNORECASE))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'\b\d{1,2}\s+de\s+(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\b',' [DATE] ',x,flags=re.IGNORECASE))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'(?i)(enero|febrero|marzo|abril|mayo|junio|julio|agosto|setiembre|septiembre|octubre|noviembre|diciembre)\s+',' [DATE] ',x,flags=re.IGNORECASE))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'\[DATE\](?:\s*\[DATE\])+',' [DATE] ',x,flags=re.IGNORECASE))

# Mask numbers to avoid learning specific or spurious patterns
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'\d+(?:\.\d+)?\s*por\s*ciento', '[NUM]', x, flags=re.IGNORECASE))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'\d+\s?(millones|miles)', '[NUM]', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'\d+','[NUM]',x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'\[NUM\](?:\.*\[NUM\])+',' [NUM] ',x,flags=re.IGNORECASE))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'\[NUM\](?:\s*\[NUM\])+',' [NUM] ',x,flags=re.IGNORECASE))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'\s+', ' ', x))

# Remove common redundant strings, unlike many NLP algorithms, deep-learning algorithms such as BERT do not require preprocessed inputs (e.g., 
# removing stop words and punctuation, stemming, and lemmatizing). Instead, entire raw sentences are taken as inputs. We have thus kept preprocessing at a 
# minimum by only separating each speech into individual sentences.
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub('cid173', '', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(' i ', '', x))    
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(' s/', '', x))    
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub('[(]s/', '', x))            
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub('U0080U0099', '', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub('U0080U0091', '', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub('U0080U0094', '', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub('U0080', '', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub('U0099', '', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub('U0093', '', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub('U009', '', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'\([^)]*\)', '', x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'\b\d+/\s*|\b\w+\)\s*', '', x, flags=re.IGNORECASE))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'\s+', ' ', x))

# Fix specific words to improve text readability 
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'bcrp|bcr','banco central de reserva del peru',x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'ipc','indice de precios al consumidor',x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'pbi','producto bruto interno',x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'soles|dolar','divisas',x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'pbs','puntos basicos',x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r' ii | iii | iv ','',x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r' ii| iii| iv','',x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'cdbcrp|cdrbcrp','certificados de deposito del banco central de reserva del peru',x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'NUMbanco','NUM banco',x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'NUMmeta','NUM meta',x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'NUMprincipalmente','NUM principalmente',x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'NUMmoneda','NUM moneda',x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'deNUM','de NUM',x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'covidNUM','covid NUM',x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'eneroagosto','enero agosto',x))

# Fixing phrases without altering the meaning of the sentence
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'banco central de reserva del peru','banco central',x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'nota informativa sobre el programa monetario','nota informativa programa monetario',x))
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r' de lima metropolitana indice de precios al consumidor','',x))

# Removing specific sentences
notas_df = notas_df[notas_df['NOTA_TK'].apply(lambda x: not bool(re.findall('febrero marzo abril mayo',x)))]
notas_df = notas_df[notas_df['NOTA_TK'].apply(lambda x: not bool(re.findall('junio julio agosto setiembre',x)))]
notas_df = notas_df[notas_df['NOTA_TK'].apply(lambda x: not bool(re.findall('directorio reafirma su compromiso',x)))]
notas_df = notas_df[notas_df['NOTA_TK'].apply(lambda x: not bool(re.findall('aprobara el siguiente programa monetario',x)))]
notas_df = notas_df[notas_df['NOTA_TK'].apply(lambda x: not bool(re.findall('evaluo la posicion de la politica monetaria',x)))]
notas_df = notas_df[notas_df['NOTA_TK'].apply(lambda x: not bool(re.findall('directorio se encuentra especialmente atento',x)))]
notas_df = notas_df[notas_df['NOTA_TK'].apply(lambda x: not bool(re.findall('proxima sesion del directorio en que se evaluara',x)))]
notas_df = notas_df[notas_df['NOTA_TK'].apply(lambda x: not bool(re.findall('directorio se encuentra atento a la proyeccion de la inflacion',x)))]
notas_df = notas_df[notas_df['NOTA_TK'].apply(lambda x: not bool(re.findall('podran ser modificados',x)))]
notas_df = notas_df[notas_df['NOTA_TK'].apply(lambda x: not bool(re.findall('acordo anunciar',x)))]
notas_df['NOTA_TK'] = notas_df['NOTA_TK'].apply(lambda x: re.sub(r'\s+', ' ', x))

# Remove duplicates
notas_df = notas_df.drop_duplicates('NOTA_TK',keep='first').reset_index(drop=True)

# Remove stopwords
stopword = nltk.corpus.stopwords.words(['spanish'])
stopword.append('[NUM]')
stopword.append('[DATE]')
def remove_stopwords(text, stopword):  
    """
    Removes stopwords.
    """
    text = [word for word in text.split(' ') if word not in stopword]
    text = ' '.join(text) 
    return text
    
notas_df['NOTA_WSW'] = notas_df['NOTA_TK'].apply(lambda x: remove_stopwords(x,stopword))
notas_df = notas_df[['ID','NOTA_WSW','NOTA_TK']]
notas_df['NOTA_WSW'] = notas_df['NOTA_WSW'].apply(lambda x: re.sub(r'\s+', ' ', x))

# Rank the sentences
notas_df['RANK'] = 1
notas_df['RANK'] = notas_df.groupby('ID')['RANK'].cumsum()

# -----
# Store data        
notas_df.to_csv(os.path.join(path_out,'notas_prensa_bcrp.csv'),index=False, encoding='utf-8')