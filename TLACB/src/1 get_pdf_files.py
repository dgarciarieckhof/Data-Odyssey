import os
import requests
import urllib.request
from tqdm import tqdm
from bs4 import BeautifulSoup

# Get files for "Notas informativas del programa monetario" from Banco Central de Reserva del Peru webpage
url = "https://www.bcrp.gob.pe/politica-monetaria/notas-informativas-del-programa-monetario.html"
folder_path = os.path.join('data','inputs','raw')

# Fetch webpage content
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all links on the webpage
links = soup.find_all('a')
links = soup.find_all('a', text=lambda text: text and 'Nota informativa' in text)

# Create folder if not exists
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Download PDF files
for link in tqdm(links):
    pdf_url = link.get('href')
    if 'pdf' in pdf_url:
        pdf_url = 'https://www.bcrp.gob.pe' + pdf_url    
        pdf_filename = os.path.join(folder_path, pdf_url.split('/')[-1])
        print("Downloading:", pdf_filename)
        urllib.request.urlretrieve(pdf_url, pdf_filename)