import re
import contractions
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup


def check_nltk_packages():
        packages = ['punkt','stopwords','omw-1.4','wordnet']

        for package in packages:
            try:
                nltk.data.find('tokenizers/' + package)
            except LookupError:
                nltk.download(package)

    # Hace text wrnagling por si acaso hay alguna URL
def wrangle_text(text):
        soup = BeautifulSoup(text,"lxml")
        new_text = soup.get_text()
        new_text = re.sub(r'https://\S+|www\.\S+','',new_text)
        new_text=contractions.fix(new_text)
        return new_text

    #Realiza text_wrangling, tokeniza por palabras, convierte a minusculas quitando caracteres alfanumericos
    #lemmatiza, y quita stopwords. Además, hace n-gramas
def prepare_data(text):
        stopwords_en = stopwords.words('english')
        wnl = WordNetLemmatizer()
        

        wrangled_text= wrangle_text(text) 
        project_tokens = wordpunct_tokenize(text) 
        project_tokens_filtered = [wd.lower() for wd in project_tokens if wd.isalnum()] 
        lemmatized_project = [wnl.lemmatize(el) for el in project_tokens_filtered] 
        clean_review = [wd for wd in lemmatized_project if wd not in stopwords_en] 

        return clean_review