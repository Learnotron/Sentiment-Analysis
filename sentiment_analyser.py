import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from afinn import Afinn

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class sentiment_predictor:
    def __init__(self, text):
        self.text = text
        self.final_text_list = self.text_preprocessor()
        
    
    def text_preprocessor(self):
        
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', self.text)
        
        cleaned_text = cleaned_text.lower()

        tokenized_text = word_tokenize(cleaned_text)
        
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in tokenized_text if word.lower() not in stop_words]

        lemmatizer = WordNetLemmatizer()  
        lemmatized_words = [lemmatizer.lemmatize(word, pos=wordnet.VERB) for word in filtered_words]
        
        return lemmatized_words
    
    def sentiment_analyser(self):
        afinn = Afinn()
        
        total_score = sum([afinn.score(token) for token in self.final_text_list])
        
        if total_score > 0:
            sentiment = "Positive"
        elif total_score < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        return sentiment
    
        