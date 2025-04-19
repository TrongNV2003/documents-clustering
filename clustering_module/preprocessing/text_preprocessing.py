import regex as re
import unicodedata
from underthesea import word_tokenize

class Preprocessing:
    def __init__(self, force_text=None, stop_word_file: str = None, segment: str = None) -> None:
        self.stop_word_file = stop_word_file
        self.segment = segment
        self.force_text = force_text
    
    def __call__(self, text):
        return self.preprocess(text)
    
    def preprocess(self, text: str) -> str:
        if self.force_text:
            text = self.clean_text(text)
        
        if self.segment:
            text = self.segment_words(text)
        
        if self.stop_word_file:
            text = self.remove_stopwords(text)
            
        return text

    def remove_stopwords(self, tokens: str) -> list:
        with open(self.stop_word_file, "r", encoding="utf-8") as f:
            stopwords = f.read().splitlines()
        
        filtered_words = [word for word in tokens if word not in stopwords]
        return filtered_words

    def segment_words(self, text: str) -> str:
        tokens = word_tokenize(text, format="text")
        text_segment = "".join(tokens)
        return text_segment
    
    def clean_text(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[\x00-\x1F\x7F-\x9F]+", " ", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        text = re.sub(r"[!\"#$%&'*+,\-<=>?@\[\\\]^`{|}~]+" , "", text)     # [!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]+
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
# if __name__ == "__main__":
#     from underthesea import sent_tokenize
    
#     chunking = Extractor()
#     document = "/home/trongnv130/Desktop/Opendataset1.docx"
#     documents = chunking.extract_document(document)
    
#     preprocessor = Preprocessing(force_text=True, segment=True)
    
#     preprocessed_text = preprocessor.preprocess(documents)
    
#     sentences = sent_tokenize(preprocessed_text)

#     print(len(sentences))