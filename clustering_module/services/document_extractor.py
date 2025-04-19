import docx
import pymupdf4llm

class Extractor:
    def __init__(self) -> None:
        self.md_read = pymupdf4llm.LlamaMarkdownReader()
    
    def extract_document(self, document: str) -> str:
        """
        Extract text from a document file (PDF or DOCX).
        Args:
            document (str): Path to the document file.
        Returns:
            str: Extracted text from the document file.
        """
        try:
            if document.endswith(".pdf"):
                return self.extract_pdf(document)
            elif document.endswith(".docx"):
                return self.extract_docx(document)
            elif document.endswith(".txt"):
                return self.extract_txt(document)
            else:
                raise ValueError("Unsupported file format. Only PDF and DOCX are supported.")
        
        except Exception as e:
            print(f"Error extracting document: {e}")
            return ""
        
    def extract_pdf(self, document: str) -> str:
        """
        Extract text from a PDF document file.
        Args:
            document (str): Path to the PDF document file.
        Returns:
            str: Extracted text from the PDF document.
        """
        docs = []
        documents = pymupdf4llm.to_markdown(document, page_chunks=True, ignore_images=True)
        for doc in documents:
            text = doc.get("text")
            docs.append(text)
        
        return "\n".join(docs)
    
    def extract_docx(self, document: str) -> str:
        """
        Extract text from a DOCX file.
        Args:
            document (str): Path to the DOCX file.
        Returns:
            str: Extracted text from the DOCX file.
        """
        docs = []
        documents = docx.Document(document)
        for para in documents.paragraphs:
            docs.append(para.text)
        
        return "\n".join(docs)
    
    def extract_txt(self, document: str) -> str:
        """
        Extract text from a TXT file.
        Args:
            document (str): Path to the TXT file.
        Returns:
            str: Extracted text from the TXT file.
        """
        with open(document, "r", encoding="utf-8") as f:
            return f.read()
    

# if __name__ == "__main__":
#     chunking = Extractor()
#     document = "/home/trongnv130/Desktop/Opendataset1.docx"
#     document = "/home/trongnv130/Desktop/Opendataset1.pdf"
#     documents = chunking.extract_document(document)
#     print(documents)
 
    