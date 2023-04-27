import os
import spacy

class Tokenizer:
    """
    Tokenizer class that uses the spaCy library for tokenizing / lemmatizing text.
    This class initializes and provides methods for tokenizing text.
    """

    def __init__(self):
        """
        Initializes the Tokenizer class by setting up the local path and loading the spaCy model.
        """
        # Get local directory
        cwd = os.getcwd()
        self.local_path = (
            cwd.replace("\\", "/") + "/supreme_court_predictions/data/clean_convokit/"
        )
        print(f"Working in {self.local_path}")

        # Set output path
        self.output_path = self.local_path.replace("convokit", "clean_convokit")
        print(f"Data will be saved to {self.output_path}")

        try:
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        except:
            print("Spacy not present. Downloading files.")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def main():
    token_inst = Tokenizer()

if __name__ == "__main__":
    main()