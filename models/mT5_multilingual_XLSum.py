import os
from dotenv import load_dotenv
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


load_dotenv("creds/.env")


class Pipeline:

    def __init__(self):
        self.model_name = "csebuetnlp/mT5_multilingual_XLSum"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=os.environ.get("MODEL_WEIGHTS_PATH")
            )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            cache_dir=os.environ.get("MODEL_WEIGHTS_PATH")
        )
        self.WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
    
    def tokinize_text(self, 
                      article_text: str, 
                      max_length: int = 512):
        input_ids = self.tokenizer(
            [self.WHITESPACE_HANDLER(article_text)],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )["input_ids"]

        return input_ids
    
    def summarize(self, 
                  article_text: str,
                  max_length_model: int = 84,
                  max_length_tokinizer: int = 512,
                  no_repeat_ngram_size: int = 2,
                  num_beans: int = 4,
                  skip_special_tokens: bool = True,
                  clean_up_tokenization_spaces: bool = False):

        input_ids = self.tokinize_text(article_text, max_length_tokinizer)

        output_ids = self.model.generate(
            input_ids=input_ids,
            max_length=max_length_model,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=num_beans
        )[0]

        summary = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )

        return summary
