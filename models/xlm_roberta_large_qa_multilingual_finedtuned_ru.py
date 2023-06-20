import os
from dotenv import load_dotenv
import re
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch 
from nltk.corpus import stopwords
import spacy


load_dotenv("creds/.env")


class Pipeline:

    def __init__(self):
        self.model_name = "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=os.environ.get("MODEL_WEIGHTS_PATH")
            )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_name,
            cache_dir=os.environ.get("MODEL_WEIGHTS_PATH")
        )
        self.WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
        self._norm_words = ["на",]
        self._nlp_model = spacy.load('ru_core_news_md')

    def tokinize(self, 
                 article_text: str,
                 question: str,
                 ) -> dict:
        """
        return dict: {"inputs": dict, "ids": list} 
        """

        inputs = self.tokenizer.encode_plus(
            question, 
            self.WHITESPACE_HANDLER(article_text), 
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].tolist()[0]

        return {"inputs": inputs, "ids": input_ids}
    
    def add_norm_words(self, word_list: list) -> None:
        """
        Добавляет слова в norm_words. norm_words используется, чтобы
        убрать некоторые слова из корпуса stopwords
        word_list: list
        return None
        """
        for word in word_list:
            self._norm_words.append(word)

    def answer(self,
               article_text: str,
               question: str,
               ) -> str:
        tokinize_dict = self.tokinize(article_text: str,
                                      question: str)
        inputs = tokinize_dict["inputs"]
        input_ids = tokinize_dict["ids"]
        model_evaluate = self.model.forward(**inputs)
        answer_start_scores, answer_end_scores = (self.model.start_logits, 
                                                  self.model.end_logits)
        answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(
                input_ids[answer_start:answer_end]
            )
        )

        return answer

    def _clear_stopwords(self, answer: str) -> list:

        stops = stopwords.words("russian")
        stops = [word for word in stops if word not in self.norm_words]

        corpus = answer.split(" ")
        text_list = " ".join(
            [word for word in corpus if word not in stops]
        ).split(", ")

        return text_list
    
    def _get_part_of_words(self)
