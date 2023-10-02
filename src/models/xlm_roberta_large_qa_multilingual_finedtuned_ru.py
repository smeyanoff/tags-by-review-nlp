import os
import re

import numpy as np
import spacy
import torch
from dotenv import load_dotenv
from nltk.corpus import stopwords
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from models.models_config import ModelsConfig as mc

load_dotenv("creds/.env")


class Pipeline:
    def __init__(self):
        self.model_name = "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=os.environ.get("MODEL_WEIGHTS_PATH")
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_name, cache_dir=os.environ.get("MODEL_WEIGHTS_PATH")
        )
        self.WHITESPACE_HANDLER = lambda k: re.sub(
            "\s+", " ", re.sub("\n+", " ", k.strip())
        )
        self._nlp_model = spacy.load("ru_core_news_md")
        self._stop_pos = mc.roberta_stop_pos
        self._norm_words = mc.roberta_norm_words

    def tokinize(self, article_text: str, question: str) -> dict:
        """
        return dict: {"inputs": dict, "ids": list}
        """

        inputs = self.tokenizer.encode_plus(
            question, self.WHITESPACE_HANDLER(article_text), return_tensors="pt"
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

    def answer(
        self, article_text: str, question: str, proba: float = 0.05
    ) -> str | list:
        """
        article_text: str - review body
        question: str - question to answer
        proba: float - min probability of token position

        return: str | empty list - answer
        """
        tokinize_dict = self.tokinize(article_text, question)
        inputs = tokinize_dict["inputs"]
        input_ids = tokinize_dict["ids"]
        try:
            model_evaluate = self.model.forward(**inputs)
        except:
            return [], ()
        answer_start_scores, answer_end_scores = (
            torch.softmax(model_evaluate.start_logits, dim=1),
            torch.softmax(model_evaluate.end_logits, dim=1),
        )
        max_start_score = torch.max(answer_start_scores)
        max_end_score = torch.max(answer_end_scores)

        if max_start_score >= proba and max_end_score >= proba:
            answer_start = torch.argmax(
                answer_start_scores
            )  # Get the most likely beginning of answer with the argmax of the score
            answer_end = (
                torch.argmax(answer_end_scores) + 1
            )  # Get the most likely end of answer with the argmax of the score

            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(
                    input_ids[answer_start:answer_end])
            )
            return (
                answer.lower(),
                (round(max_start_score.item(), 3), round(max_end_score.item(), 3)),
            )
        else:
            return [], ()

    def _clear_stopwords(self, answer: str) -> list:
        stops = stopwords.words("russian")
        stops = [word for word in stops if word not in self._norm_words]

        corpus = answer.split(" ")
        text_list = " ".join(
            [word for word in corpus if word not in stops]).split(", ")

        return text_list

    def _get_part_of_words(self, answer: str) -> dict:
        document = self._nlp_model(answer)
        doc_dict = {}
        for token in document:
            if token.pos_ not in self._stop_pos:
                doc_dict[token.text] = {
                    "lemma": token.lemma_,
                    "pos": token.pos_,
                    "dep": token.dep_,
                }

        return doc_dict

    def get_tags(self, article_text: str, question: str, proba: float) -> list[str]:
        """
        article_text: str - review body
        question: str - question to answer
        proba: float - min probability of token position

        return: list[str] - list of tags
        """
        answer, probas = self.answer(article_text, question, proba)
        if not answer:
            return []
        answer_cleared = self._clear_stopwords(answer)

        parts = []
        for token in answer_cleared:
            parts.append(self._get_part_of_words(token))
            logic = [
                [part[key]["dep"] == "ROOT" for key in part.keys()] for part in parts
            ]
        try:
            logic = np.reshape(logic, -1)
        except ValueError:
            return []
        if all(condition == True for condition in logic):
            tags = [[part[key]["lemma"] for key in part.keys()]
                    for part in parts]
        else:
            tags = [[key for key in part.keys()] for part in parts]
            if np.sum(logic) == 1:
                tags = [" ".join(tags[0])]
        return [np.reshape(tags, -1), probas]
