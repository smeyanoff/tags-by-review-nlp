"""
В этом модуле реализован пайплайн модели
"""

import re

import numpy as np
import spacy
import torch
from nltk.corpus import stopwords
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from yaml import safe_load

with open("config.yaml", "r") as file:
    config = safe_load(file)["roberta"]


class Pipeline:

    """
    Пайплайн модели roberta
    """

    def __init__(self):
        self.model_name = (
            "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=config["model_weights"],
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_name,
            cache_dir=config["model_weights"],
        )
        self.whitespace_handler = lambda k: re.sub(
            r"\s+", " ", re.sub("\n+", " ", k.strip()),
        )
        self._nlp_model = spacy.load("ru_core_news_md")
        self._stop_pos = config["stop_pos"]
        self._norm_words = config["norm_words"]

    def tokinize(self, article_text: str, question: str) -> dict:
        """
        return dict: {"inputs": dict, "ids": list}
        """

        inputs = self.tokenizer.encode_plus(
            question,
            self.whitespace_handler(article_text),
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].tolist()[0]

        return {"inputs": inputs, "ids": input_ids}

    def answer(
        self,
        article_text: str,
        question: str,
        proba: float = 0.05,
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
        except ValueError:
            return [], ()
        answer_start_scores, answer_end_scores = (
            torch.softmax(model_evaluate.start_logits, dim=1),
            torch.softmax(model_evaluate.end_logits, dim=1),
        )
        max_start_score = torch.max(answer_start_scores)
        max_end_score = torch.max(answer_end_scores)

        if max_start_score >= proba and max_end_score >= proba:
            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1

            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(
                    input_ids[answer_start:answer_end],
                ),
            )
            return (
                answer.lower(),
                (
                    round(max_start_score.item(), 3),
                    round(max_end_score.item(), 3),
                ),
            )
        else:
            return [], ()

    def _clear_stopwords(self, answer: str) -> list:
        stops = stopwords.words("russian")
        stops = [word for word in stops if word not in self._norm_words]

        corpus = answer.split(" ")
        text_list = " ".join(
            [word for word in corpus if word not in stops],
        ).split(", ")

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

    def get_tags(
        self,
        article_text: str,
        question: str,
        proba: float,
    ) -> list[str]:
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
                [part[key]["dep"] == "ROOT" for key in part.keys()]
                for part in parts
            ]
        try:
            logic = np.reshape(logic, -1)
        except ValueError:
            return []
        if all(condition for condition in logic):
            tags = [
                [part[key]["lemma"] for key in part.keys()]
                for part in parts
            ]
        else:
            tags = [[key for key in part.keys()] for part in parts]
            if np.sum(logic) == 1:
                tags = [" ".join(tags[0])]
        return [np.reshape(tags, -1), probas]
