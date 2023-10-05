"""
В этом модуле реализован пайплайн модели
"""

import re

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
        except RuntimeError:
            return [], ()
        except IndexError:
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
        try:
            answer = answer.split(">")[1]
        except IndexError:
            pass
        stops = stopwords.words("russian")
        stops = [word for word in stops if word not in self._norm_words]

        corpus = answer.split(" ")
        text = " ".join([word for word in corpus if word not in stops])
        text_list = re.split(r', |\. |\.|,| а | и | просто | или ', text)

        return text_list

    def _get_parts(self, answer_cleared: list) -> (list[dict], list):
        parts = []
        roots_dep = []
        for answer in answer_cleared:
            document = self._nlp_model(answer)
            doc_dict = {}
            for token in document:
                if token.pos_ not in "PUNCT":
                    doc_dict[token.text] = {
                        "lemma": token.lemma_,
                        "pos": token.pos_,
                        "dep": token.dep_,
                    }
                    if token.dep_ == "ROOT":
                        roots_dep.append(token.pos_)
        return (parts, roots_dep)

    def _return_tag(self, part: dict, root_dep: str) -> str:
        tags = []
        # для того, чтобы оставить только главное существительное
        root_stop = False
        # для того, чтобы возврщать слова из предложения, а не леммы
        return_keys = False
        single_word = True if len(part.keys()) == 1 else False
        for num, key in enumerate(part.keys()):

            if single_word:
                tags.append(part[key]["lemma"])
                continue

            if root_stop:
                break

            # should return keys?
            if (
                num == 0 and (
                    (
                        part[key]["dep"] == "case"
                        or part[key]["dep"] == "advmod"
                    )
                    or (part[key]["dep"] == "nsubj")
                    or (
                        part[key]["pos"] == 'CCONJ'
                        and part[key]["dep"] == 'cc' and num == 0
                    )
                    or (part[key]["pos"] == 'ROOT' and num == 0)
                )
            ):
                return_keys = True

            # should pass the word?
            if (
                (
                    part[key]["pos"] == "ADV" and (
                        part[key]["dep"]
                        == "advmod" or part[key]["dep"] == "punct"
                    )
                )
                or (part[key]["dep"] == ['nmod'] and num == 0)
                or (
                    part[key]["pos"] == 'CCONJ'
                    and part[key]["dep"] == 'cc' and num == 0
                )
            ):
                continue

            if root_dep == 'VERB':
                if part[key]["dep"] == "ROOT":
                    if return_keys:
                        tags.append(key)
                    else:
                        tags.append(part[key]["lemma"])
                elif part[key]["dep"] == "mark":
                    continue
                else:
                    if part[key]["dep"] == 'xcomp' and len(tags) == 1:
                        tags.pop(0)
                    tags.append(key)

            elif root_dep == "NOUN":
                if part[key]["dep"] == "ROOT":
                    if return_keys:
                        tags.append(key)
                    else:
                        tags.append(part[key]["lemma"])
                    if num != 0:
                        root_stop = True
                else:
                    tags.append(part[key]["lemma"])

            elif root_dep == "ADJ":
                if part[key]["dep"] == "mark":
                    continue
                else:
                    tags.append(key)

            elif root_dep == "ADV":
                tags.append(key)

            elif root_dep == "NUM":
                tags.append(key)

            else:
                tags.append(part[key]["lemma"])

        return " ".join(tags)

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
        if (
            not answer or
            answer == '' or
            answer == question.lower().replace('?', '')
        ):
            return []
        # clear answer
        answer_cleared = self._clear_stopwords(answer)

        parts, roots_dep = self._get_parts(answer_cleared)

        tags = []
        for part, root_dep in zip(parts, roots_dep):
            if part:
                if len(part.keys()) == 1:
                    tags.append(list(part.keys())[0])
                else:
                    tags.append(self._return_tag(part, root_dep))
        return [tags, probas]
