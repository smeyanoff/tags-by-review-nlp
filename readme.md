# tags-by-review-nlp

## description
Цель данного проекта, получить пайплайн, способный доставать краткие теги из отзывов с сервиса Trapadvisor. В проекте используется [тюненая модель roberta](https://huggingface.co/AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru?doi=true), а также методы NLP.

## Пример использования
В модель передается корпус текста и вопрос, результатом работы является ответ на вопрос из текста. После softmax, получаю некоторые "вероятности". Регулируя cutoff, можно выкидывать теги, в которых модель не уверена, тогда возвращается пустой list.
 ![pres](https://github.com/smeyanoff/tags-by-review-nlp/assets/108741347/19357663-d8c8-416f-928c-f6e0437634ba)
