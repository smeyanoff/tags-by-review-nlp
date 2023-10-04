# tags-by-review-nlp

## description
Цель данного проекта, получить пайплайн, способный доставать краткие теги из отзывов с сервиса Trapadvisor. В проекте используется [тюненая модель roberta](https://huggingface.co/AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru?doi=true), а также методы NLP.

## start

Работоспособность тестировалась на linux ubuntu22.04 WSL
Архитектура процессора x86_64
GPU 4GB

Может потребоваться сменить url загрузки torch в файле `pyproject.toml` на совместимую с вашей системой. Ознакомится можно по [ссылке](https://pytorch.org/get-started/locally/).

Сначала необходимо установить зависимости и настроить окружение
```
make start
```

Опционально (только для Levart):
Для использования dvc с minio необходимо настроить локальные креды:
```
dvc remote modify --local minio access_key_id "accessKey"
dvc remote modify --local minio secret_access_key "secretKey"
```
После можно загрузить данные:
```
dvc pull
```

## Повторение эксперимента
Для того, чтобы воспроизвести эксперимент воспользуйтесь командой
```
make reproduce_experiment
```

## Пример использования
В модель передается корпус текста и вопрос, результатом работы является ответ на вопрос из текста. После softmax, получаю некоторые "вероятности". Регулируя cutoff, можно выкидывать теги, в которых модель не уверена, тогда возвращается пустой list.
 ![pres](https://github.com/smeyanoff/tags-by-review-nlp/assets/108741347/19357663-d8c8-416f-928c-f6e0437634ba)
