from dataclasses import dataclass


@dataclass
class ModelsConfig:

    # roberta
    roberta_stop_pos = ['PART',]
    roberta_norm_words = ["на",]
