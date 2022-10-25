from typing import Callable, Iterator, List, Optional

import utils


class Example:
    def is_valid(self, raise_exception: Optional[bool] = True) -> None:
        raise NotImplementedError


class NERExample:

    def __init__(self, raw: dict) -> None:
        self.raw = raw
        self.id = raw['id']
        self.text = raw['text'].replace('…', '...\n').replace('µ', 'μ').replace("\n\n", "\n") + '\n'
        self.sentences = utils.split_sentences(self.text)
        self.sentence_offsets = utils.get_sentence_offsets(self.text, self.sentences)
        self.sentence_offsets.append(len(self.text))

    @property    
    def labels(self):
        if 'label' in self.raw:
            labels = self.raw['label']

        elif 'labels' in self.raw:
            labels = self.raw['labels']
        else:
            raise KeyError(
                'The file should includes either "labels" or "label".'
            )        
        return labels

    def get_tokens_and_token_offsets(self, tokenizer):
        tokens = [tokenizer(sentence) for sentence in self.sentences]
        token_offsets_sentences = [[id_sentence, 
            utils.get_offsets(sentence, tokens, offset)]
            for (id_sentence, sentence), tokens, offset in zip(
                enumerate(self.sentences), tokens, self.sentence_offsets
            )
        ]
        token_offsets = [token[1] for token in token_offsets_sentences]
        id_sentences = [[token[0] for _ in range(len(token[1]))] for token in token_offsets_sentences]
        return tokens, token_offsets, id_sentences

    def is_valid(self, raise_exception: Optional[bool] = True) -> bool:
        return True

    def to_pandas(
        self, tokenizer: Callable[[str], List[str]]
    ) -> Iterator[dict]:
        all_tokens, all_token_offsets, all_id_sentences = self.get_tokens_and_token_offsets(tokenizer)
        label_split = [[] for _ in range(len(self.sentences))]
        for label in self.labels:
            for i, (start, end) in enumerate(
                    zip(self.sentence_offsets, self.sentence_offsets[1:])):
                if start <= label[0] <= label[1] <= end:
                    label_split[i].append(label)                

        i = 0
        for tokens, offsets, label, id_sentences in zip(
                all_tokens, all_token_offsets, label_split, all_id_sentences):
            tags = utils.create_bio_tags(tokens, offsets, label)
            for token, tag, id_sentence in zip(tokens, tags, id_sentences):
                yield(token, tag, int(self.id)*1000 + int(id_sentence))
            i += 1

