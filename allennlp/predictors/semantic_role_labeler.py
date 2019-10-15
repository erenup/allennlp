from typing import List

from overrides import overrides
from spacy.tokens import Doc

from allennlp.common.util import JsonDict, sanitize, group_by_count
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register("semantic-role-labeling")
class SemanticRoleLabelerPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.SemanticRoleLabeler` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)
    def predict(self, sentence: str) -> JsonDict:
        """
        Predicts the semantic roles of the supplied sentence and returns a dictionary
        with the results.

        .. code-block:: js

            {"words": [...],
             "verbs": [
                {"verb": "...", "description": "...", "tags": [...]},
                ...
                {"verb": "...", "description": "...", "tags": [...]},
            ]}

        Parameters
        ----------
        sentence, ``str``
            The sentence to parse via semantic role labeling.

        Returns
        -------
        A dictionary representation of the semantic roles in the sentence.
        """
        return self.predict_json({"sentence": sentence})

    def predict_tokenized(self, tokenized_sentence: List[str]) -> JsonDict:
        """
        Predicts the semantic roles of the supplied sentence tokens and returns a dictionary
        with the results.

        Parameters
        ----------
        tokenized_sentence, ``List[str]``
            The sentence tokens to parse via semantic role labeling.

        Returns
        -------
        A dictionary representation of the semantic roles in the sentence.
        """
        spacy_doc = Doc(self._tokenizer.spacy.vocab, words=tokenized_sentence)
        for pipe in filter(None, self._tokenizer.spacy.pipeline):
            pipe[1](spacy_doc)

        tokens = [token for token in spacy_doc]
        instances = self.tokens_to_instances(tokens)

        if not instances:
            return sanitize({"verbs": [], "words": tokens})

        return self.predict_instances(instances)

    @staticmethod
    def make_srl_string(words: List[str], tags: List[str]) -> str:
        frame = []
        chunk = []

        for (token, tag) in zip(words, tags):
            if tag.startswith("I-"):
                chunk.append(token)
            else:
                if chunk:
                    frame.append("[" + " ".join(chunk) + "]")
                    chunk = []

                if tag.startswith("B-"):
                    chunk.append(tag[2:] + ": " + token)
                elif tag == "O":
                    frame.append(token)

        if chunk:
            frame.append("[" + " ".join(chunk) + "]")

        return " ".join(frame)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict):
        raise NotImplementedError("The SRL model uses a different API for creating instances.")

    def tokens_whitespace_to_instances(self, tokens):
        #add a special token to distinguished token and subtoken. this is not totally correct, but it should work.
        sentence_joined = " ".join(tokens)
        word2start = {}
        start = 0
        for word in tokens:
            word2start[word] = start
            start += len(word) + 1

        spacy_start2word = {w.idx: w.pos_ for w in self._tokenizer.spacy(sentence_joined)}
        instances: List[Instance] = []
        for i, word in enumerate(tokens):
            pos = spacy_start2word.get(word2start[word], None)
            if pos is None:
                print('wraning: a word don\'t get pos:')
                try:
                    print(sentence_joined)
                except:
                    print('cannot print')
            if pos == "VERB":
                verb_labels = [0 for _ in tokens]
                verb_labels[i] = 1
                instance = self._dataset_reader.text_to_instance(tokens, verb_labels)
                instances.append(instance)
        return instances

    def tokens_to_instances(self, tokens):
        words = [token.text for token in tokens]
        instances: List[Instance] = []
        for i, word in enumerate(tokens):
            if word.pos_ == "VERB":
                verb_labels = [0 for _ in words]
                verb_labels[i] = 1
                instance = self._dataset_reader.text_to_instance(tokens, verb_labels)
                instances.append(instance)
        return instances

    def whitespace_tokenize(self, sentence):
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in sentence:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        return doc_tokens

    def _sentence_to_srl_instances(self, json_dict: JsonDict) -> List[Instance]:
        """
        The SRL model has a slightly different API from other models, as the model is run
        forward for every verb in the sentence. This means that for a single sentence, we need
        to generate a ``List[Instance]``, where the length of this list corresponds to the number
        of verbs in the sentence. Additionally, all of these verbs share the same return dictionary
        after being passed through the model (as really we care about all the frames of the sentence
        together, rather than separately).

        Parameters
        ----------
        json_dict : ``JsonDict``, required.
            JSON that looks like ``{"sentence": "..."}``.

        Returns
        -------
        instances : ``List[Instance]``
            One instance per verb.
        """


        sentence = json_dict["sentence"]
        doc_tokens = self.whitespace_tokenize(sentence)
        # tokens = [self._tokenizer.split_words(token) for token in doc_tokens]
        # tokens = self._tokenizer.split_words(sentence)
        return self.tokens_whitespace_to_instances(tokens=doc_tokens)

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        """
        Expects JSON that looks like ``[{"sentence": "..."}, {"sentence": "..."}, ...]``
        and returns JSON that looks like

        .. code-block:: js

            [
                {"words": [...],
                 "verbs": [
                    {"verb": "...", "description": "...", "tags": [...]},
                    ...
                    {"verb": "...", "description": "...", "tags": [...]},
                ]},
                {"words": [...],
                 "verbs": [
                    {"verb": "...", "description": "...", "tags": [...]},
                    ...
                    {"verb": "...", "description": "...", "tags": [...]},
                ]}
            ]
        """
        # For SRL, we have more instances than sentences, but the user specified
        # a batch size with respect to the number of sentences passed, so we respect
        # that here by taking the batch size which we use to be the number of sentences
        # we are given.
        batch_size = len(inputs)
        instances_per_sentence = [self._sentence_to_srl_instances(json) for json in inputs]

        flattened_instances = [instance for sentence_instances in instances_per_sentence
                               for instance in sentence_instances]

        if not flattened_instances:
            return_dicts: List[JsonDict] = [{"verbs": []} for x in inputs]
            for sen_index, x in enumerate(inputs):
                doc_tokens = self.whitespace_tokenize(inputs[sentence_index]["sentence"])
                instance = self._dataset_reader.text_to_instance(doc_tokens, [0 for _ in doc_tokens])
                wordpieces = instance.fields['metadata'].metadata['wordpieces']
                wordpiece_tags = ["O" for _ in wordpieces]
                return_dicts[sen_index].update({'wordpieces': wordpieces, 'wordpiece_tags': wordpiece_tags, 'verbs': [], 'words': doc_tokens})
            return sanitize(return_dicts)

        # Make the instances into batches and check the last batch for
        # padded elements as the number of instances might not be perfectly
        # divisible by the batch size.
        batched_instances = group_by_count(flattened_instances, batch_size, None)
        batched_instances[-1] = [instance for instance in batched_instances[-1]
                                 if instance is not None]
        # Run the model on the batches.
        outputs = []
        for batch in batched_instances:
            outputs.extend(self._model.forward_on_instances(batch))

        verbs_per_sentence = [len(sent) for sent in instances_per_sentence]
        return_dicts: List[JsonDict] = [{"verbs": []} for x in inputs]

        output_index = 0
        for sentence_index, verb_count in enumerate(verbs_per_sentence):
            if verb_count == 0:
                # We didn't run any predictions for sentences with no verbs,
                # so we don't have a way to extract the original sentence.
                # Here we just tokenize the input again.
                # tokens = self._tokenizer.split_words(inputs[sentence_index]["sentence"])
                doc_tokens = self.whitespace_tokenize(inputs[sentence_index]["sentence"])
                instance = self._dataset_reader.text_to_instance(doc_tokens, [0 for _ in doc_tokens])
                wordpieces = instance.fields['metadata'].metadata['wordpieces']
                wordpiece_tags = ["O" for _ in wordpieces]

                return_dicts[sentence_index]["words"] = doc_tokens
                return_dicts[sentence_index]['verbs'] = []
                return_dicts[sentence_index]['wordpieces'] = wordpieces
                return_dicts[sentence_index]['wordpiece_tags'] = wordpiece_tags
                continue

            for _ in range(verb_count):
                output = outputs[output_index]
                words = output["words"]
                tags = output['tags']
                description = self.make_srl_string(words, tags)
                return_dicts[sentence_index]["words"] = words
                return_dicts[sentence_index]["wordpieces"] = output["wordpieces"]
                return_dicts[sentence_index]["verbs"].append({
                        "verb": output["verb"],
                        "description": description,
                        "tags": tags,
                        "wordpiece_offsets": output['wordpiece_offsets'],
                        "wordpiece_tags": output["wordpiece_tags"],
                        # "srl_bert_last_layer": output["srl_bert_last_layer"]
                })
                output_index += 1

        return sanitize(return_dicts)

    def predict_instances(self, instances: List[Instance]) -> JsonDict:
        outputs = self._model.forward_on_instances(instances)

        results = {"verbs": [], "words": outputs[0]["words"], "wordpieces": outputs[0]["wordpieces"]}
        for output in outputs:
            tags = output['tags']
            description = self.make_srl_string(output["words"], tags)
            results["verbs"].append({
                    "verb": output["verb"],
                    "description": description,
                    "tags": tags,
                    "wordpiece_offsets": output['wordpiece_offsets'],
                    "wordpiece_tags": output["wordpiece_tags"],
                    # "srl_bert_last_layer": output["srl_bert_last_layer"]
            })

        return sanitize(results)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Expects JSON that looks like ``{"sentence": "..."}``
        and returns JSON that looks like

        .. code-block:: js

            {"words": [...],
             "verbs": [
                {"verb": "...", "description": "...", "tags": [...]},
                ...
                {"verb": "...", "description": "...", "tags": [...]},
            ]}
        """
        instances = self._sentence_to_srl_instances(inputs)

        if not instances:
            return sanitize({"verbs": [], "words": self._tokenizer.split_words(inputs["sentence"])})

        return self.predict_instances(instances)
