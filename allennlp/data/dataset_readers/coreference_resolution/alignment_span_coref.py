import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set

from overrides import overrides
import glob
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, ListField, TextField, SpanField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, enumerate_spans
import re
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def canonicalize_clusters(clusters: DefaultDict[int, List[Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
    """
    The CONLL 2012 data includes 2 annotated spans which are identical,
    but have different ids. This checks all clusters for spans which are
    identical, and if it finds any, merges the clusters containing the
    identical spans.
    """
    merged_clusters: List[Set[Tuple[int, int]]] = []
    for cluster in clusters.values():
        cluster_with_overlapping_mention = None
        for mention in cluster:
            # Look at clusters we have already processed to
            # see if they contain a mention in the current
            # cluster for comparison.
            for cluster2 in merged_clusters:
                if mention in cluster2:
                    # first cluster in merged clusters
                    # which contains this mention.
                    cluster_with_overlapping_mention = cluster2
                    break
            # Already encountered overlap - no need to keep looking.
            if cluster_with_overlapping_mention is not None:
                break
        if cluster_with_overlapping_mention is not None:
            # Merge cluster we are currently processing into
            # the cluster in the processed list.
            cluster_with_overlapping_mention.update(cluster)
        else:
            merged_clusters.append(set(cluster))
    return [list(c) for c in merged_clusters]


@DatasetReader.register("alignment_span")
class AlignmentSpan(DatasetReader):
    """
    Reads a single CoNLL-formatted file. This is the same file format as used in the
    :class:`~allennlp.data.dataset_readers.semantic_role_labelling.SrlReader`, but is preprocessed
    to dump all documents into a single file per train, dev and test split. See
    scripts/compile_coref_data.sh for more details of how to pre-process the Ontonotes 5.0 data
    into the correct format.

    Returns a ``Dataset`` where the ``Instances`` have four fields: ``text``, a ``TextField``
    containing the full document text, ``spans``, a ``ListField[SpanField]`` of inclusive start and
    end indices for span candidates, and ``metadata``, a ``MetadataField`` that stores the instance's
    original text. For data with gold cluster labels, we also include the original ``clusters``
    (a list of list of index pairs) and a ``SequenceLabelField`` of cluster ids for every span
    candidate.

    Parameters
    ----------
    max_span_width: ``int``, required.
        The maximum width of candidate spans to consider.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        This is used to index the words in the document.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, data_dir: str):
        source_files = glob.glob(data_dir +  "/*/source/character_tokenized/*.cmn.tkn")
        target_files = glob.glob(data_dir + "/*/translation/tokenized/*.eng.tkn")
        wa_files = glob.glob(data_dir + "/*/WA/character_aligned/*.wa")
        assert len(source_files) == len(target_files)
        assert len(source_files) == len(wa_files)
        for source_file, target_file, wa_file in zip(source_files, target_files, wa_files):
            f1 = source_file.split('/')[-1].split('.')[0]
            f2 = target_file.split('/')[-1].split('.')[0]
            f3 = wa_file.split('/')[-1].split('.')[0]
            assert f1 == f2
            assert f2 == f3

            chinese_lines = []
            english_lines = []
            wa = []
            with open(source_file, 'r', encoding='utf-8') as fin:
                chinese_lines = fin.readlines()
            with open(target_file, 'r', encoding='utf-8') as fin:
                english_lines = fin.readlines()
            with open(wa_file, 'r', encoding='utf-8') as fin:
                wa = fin.readlines()
            print('len wa:', len(wa))
            assert len(wa) == len(chinese_lines)
            assert len(wa) == len(english_lines)

            for data in zip(chinese_lines, english_lines, wa):
                if 'rejected' in data[2]:
                    continue
                chinese = data[0].split()
                english = data[1].split()
                wa = data[2].split()
                example = {'chinese': chinese, 'english': english}
                spans = []
                source = {}
                target = {}
                for index, w in enumerate(wa):
                    w1 = w.split('-')
                    source[index] = {'span': [], 'ctype': [], 'link_type': None}
                    target[index] = {'span': [], 'ctype': [], 'link_type': None}
                    if len(w1) == 2:
                        # source = [int(id_) for id_ in re.findall(r'\d+', w1[0])]
                        # target = [int(id_) for id_ in re.findall(r'\d+', w1[1])]
                        if w1[0]:
                            for c in w1[0].split(','):
                                if c.isdigit():
                                    source[index]['span'].append(int(c))
                                elif '[' in c and ']' in c:
                                    c = c.strip(']')
                                    intc = int(c.split('[')[0])
                                    ctype = c.split('[')[1]
                                    source[index]['ctype'].append({intc: ctype})
                                    source[index]['span'].append(intc)
                                else:
                                    print('wrong format:', w1)
                        if w1[1]:
                            for c in w1[1].split(','):
                                if c.isdigit():
                                    target[index]['span'].append(int(c))
                                elif ']' in c and ']' in c and '(' not in c:
                                    c = c.strip(']')
                                    intc = int(c.split('[')[0])
                                    ctype = c.split('[')[1]
                                    target[index]['ctype'].append({intc: ctype})
                                    target[index]['span'].append(intc)
                                elif '(' in c and ')' in c and not '[' in c:
                                    if c.split('(')[0]:
                                        intc = int(c.split('(')[0])
                                        target[index]['span'].append(intc)
                                    if not target[index]['link_type']:
                                        target[index]['link_type'] = c[c.index('(') + 1: c.index(')')]
                                    else:
                                        print('link type error:', w)
                                elif '[' in c and ']' in c and '(' in c and ')' in c:
                                    intc = int(c.split('[')[0])
                                    ctype = c[c.index('[') + 1:c.index(']')]
                                    target[index]['ctype'].append({intc: ctype})
                                    target[index]['span'].append(intc)
                                    if not target[index]['link_type']:
                                        target[index]['link_type'] = c[c.index('(') + 1: c.index(')')]
                                    else:
                                        print('link type error:', w)
                                else:
                                    print('wrong format:', w1)
                    else:
                        print('wrong format:', w)
                for index, value in source.items():
                    source_spans = value['span']
                    source_words = [chinese[x - 1] for x in source_spans]
                    target_spans = target[index]['span']
                    target_words = [english[x - 1] for x in target_spans]
                    spans.append([source_spans, target_spans, source_words, target_words])

                example.update({'spans': spans})
                example.update({'source': source})
                example.update({'target': target})
                yield self.text_to_instance(source=example['chinese'], target=example['english'], gold_clusters=example['spans'], example=example)


        # --chinese_file
        # data / parallel_word_aligned_treebank / bc / source / character_tokenized / CCTV4_ACROSSCHINA_CMN_20050812_150501.cmn.tkn
        # --english_file
        # data / parallel_word_aligned_treebank / bc / translation / tokenized / CCTV4_ACROSSCHINA_CMN_20050812_150501.eng.tkn
        # --wa_file
        # data / parallel_word_aligned_treebank / bc / WA / character_aligned / CCTV4_ACROSSCHINA_CMN_20050812_150501.wa

        # ontonotes_reader = Ontonotes()
        # for sentences in ontonotes_reader.dataset_document_iterator(file_path):
        #     clusters: DefaultDict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
        #
        #     total_tokens = 0
        #     for sentence in sentences:
        #         for typed_span in sentence.coref_spans:
        #             # Coref annotations are on a _per sentence_
        #             # basis, so we need to adjust them to be relative
        #             # to the length of the document.
        #             span_id, (start, end) = typed_span
        #             clusters[span_id].append((start + total_tokens,
        #                                       end + total_tokens))
        #         total_tokens += len(sentence.words)
        #
        #     canonical_clusters = canonicalize_clusters(clusters)
        #     yield self.text_to_instance([s.words for s in sentences], canonical_clusters)

    @overrides
    def text_to_instance(self,  # type: ignore
                         source, target, gold_clusters, example) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        sentences : ``List[List[str]]``, required.
            A list of lists representing the tokenised words and sentences in the document.
        gold_clusters : ``Optional[List[List[Tuple[int, int]]]]``, optional (default = None)
            A list of all clusters in the document, represented as word spans. Each cluster
            contains some number of spans, which can be nested and overlap, but will never
            exactly match between clusters.

        Returns
        -------
        An ``Instance`` containing the following ``Fields``:
            text : ``TextField``
                The text of the full document.
            spans : ``ListField[SpanField]``
                A ListField containing the spans represented as ``SpanFields``
                with respect to the document text.
            span_labels : ``SequenceLabelField``, optional
                The id of the cluster which each possible span belongs to, or -1 if it does
                 not belong to a cluster. As these labels have variable length (it depends on
                 how many spans we are considering), we represent this a as a ``SequenceLabelField``
                 with respect to the ``spans ``ListField``.
        """
        source = ['cls'] + source[1:]
        target = ['sep'] + target[1:]
        sentences = [source, target]
        flattened_sentences = source + target
        metadata: Dict[str, Any] = {"original_text": flattened_sentences}
        metadata['example'] = example
        gold_clusters = self.convert_span_indices2spans(source, target, gold_clusters)
        if gold_clusters is not None:
            metadata["clusters"] = gold_clusters

        text_field = TextField([Token(word) for word in flattened_sentences], self._token_indexers)

        cluster_dict = {}
        if gold_clusters is not None:
            for cluster_id, cluster in enumerate(gold_clusters):
                for mention in cluster:
                    cluster_dict[tuple(mention)] = cluster_id

        spans: List[Field] = []
        span_labels: Optional[List[int]] = [] if gold_clusters is not None else None

        sentence_offset = 0
        for sentence in sentences:
            for start, end in enumerate_spans(sentence,
                                              offset=sentence_offset,
                                              max_span_width=self._max_span_width):
                if span_labels is not None:
                    if (start, end) in cluster_dict:
                        span_labels.append(cluster_dict[(start, end)])
                    else:
                        span_labels.append(-1)

                spans.append(SpanField(start, end, text_field))
            sentence_offset += len(sentence)

        span_field = ListField(spans)
        metadata_field = MetadataField(metadata)

        fields: Dict[str, Field] = {"text": text_field,
                                    "spans": span_field,
                                    "metadata": metadata_field}
        if span_labels is not None:
            fields["span_labels"] = SequenceLabelField(span_labels, span_field)

        return Instance(fields)

    @staticmethod
    def _normalize_word(word):
        if word in ("/.", "/?"):
            return word[1:]
        else:
            return word

    def check_continues_span(self, span_indices):
        if not span_indices:
            # [] return [0] which points to first token
            return [[0, 0]]
        if len(span_indices) == 1:
            return [[span_indices[0], span_indices[0]]]
        start = span_indices[0]
        end = start
        span = []
        for index in span_indices[1:]:
            if end + 1 == index:
                end = index
            else:
                span.append([start, end])
                start = index
                end = index
        span.append([start, end])
        return span



    def convert_span_indices2spans(self, source, target, gold_clusters):
        gold_spans = []
        for data in gold_clusters:
            source_span = self.check_continues_span(data[0])
            target_span = self.check_continues_span(data[1])
            source_span =[[span[0]  - 1, span[1]  - 1] for span in source_span]
            target_span = [[span[0] + len(source) - 1, span[1] + len(source) - 1] for span in target_span]
            gold_spans.append(source_span + target_span)
            st = source + target
            span_words = [st[span[0]:span[1] + 1] for span in source_span + target_span]
            source_words = data[2]
            target_words = data[3]
        return gold_spans



