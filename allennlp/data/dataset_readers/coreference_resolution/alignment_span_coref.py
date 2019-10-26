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


    @overrides
    def text_to_instance(self,  # type: ignore
                         source, target, gold_clusters, example) -> Instance:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        source: list of chinese character:  <class 'list'>: ['[Speaker#1]', '二', '零', '零', '五', '年', '的', '夏', '天', '，', '一', '个', '*OP*', '*T*-1', '被', '人', '们', '期', '待', '*T*-2', '已', '久', '的', '画', '面', '开', '始', '在', '香', '港', '的', '各', '大', '媒', '体', '频', '繁', '出', '现', '，']
        target: list of english words: <class 'list'>: ['<47.836:55.349620893:Speaker#1:1>', 'In', 'the', 'summer', 'of', '2005', ',', 'a', 'picture', 'that', 'people', 'have', 'long', 'been', 'looking', 'forward', 'to', '*T*-1', 'started', '*-2', 'emerging', 'with', 'frequency', 'in', 'various', 'major', 'Hong', 'Kong', 'media', '.']
        gold_clusters: list gold span pairs: <class 'list'>: [[[8, 9], [2, 3, 4], ['夏', '天'], ['In', 'the', 'summer']], [[21, 22], [13], ['已', '久'], ['long']], [[33], [26], ['大'], ['major']], [[40], [30], ['，'], ['.']], [[32], [25], ['各'], ['various']], [[34, 35], [29], ['媒', '体'], ['media']], [[28], [24], ['在'], ['in']], [[11, 12], [8], ['一', '个'], ['a']], [[23], [10], ['的'], ['that']], [[2, 3, 4, 5, 6], [6], ['二', '零', '零', '五', '年'], ['2005']], [[15, 18, 19], [12, 14, 15, 16], ['被', '期', '待'], ['have', 'been', 'looking', 'forward']], [[10], [7], ['，'], [',']], [[7], [5], ['的'], ['of']], [[16, 17], [11], ['人', '们'], ['people']], [[26, 27], [17, 19], ['开', '始'], ['to', 'started']], [[36, 37], [22, 23], ['频', '繁'], ['with', 'frequency']], [[24, 25], [9], ['画', '面'], ['picture']], [[38, 39], [21], ['出', '现'], ['emerging']], [[29, 30, 31], [27, 28], ['香', '港', '的'], ['Hong', 'Kong']], [[], [1], [], ['<47.836:55.349620893:Speaker#1:1>']], [[], [18], [], ['*T*-1']], [[], [20], [], ['*-2']], [[1], [], ['[Speaker#1]'], []], [[13], [], ['*OP*'], []], [[14], [], ['*T*-1'], []], [[20], [], ['*T*-2'], []]]
                       [[[chinese character indices], [english word indices], [corresponding chinese character], [corresponding english words]], ....]
        example: a dict, {'chinese': source, 'english': target, 'spans': gold_clusters, 'source': original alignment annotation of chinese sentence, 'target': original aligenment annotation of english sentence}

        Returns
        instance:     {"text": text_field (this is a concat of chinese and english: TextField of length 70 with text:
 		                        [cls, 二, 零, 零, 五, 年, 的, 夏, 天, ，, 一, 个, *OP*, *T*-1, 被, 人, 们, 期, 待, *T*-2, 已, 久, 的, 画, 面, 开, 始, 在, 香,
		                        港, 的, 各, 大, 媒, 体, 频, 繁, 出, 现, ，, sep, In, the, summer, of, 2005, ,, a, picture, that, people, have,
		                        long, been, looking, forward, to, *T*-1, started, *-2, emerging, with, frequency, in, various,
		                        major, Hong, Kong, media, .]),
                                    "spans": span_field (this will indicate each span's label and the label is the indice of gold_clusters. ,
                                    "metadata": metadata_field,
                                    "span_labels": SequenceLabelField(span_labels, span_field)
                    }
        -------
        """
        source = ['cls'] + source[1:]
        target = ['sep'] + target[1:]
        sentences = [source, target]
        flattened_sentences = source + target
        metadata: Dict[str, Any] = {"original_text": flattened_sentences}
        metadata['example'] = example
        # here covert spans label of [[[8,9],[2,3,4]], ...] to [[[8,9], [2,4]],....] this may be what conll2012srl outputs
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



