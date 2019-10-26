# pylint: disable=no-self-use,invalid-name
from typing import List, Tuple

import pytest

from allennlp.data.dataset_readers import AlignmentSpan
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

class TestCorefReader:
    span_width = 5

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        span_reader = AlignmentSpan(max_span_width=self.span_width, lazy=lazy)
        instances = ensure_list(span_reader.read(str(AllenNlpTestCase.FIXTURES_ROOT /
                                                      'coref' / 'alignment_span')))

        assert len(instances) == 2
