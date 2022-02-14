import numpy as np
import itertools

from cubert import code_to_subtokenized_sentences
from cubert import unified_tokenizer
from bert import tokenization
from tensor2tensor.data_generators import text_encoder


HOLE_NAME = "__HOLE__"
UNKNOWN_TOKEN = "unknown_token_default"

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 guid,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example
        self.guid = guid

    def __eq__(self, other):
        return (self.input_ids == other.input_ids and
                self.input_mask == other.input_mask and
                self.segment_ids == other.segment_ids and
                self.label_id == other.label_id and
                self.is_real_example == other.is_real_example)

    def __repr__(self):
        return ("Input IDs: {}\n"
                "Input Mask: {}\n"
                "Segment IDs: {}\n"
                "Label ID: {}\n"
                "Real: {}\n"
                "GUID: {}".format(
                    self.input_ids,
                    self.input_mask,
                    self.segment_ids,
                    self.label_id,
                    self.is_real_example,
                    self.guid,
                ))


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


# token_dict = load_vocabulary(paths.vocab)
# token_dict_inv = {v: k for k, v in token_dict.items()}
class FullCuBertTokenizer:
    """Wraps the CuBERT tokenizers to behave like BERT's tokenization API."""

    def __init__(self, code_tokenizer_class, vocab_file):

        # "Tokenizer" going from code to subtokenized sentences:
        self.code_tokenizer = code_tokenizer_class()
        # CuBERT skips Comments/Whitespace in finetuned tasks.
        self.code_tokenizer.replace_reserved_keywords((
            HOLE_NAME,
            # Although we don't produce the unknown token when generating VarMisuse
            # examples, we put the unknown token into the common initialization for
            # the tokenizer so that, when the model asks for the tokenization of
            # that special token, it gets a consistent result.
            UNKNOWN_TOKEN))
        self.code_tokenizer.update_types_to_skip((
            unified_tokenizer.TokenKind.COMMENT,
            unified_tokenizer.TokenKind.WHITESPACE,
        ))

        self.subwork_tokenizer = text_encoder.SubwordTextEncoder(vocab_file)

    def tokenize(self, text):
        subtokenized_sentences = (
            code_to_subtokenized_sentences.code_to_cubert_sentences(
                code=text,
                initial_tokenizer=self.code_tokenizer,
                subword_tokenizer=self.subwork_tokenizer))
        return list(itertools.chain(*subtokenized_sentences))

    def convert_tokens_to_ids(self, tokens):
        return tokenization.convert_by_vocab(
            self.subwork_tokenizer._subtoken_string_to_id,  # pylint: disable = protected-access
            tokens)


    def convert_single_example(self, example, label_list, max_seq_length):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i

        tokens_a = self.tokenize(example.text_a)
        assert example.text_b is None

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]_")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]_")
        segment_ids.append(0)

        input_ids = self.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        feature = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            label_id=label_id,
            is_real_example=True,
            guid=example.guid)
        return feature

    def convert_examples_to_features(self, examples, label_list, max_seq_length):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""

        features = []
        for (ex_index, example) in enumerate(examples):
            feature = self.convert_single_example(example, label_list, max_seq_length)
            features.append(feature)
        return features


class CuBertFunctionDocstringProcessor:

  def get_labels(self):
    """See base class."""
    return ["Correct", "Incorrect"]


class CuBertExceptionClassificationProcessor:

  def get_labels(self):
    """See base class."""
    return [
        "ValueError",
        "KeyError",
        "AttributeError",
        "TypeError",
        "OSError",
        "IOError",
        "ImportError",
        "IndexError",
        "DoesNotExist",
        "KeyboardInterrupt",
        "StopIteration",
        "AssertionError",
        "SystemExit",
        "RuntimeError",
        "HTTPError",
        "UnicodeDecodeError",
        "NotImplementedError",
        "ValidationError",
        "ObjectDoesNotExist",
        "NameError",
        "None",
    ]


class CuBertVariableMisuseProcessor:

  def get_labels(self):
    """See base class."""
    return ["Correct", "Variable misuse"]


class CuBertSwappedOperandProcessor:

  def get_labels(self):
    """See base class."""
    return ["Correct", "Swapped operands"]


class CuBertWrongOperatorProcessor:

  def get_labels(self):
    """See base class."""
    return ["Correct", "Wrong binary operator"]
