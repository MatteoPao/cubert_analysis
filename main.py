import sys
import numpy as np
import itertools

import tensorflow.compat.v1 as tf
import glob
import os
import json

from bert import tokenization
from tensor2tensor.data_generators import text_encoder
from keras_bert import load_vocabulary, load_trained_model_from_checkpoint, Tokenizer, get_checkpoint_paths
from cubert import code_to_subtokenized_sentences
from cubert import tokenizer_registry
from cubert import unified_tokenizer
from typing import Any, Dict, List


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


def convert_single_example(example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    assert example.text_b is None

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]_")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]_")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

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

    print("*** Example ***")
    print("guid: ", example.guid)
    print("tokens: ", [tokenization.printable_text(x) for x in tokens])
    print("input_ids: ", [str(x) for x in input_ids])
    print("input_mask: ", [str(x) for x in input_mask])
    print("segment_ids: ", [str(x) for x in segment_ids])
    print("label: ", example.label + " id: ", label_id)

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True,
        guid=example.guid)
    return feature


def _read_jsonl(input_file):
    """Reads a tab-separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
        for line in f:
            yield json.loads(line)


def _create_examples(raw_examples, set_type):
    """Creates a single example from the loaded JSON dictionary."""
    examples = []
    for (i, example) in enumerate(raw_examples):
        guid = "%s-%s" % (set_type, i)
        # No convert_to_unicode here, since the CuBERT datasets are already
        # generated as valid Unicode when released, and parsed in as Unicode
        # directly during data reading.
        text_a = example["function"]
        label = example["label"]
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def _read_examples_from_jsonls(data_dir, set_type):
    examples: List[Dict[str, Any]] = []
    for file in glob.glob(os.path.join(data_dir, f"{set_type}.jsontxt-*")):
        examples.extend(_read_jsonl(file))
    return _create_examples(examples, set_type)


HOLE_NAME = "__HOLE__"
UNKNOWN_TOKEN = "unknown_token_default"
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_enum_class(
    "code_tokenizer",
    default=tokenizer_registry.TokenizerEnum.PYTHON,
    enum_class=tokenizer_registry.TokenizerEnum,
    help="The tokenizer to use.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")


print('This demo demonstrates how to load the pre-trained model and check whether the two sentences are continuous')

if len(sys.argv) == 3:
    model_path = sys.argv[1]
    data_path = sys.argv[2]
else:
    from keras_bert.datasets import get_pretrained, PretrainedList

    model_path = get_pretrained(PretrainedList.chinese_base)

paths = get_checkpoint_paths(model_path)

#vars_in_checkpoint = tf.train.list_variables(paths.checkpoint)
#print(vars_in_checkpoint)

model = load_trained_model_from_checkpoint(paths.config, paths.checkpoint, training=True, seq_len=128)
model.summary()

examples = _read_examples_from_jsonls(data_path, "eval")

tokenizer = FullCuBertTokenizer(
    code_tokenizer_class=FLAGS.code_tokenizer.value,
    vocab_file=paths.vocab)

processors = {
    "docstring": CuBertFunctionDocstringProcessor,
    "exception": CuBertExceptionClassificationProcessor,
    "varmisuse": CuBertVariableMisuseProcessor,
    "swappedop": CuBertSwappedOperandProcessor,
    "wrongop": CuBertWrongOperatorProcessor,
}
processor = processors["exception"]()
label_list = processor.get_labels()

s_f = convert_single_example(examples[1], label_list, FLAGS.max_seq_length, tokenizer)

predicts = model.predict([np.expand_dims(np.array(s_f.input_ids), axis=0),
                          np.expand_dims(np.array(s_f.segment_ids), axis=0),
                          np.expand_dims(np.array(s_f.input_mask), axis=0)])
print(predicts)
