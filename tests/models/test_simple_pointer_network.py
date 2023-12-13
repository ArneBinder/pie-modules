from dataclasses import dataclass

import pytest
import torch
from pytorch_ie import AnnotationList, annotation_field
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import TextBasedDocument
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForCausalLM,
    BartForConditionalGeneration,
    BartTokenizer,
    BeamSearchScorer,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
)

from pie_modules.models.simple_pointer_network import BartAsPointerNetwork
from pie_modules.taskmodules import PointerNetworkTaskModule

ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)
MODEL_NAME_OR_PATH = "sshleifer/distilbart-xsum-12-1"


@pytest.fixture(scope="module")
def document():
    @dataclass
    class ExampleDocument(TextBasedDocument):
        entities: AnnotationList[LabeledSpan] = annotation_field(target="text")
        relations: AnnotationList[BinaryRelation] = annotation_field(target="entities")
        sentences: AnnotationList[LabeledSpan] = annotation_field(target="text")

    doc = ExampleDocument(text="This is a dummy text about nothing. Trust me.")
    span1 = LabeledSpan(start=10, end=20, label="content")
    span2 = LabeledSpan(start=27, end=34, label="topic")
    span3 = LabeledSpan(start=42, end=44, label="person")
    doc.entities.extend([span1, span2, span3])
    assert str(span1) == "dummy text"
    assert str(span2) == "nothing"
    assert str(span3) == "me"
    rel = BinaryRelation(head=span1, tail=span2, label="is_about")
    doc.relations.append(rel)
    assert str(rel.label) == "is_about"
    assert str(rel.head) == "dummy text"
    assert str(rel.tail) == "nothing"

    no_rel = BinaryRelation(head=span1, tail=span3, label="no_relation")
    doc.relations.append(no_rel)
    assert str(no_rel.label) == "no_relation"
    assert str(no_rel.head) == "dummy text"
    assert str(no_rel.tail) == "me"

    sent1 = LabeledSpan(start=0, end=35, label="1")
    sent2 = LabeledSpan(start=36, end=45, label="2")
    doc.sentences.extend([sent1, sent2])
    assert str(sent1) == "This is a dummy text about nothing."
    assert str(sent2) == "Trust me."
    return doc


@pytest.fixture(scope="module")
def taskmodule(document):
    taskmodule = PointerNetworkTaskModule(
        annotation_encoder_decoder_kwargs={
            "span_layer_name": "entities",
            "relation_layer_name": "relations",
            "exclude_labels_per_layer": {"relations": ["no_relation"]},
        },
        annotation_field_mapping={
            "entities": "labeled_spans",
            "relations": "binary_relations",
        },
    )

    taskmodule.prepare(documents=[document])

    return taskmodule


def test_bart_generate():
    # model_name_or_path = "facebook/bart-large-cnn"
    model_name_or_path = MODEL_NAME_OR_PATH  # "sshleifer/distilbart-xsum-12-1"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = BartForConditionalGeneration.from_pretrained(model_name_or_path)
    # or BartForConditionalGeneration?

    input_text = ARTICLE_TO_SUMMARIZE
    inputs = tokenizer([input_text], max_length=1024, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], num_beams=3, min_length=5, max_length=20)
    result = tokenizer.batch_decode(
        summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    assert result == [" power lines in California have been shut down on Friday."]


def test_bart_pointer_network_generate(taskmodule):
    model_name_or_path = MODEL_NAME_OR_PATH
    tokenizer = taskmodule.tokenizer
    torch.random.manual_seed(42)
    model = BartAsPointerNetwork.from_pretrained(
        model_name_or_path,
        # label id space
        bos_token_id=taskmodule.annotation_encoder_decoder.bos_id,
        eos_token_id=taskmodule.annotation_encoder_decoder.eos_id,
        pad_token_id=taskmodule.annotation_encoder_decoder.eos_id,
        label_ids=taskmodule.annotation_encoder_decoder.label_ids,
        # target token id space
        target_token_ids=taskmodule.target_token_ids,
        target_pad_id=taskmodule.tokenizer.pad_token_id,
    )
    model.resize_token_embeddings(len(taskmodule.tokenizer))

    encoder_input_str = ARTICLE_TO_SUMMARIZE  # "translate English to German: How old are you?"
    inputs = tokenizer(encoder_input_str, max_length=1024, return_tensors="pt")

    outputs = model.generate(inputs["input_ids"], num_beams=3, min_length=5, max_length=20)
    torch.testing.assert_allclose(
        outputs,
        torch.tensor([[0, 8, 9, 10, 30, 19, 49, 21, 14, 55, 35, 14, 36, 17, 14, 36, 27, 1]]),
    )

    # result = tokenizer.batch_decode(
    #    summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # assert result == [" power lines in California have been shut down on Friday."]


def test_bart_pointer_network_beam_search(taskmodule):
    model_name_or_path = MODEL_NAME_OR_PATH
    tokenizer = taskmodule.tokenizer
    torch.random.manual_seed(42)
    model = BartAsPointerNetwork.from_pretrained(
        model_name_or_path,
        # label id space
        bos_token_id=taskmodule.annotation_encoder_decoder.bos_id,
        eos_token_id=taskmodule.annotation_encoder_decoder.eos_id,
        pad_token_id=taskmodule.annotation_encoder_decoder.eos_id,
        label_ids=taskmodule.annotation_encoder_decoder.label_ids,
        # target token id space
        target_token_ids=taskmodule.target_token_ids,
        target_pad_id=taskmodule.tokenizer.pad_token_id,
    )
    model.resize_token_embeddings(len(taskmodule.tokenizer))

    encoder_input_str = ARTICLE_TO_SUMMARIZE  # "translate English to German: How old are you?"
    encoder_input_tokenized = tokenizer(encoder_input_str, return_tensors="pt")

    # lets run beam search using 3 beams
    num_beams = 3
    # define decoder start token ids
    decoder_input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
    decoder_input_ids = decoder_input_ids * model.config.decoder_start_token_id

    # add encoder_outputs to model keyword arguments
    encoder = model.get_encoder()
    encoder_input_ids = encoder_input_tokenized.input_ids.repeat_interleave(num_beams, dim=0)
    encoder_attention_mask = encoder_input_tokenized.attention_mask.repeat_interleave(
        num_beams, dim=0
    )
    encoder_outputs = encoder(encoder_input_ids, return_dict=True)
    model_kwargs = {
        "encoder_outputs": encoder_outputs,
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": encoder_attention_mask,
    }

    # instantiate beam scorer
    beam_scorer = BeamSearchScorer(
        batch_size=1,
        num_beams=num_beams,
        device=model.device,
    )

    # instantiate logits processors
    logits_processor = LogitsProcessorList(
        [
            MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ]
    )

    outputs = model.beam_search(
        decoder_input_ids,
        beam_scorer,
        logits_processor=logits_processor,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        **model_kwargs
    )

    torch.testing.assert_allclose(
        outputs,
        torch.tensor(
            [
                [
                    0,
                    8,
                    9,
                    10,
                    30,
                    19,
                    49,
                    21,
                    14,
                    55,
                    35,
                    14,
                    36,
                    17,
                    14,
                    55,
                    35,
                    14,
                    36,
                    17,
                    14,
                    36,
                    27,
                    1,
                ]
            ]
        ),
    )

    # result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # assert result == [
    #    " power lines in California have been shut down after a power provider said it was due to high winds."
    # ]
