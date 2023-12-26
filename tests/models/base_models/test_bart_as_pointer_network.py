from dataclasses import dataclass

import pytest
import torch
from pytorch_ie import AnnotationList, annotation_field
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import TextBasedDocument
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    BeamSearchScorer,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
)
from transformers.generation import BeamSearchEncoderDecoderOutput

from pie_modules.models.base_models import BartAsPointerNetwork
from pie_modules.taskmodules import PointerNetworkTaskModuleForEnd2EndRE

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
    taskmodule = PointerNetworkTaskModuleForEnd2EndRE(
        span_layer_name="entities",
        relation_layer_name="relations",
        exclude_labels_per_layer={"relations": ["no_relation"]},
        annotation_field_mapping={
            "entities": "labeled_spans",
            "relations": "binary_relations",
        },
        create_constraints=False,
        # tokenizer_kwargs={"strict_span_conversion": False},
    )

    taskmodule.prepare(documents=[document])

    return taskmodule


@pytest.mark.skip("This is just a test to see how Bart works")
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


@pytest.fixture(scope="module")
def model(taskmodule) -> BartAsPointerNetwork:
    model_name_or_path = MODEL_NAME_OR_PATH

    torch.random.manual_seed(42)
    model = BartAsPointerNetwork.from_pretrained(
        model_name_or_path,
        # label id space
        bos_token_id=taskmodule.bos_id,
        eos_token_id=taskmodule.eos_id,
        pad_token_id=taskmodule.eos_id,
        label_ids=taskmodule.label_ids,
        # target token id space
        target_token_ids=taskmodule.target_token_ids,
        # mapping to better initialize the label embedding weights
        embedding_weight_mapping=taskmodule.label_embedding_weight_mapping,
    )

    return model


def test_bart_pointer_network_beam_search(model, taskmodule):
    encoder_input_str = ARTICLE_TO_SUMMARIZE  # "translate English to German: How old are you?"
    encoder_input_tokenized = taskmodule.tokenizer(encoder_input_str, return_tensors="pt")

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

    torch.testing.assert_close(
        outputs,
        torch.tensor(
            [[0, 28, 41, 35, 33, 36, 17, 33, 36, 17, 33, 36, 17, 33, 36, 17, 33, 36, 37, 1]]
        ),
    )

    # result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # assert result == [
    #    " power lines in California have been shut down after a power provider said it was due to high winds."
    # ]


def test_bart_pointer_network_generate_with_scores(model, taskmodule):
    encoder_input_str = ARTICLE_TO_SUMMARIZE  # "translate English to German: How old are you?"
    inputs = taskmodule.tokenizer(encoder_input_str, max_length=1024, return_tensors="pt")

    outputs = model.generate(
        inputs["input_ids"],
        num_beams=3,
        min_length=5,
        max_length=20,
        return_dict_in_generate=True,
        output_scores=True,
    )
    assert isinstance(outputs, BeamSearchEncoderDecoderOutput)
    torch.testing.assert_close(outputs.sequences_scores, torch.tensor([-6.784079074859619]))
    torch.testing.assert_close(
        outputs.sequences,
        torch.tensor(
            [[0, 28, 41, 35, 33, 36, 17, 48, 36, 17, 33, 55, 35, 33, 17, 48, 55, 35, 48, 36]]
        ),
    )

    # result = tokenizer.batch_decode(
    #    summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    # assert result == [" power lines in California have been shut down on Friday."]


def test_forward_with_labels(model, taskmodule, document):
    task_encodings = taskmodule.encode([document], encode_target=True)
    batch = taskmodule.collate(task_encodings)
    inputs, targets = batch
    input_ids = inputs["src_tokens"]
    attention_mask = inputs["src_attention_mask"]
    # Truncate the bos_id. The decoder input_ids will be created by the model
    # by shifting the labels one position to the right and adding the bos_id
    labels = targets["tgt_tokens"][:, 1:]
    decoder_attention_mask = targets["tgt_attention_mask"][:, 1:]

    torch.manual_seed(42)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        decoder_attention_mask=decoder_attention_mask,
    )
    loss = outputs.loss
    torch.testing.assert_close(loss, torch.tensor(4.802009105682373))


def test_head_named_params(model):
    parameter_shapes = {name: tuple(param.shape) for name, param in model.head_named_params()}
    assert parameter_shapes == {
        "pointer_head.encoder_mlp.0.bias": (1024,),
        "pointer_head.encoder_mlp.0.weight": (1024, 1024),
        "pointer_head.encoder_mlp.3.bias": (1024,),
        "pointer_head.encoder_mlp.3.weight": (1024, 1024),
    }


def test_encoder_only_named_params(model):
    parameter_shapes = {
        name: tuple(param.shape) for name, param in model.encoder_only_named_params()
    }
    assert len(parameter_shapes) == 195
    assert parameter_shapes == {
        "model.encoder.embed_positions.weight": (1026, 1024),
        "model.encoder.layers.0.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.0.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.0.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.0.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.0.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.0.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.0.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.0.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.0.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.0.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.0.fc1.weight": (4096, 1024),
        "model.encoder.layers.0.fc1.bias": (4096,),
        "model.encoder.layers.0.fc2.weight": (1024, 4096),
        "model.encoder.layers.0.fc2.bias": (1024,),
        "model.encoder.layers.0.final_layer_norm.weight": (1024,),
        "model.encoder.layers.0.final_layer_norm.bias": (1024,),
        "model.encoder.layers.1.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.1.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.1.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.1.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.1.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.1.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.1.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.1.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.1.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.1.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.1.fc1.weight": (4096, 1024),
        "model.encoder.layers.1.fc1.bias": (4096,),
        "model.encoder.layers.1.fc2.weight": (1024, 4096),
        "model.encoder.layers.1.fc2.bias": (1024,),
        "model.encoder.layers.1.final_layer_norm.weight": (1024,),
        "model.encoder.layers.1.final_layer_norm.bias": (1024,),
        "model.encoder.layers.2.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.2.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.2.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.2.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.2.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.2.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.2.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.2.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.2.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.2.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.2.fc1.weight": (4096, 1024),
        "model.encoder.layers.2.fc1.bias": (4096,),
        "model.encoder.layers.2.fc2.weight": (1024, 4096),
        "model.encoder.layers.2.fc2.bias": (1024,),
        "model.encoder.layers.2.final_layer_norm.weight": (1024,),
        "model.encoder.layers.2.final_layer_norm.bias": (1024,),
        "model.encoder.layers.3.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.3.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.3.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.3.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.3.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.3.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.3.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.3.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.3.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.3.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.3.fc1.weight": (4096, 1024),
        "model.encoder.layers.3.fc1.bias": (4096,),
        "model.encoder.layers.3.fc2.weight": (1024, 4096),
        "model.encoder.layers.3.fc2.bias": (1024,),
        "model.encoder.layers.3.final_layer_norm.weight": (1024,),
        "model.encoder.layers.3.final_layer_norm.bias": (1024,),
        "model.encoder.layers.4.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.4.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.4.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.4.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.4.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.4.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.4.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.4.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.4.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.4.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.4.fc1.weight": (4096, 1024),
        "model.encoder.layers.4.fc1.bias": (4096,),
        "model.encoder.layers.4.fc2.weight": (1024, 4096),
        "model.encoder.layers.4.fc2.bias": (1024,),
        "model.encoder.layers.4.final_layer_norm.weight": (1024,),
        "model.encoder.layers.4.final_layer_norm.bias": (1024,),
        "model.encoder.layers.5.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.5.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.5.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.5.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.5.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.5.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.5.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.5.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.5.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.5.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.5.fc1.weight": (4096, 1024),
        "model.encoder.layers.5.fc1.bias": (4096,),
        "model.encoder.layers.5.fc2.weight": (1024, 4096),
        "model.encoder.layers.5.fc2.bias": (1024,),
        "model.encoder.layers.5.final_layer_norm.weight": (1024,),
        "model.encoder.layers.5.final_layer_norm.bias": (1024,),
        "model.encoder.layers.6.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.6.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.6.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.6.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.6.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.6.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.6.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.6.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.6.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.6.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.6.fc1.weight": (4096, 1024),
        "model.encoder.layers.6.fc1.bias": (4096,),
        "model.encoder.layers.6.fc2.weight": (1024, 4096),
        "model.encoder.layers.6.fc2.bias": (1024,),
        "model.encoder.layers.6.final_layer_norm.weight": (1024,),
        "model.encoder.layers.6.final_layer_norm.bias": (1024,),
        "model.encoder.layers.7.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.7.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.7.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.7.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.7.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.7.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.7.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.7.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.7.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.7.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.7.fc1.weight": (4096, 1024),
        "model.encoder.layers.7.fc1.bias": (4096,),
        "model.encoder.layers.7.fc2.weight": (1024, 4096),
        "model.encoder.layers.7.fc2.bias": (1024,),
        "model.encoder.layers.7.final_layer_norm.weight": (1024,),
        "model.encoder.layers.7.final_layer_norm.bias": (1024,),
        "model.encoder.layers.8.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.8.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.8.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.8.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.8.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.8.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.8.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.8.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.8.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.8.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.8.fc1.weight": (4096, 1024),
        "model.encoder.layers.8.fc1.bias": (4096,),
        "model.encoder.layers.8.fc2.weight": (1024, 4096),
        "model.encoder.layers.8.fc2.bias": (1024,),
        "model.encoder.layers.8.final_layer_norm.weight": (1024,),
        "model.encoder.layers.8.final_layer_norm.bias": (1024,),
        "model.encoder.layers.9.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.9.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.9.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.9.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.9.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.9.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.9.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.9.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.9.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.9.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.9.fc1.weight": (4096, 1024),
        "model.encoder.layers.9.fc1.bias": (4096,),
        "model.encoder.layers.9.fc2.weight": (1024, 4096),
        "model.encoder.layers.9.fc2.bias": (1024,),
        "model.encoder.layers.9.final_layer_norm.weight": (1024,),
        "model.encoder.layers.9.final_layer_norm.bias": (1024,),
        "model.encoder.layers.10.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.10.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.10.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.10.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.10.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.10.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.10.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.10.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.10.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.10.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.10.fc1.weight": (4096, 1024),
        "model.encoder.layers.10.fc1.bias": (4096,),
        "model.encoder.layers.10.fc2.weight": (1024, 4096),
        "model.encoder.layers.10.fc2.bias": (1024,),
        "model.encoder.layers.10.final_layer_norm.weight": (1024,),
        "model.encoder.layers.10.final_layer_norm.bias": (1024,),
        "model.encoder.layers.11.self_attn.k_proj.weight": (1024, 1024),
        "model.encoder.layers.11.self_attn.k_proj.bias": (1024,),
        "model.encoder.layers.11.self_attn.v_proj.weight": (1024, 1024),
        "model.encoder.layers.11.self_attn.v_proj.bias": (1024,),
        "model.encoder.layers.11.self_attn.q_proj.weight": (1024, 1024),
        "model.encoder.layers.11.self_attn.q_proj.bias": (1024,),
        "model.encoder.layers.11.self_attn.out_proj.weight": (1024, 1024),
        "model.encoder.layers.11.self_attn.out_proj.bias": (1024,),
        "model.encoder.layers.11.self_attn_layer_norm.weight": (1024,),
        "model.encoder.layers.11.self_attn_layer_norm.bias": (1024,),
        "model.encoder.layers.11.fc1.weight": (4096, 1024),
        "model.encoder.layers.11.fc1.bias": (4096,),
        "model.encoder.layers.11.fc2.weight": (1024, 4096),
        "model.encoder.layers.11.fc2.bias": (1024,),
        "model.encoder.layers.11.final_layer_norm.weight": (1024,),
        "model.encoder.layers.11.final_layer_norm.bias": (1024,),
        "model.encoder.layernorm_embedding.weight": (1024,),
        "model.encoder.layernorm_embedding.bias": (1024,),
    }


def test_decoder_only_named_params(model):
    parameter_shapes = {
        name: tuple(param.shape) for name, param in model.decoder_only_named_params()
    }
    assert len(parameter_shapes) == 29
    assert parameter_shapes == {
        "model.decoder.embed_positions.weight": (1026, 1024),
        "model.decoder.layers.0.self_attn.k_proj.weight": (1024, 1024),
        "model.decoder.layers.0.self_attn.k_proj.bias": (1024,),
        "model.decoder.layers.0.self_attn.v_proj.weight": (1024, 1024),
        "model.decoder.layers.0.self_attn.v_proj.bias": (1024,),
        "model.decoder.layers.0.self_attn.q_proj.weight": (1024, 1024),
        "model.decoder.layers.0.self_attn.q_proj.bias": (1024,),
        "model.decoder.layers.0.self_attn.out_proj.weight": (1024, 1024),
        "model.decoder.layers.0.self_attn.out_proj.bias": (1024,),
        "model.decoder.layers.0.self_attn_layer_norm.weight": (1024,),
        "model.decoder.layers.0.self_attn_layer_norm.bias": (1024,),
        "model.decoder.layers.0.encoder_attn.k_proj.weight": (1024, 1024),
        "model.decoder.layers.0.encoder_attn.k_proj.bias": (1024,),
        "model.decoder.layers.0.encoder_attn.v_proj.weight": (1024, 1024),
        "model.decoder.layers.0.encoder_attn.v_proj.bias": (1024,),
        "model.decoder.layers.0.encoder_attn.q_proj.weight": (1024, 1024),
        "model.decoder.layers.0.encoder_attn.q_proj.bias": (1024,),
        "model.decoder.layers.0.encoder_attn.out_proj.weight": (1024, 1024),
        "model.decoder.layers.0.encoder_attn.out_proj.bias": (1024,),
        "model.decoder.layers.0.encoder_attn_layer_norm.weight": (1024,),
        "model.decoder.layers.0.encoder_attn_layer_norm.bias": (1024,),
        "model.decoder.layers.0.fc1.weight": (4096, 1024),
        "model.decoder.layers.0.fc1.bias": (4096,),
        "model.decoder.layers.0.fc2.weight": (1024, 4096),
        "model.decoder.layers.0.fc2.bias": (1024,),
        "model.decoder.layers.0.final_layer_norm.weight": (1024,),
        "model.decoder.layers.0.final_layer_norm.bias": (1024,),
        "model.decoder.layernorm_embedding.weight": (1024,),
        "model.decoder.layernorm_embedding.bias": (1024,),
    }


def test_encoder_decoder_shared_named_params(model):
    parameter_shapes = {
        name: tuple(param.shape) for name, param in model.encoder_decoder_shared_named_params()
    }
    assert len(parameter_shapes) == 1
    assert parameter_shapes == {"model.shared.weight": (50270, 1024)}


def test_base_model_named_params(model):
    parameter_shapes = {
        name: tuple(param.shape) for name, param in model.base_model_named_params()
    }
    assert len(parameter_shapes) == 225
    encoder_only_parameter_shapes = {
        name: tuple(param.shape) for name, param in model.encoder_only_named_params()
    }
    decoder_only_parameter_shapes = {
        name: tuple(param.shape) for name, param in model.decoder_only_named_params()
    }
    shared_parameter_shapes = {
        name: tuple(param.shape) for name, param in model.encoder_decoder_shared_named_params()
    }
    expected_parameter_shapes = {
        **encoder_only_parameter_shapes,
        **decoder_only_parameter_shapes,
        **shared_parameter_shapes,
    }

    assert parameter_shapes == expected_parameter_shapes
