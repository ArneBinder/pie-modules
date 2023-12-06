import logging
from dataclasses import dataclass

import pytest
import torch
from pytorch_ie import AnnotationList, annotation_field
from pytorch_ie.annotations import BinaryRelation, LabeledSpan
from pytorch_ie.documents import TextBasedDocument

from pie_modules.models import PointerNetworkModel
from pie_modules.taskmodules import PointerNetworkForJointTaskModule
# from src.models.components.gmam.metrics import LabeledAnnotationScore, AnnotationLayerMetric
from tests import FIXTURES_ROOT

# from tests.taskmodules.test_gmam_taskmodule import FIXTURES_DIR as TASKMODULE_FIXTURE_DIR

logger = logging.getLogger(__name__)

# FIXTURES_DIR = FIXTURES_ROOT / "models" / "gmam_model"

DUMP_FIXTURE_DATA = False


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
    taskmodule = PointerNetworkForJointTaskModule(
        text_field_name="text",
        span_layer_name="entities",
        relation_layer_name="relations",
        exclude_annotation_names={"relations": ["no_relation"]},
    )

    taskmodule.prepare(documents=[document])

    assert taskmodule.bos_id == 0
    assert taskmodule.eos_id == 1
    assert taskmodule.relation_ids == [4]
    assert taskmodule.none_ids == 3
    assert taskmodule.span_ids == [6, 5, 2]
    assert taskmodule.label_ids == [2, 3, 4, 5, 6]
    assert taskmodule.target_token_ids == [0, 2, 50265, 50266, 50267, 50268, 50269]
    assert taskmodule.target_tokens == [
        "<s>",
        "</s>",
        "<<topic>>",
        "<<none>>",
        "<<is_about>>",
        "<<person>>",
        "<<content>>",
    ]
    assert taskmodule.tokenizer.pad_token_id == 1
    assert taskmodule.target_token2id == {
        "<s>": 0,
        "</s>": 1,
        "<<topic>>": 2,
        "<<none>>": 3,
        "<<is_about>>": 4,
        "<<person>>": 5,
        "<<content>>": 6,
    }
    assert len(taskmodule.tokenizer) == 50270
    assert taskmodule.embedding_weight_mapping == {
        50267: [354, 1215, 9006],
        50269: [10166],
        50268: [5970],
        50265: [45260],
        50266: [39763],
    }

    return taskmodule


def test_taskmodule(taskmodule):
    assert taskmodule is not None


@pytest.fixture(scope="module")
def model(taskmodule) -> PointerNetworkModel:
    torch.manual_seed(42)
    model = PointerNetworkModel(
        relation_ids=taskmodule.relation_ids,
        span_ids=taskmodule.span_ids,
        none_ids=taskmodule.none_ids,
        label_ids=taskmodule.label_ids,
        bart_model="facebook/bart-base",
        bos_id=taskmodule.bos_id,
        eos_id=taskmodule.eos_id,
        pad_id=taskmodule.pad_id,
        target_token_ids=taskmodule.target_token_ids,
        target_tokens=taskmodule.target_tokens,
        pad_token_id=taskmodule.tokenizer.pad_token_id,
        vocab_size=len(taskmodule.tokenizer),
        embedding_weight_mapping=taskmodule.embedding_weight_mapping,
        decoder_type="avg_score",
        copy_gate=False,
        use_encoder_mlp=True,
        use_recur_pos=False,
        replace_pos=True,
        position_type=0,
        max_length=10,
        max_len_a=0.5,
        num_beams=4,
        do_sample=False,
        repetition_penalty=1,
        length_penalty=1.0,
        restricter=None,
        decode_mask=True,
        biloss=True,
        lr=5e-5,
        max_target_positions=512,
    )
    # set model to training mode, otherwise model.encoder.bart_encoder.training will be False!
    model.train()
    return model


def test_model(model):
    assert model is not None


# not used
@pytest.fixture(scope="module")
def batch(taskmodule, document):
    task_encodings = taskmodule.encode(documents=[document], encode_target=True)
    batch = taskmodule.collate(task_encodings)
    return batch


def test_batch(batch):
    assert batch is not None
    inputs, targets = batch
    assert inputs is not None
    assert set(inputs) == {"src_tokens", "src_seq_len"}
    torch.testing.assert_close(
        inputs["src_tokens"],
        torch.tensor([[0, 713, 16, 10, 34759, 2788, 59, 1085, 4, 3101, 162, 4, 2]]),
    )
    torch.testing.assert_close(
        inputs["src_seq_len"],
        torch.tensor([13]),
    )

    assert targets is not None
    assert set(targets) == {"tgt_tokens", "tgt_seq_len", "CPM_tag"}
    torch.testing.assert_close(
        targets["tgt_tokens"],
        torch.tensor([[0, 14, 14, 2, 11, 12, 6, 4, 17, 17, 5, 3, 3, 3, 3, 1]]),
    )
    torch.testing.assert_close(
        targets["tgt_seq_len"],
        torch.tensor([16]),
    )
    torch.testing.assert_close(
        targets["CPM_tag"],
        torch.tensor(
            [
                [
                    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ]
            ]
        ),
    )


def test_training_step(model, batch):
    torch.manual_seed(42)
    loss = model.training_step(batch, 0)
    assert loss is not None
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert loss.dim() == 0
    torch.testing.assert_close(loss, torch.tensor(17.09836769104004))


def test_predict_step(model, batch):
    torch.manual_seed(42)
    prediction = model.predict_step(batch, 0)
    assert prediction is not None


def test_metric_val(model, batch):
    metric = model.metrics["val"]
    metric.reset()
    inputs, targets = batch
    prediction = model.predict(inputs)
    metric(prediction["pred"], targets["tgt_tokens"])

    values = metric.get_metric(reset=True)
    assert values is not None
    assert values == {
        "em": 0.0,
        "span": {
            "<<content>>": {"acc": 0, "recall": 0.0, "f1": 0.0},
            "<<person>>": {"acc": 0, "recall": 0.0, "f1": 0.0},
            "<<topic>>": {"acc": 0, "recall": 0.0, "f1": 0.0},
        },
        "span/micro": {"acc": 0, "recall": 0.0, "f1": 0.0},
        "relation": {"<<is_about>>": {"acc": 0, "recall": 0.0, "f1": 0.0}},
        "relation/micro": {"acc": 0, "recall": 0.0, "f1": 0.0},
        "invalid/len": 0.0,
        "invalid/order": 0.0,
        "invalid/cross": 0.0,
        "invalid/cover": 0.0,
        "invalid/all": 0.0,
    }
