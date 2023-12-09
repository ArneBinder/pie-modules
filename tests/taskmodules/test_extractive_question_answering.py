from collections import defaultdict

import pytest
import torch
import transformers
from pytorch_ie.core import AnnotationList

from pie_modules.annotations import ExtractiveAnswer, Question
from pie_modules.documents import ExtractiveQADocument
from pie_modules.taskmodules.extractive_question_answering import (
    ExtractiveQuestionAnsweringTaskModule,
)


@pytest.fixture()
def document():
    document = ExtractiveQADocument(text="This is a test document", id="doc0")
    document.questions.append(Question(text="What is the first word?"))
    document.answers.append(ExtractiveAnswer(question=document.questions[0], start=0, end=3))
    return document


@pytest.fixture()
def document_with_no_answer():
    document = ExtractiveQADocument(text="This is a test document", id="doc0")
    document.questions.append(Question(text="What is the first word?"))
    return document


@pytest.fixture()
def document_with_multuple_answers():
    document = ExtractiveQADocument(text="This is a test document", id="doc0")
    document.questions.append(Question(text="What is the first word?"))
    document.answers.append(ExtractiveAnswer(question=document.questions[0], start=0, end=3))
    document.answers.append(ExtractiveAnswer(question=document.questions[0], start=0, end=3))
    return document


@pytest.fixture()
def document_batch():
    document0 = ExtractiveQADocument(text="This is a test document", id="doc0")
    document0.questions.append(Question(text="What is the first word?"))
    document0.answers.append(ExtractiveAnswer(question=document0.questions[0], start=0, end=3))

    document1 = ExtractiveQADocument(text="This is the second document", id="doc1")
    document1.questions.append(Question(text="Which document is this?"))
    document1.answers.append(ExtractiveAnswer(question=document1.questions[0], start=13, end=18))
    return [document0, document1]


@pytest.fixture()
def model_outputs():
    # create dummy model outputs
    start_logits = [[0.05] * 30, [0.05] * 30]
    end_logits = [[0.05] * 30, [0.05] * 30]
    # set some values to 0.95 as a dummy value
    start1 = 0
    end1 = 3
    start2 = 13
    end2 = 18
    start_logits[0][start1] = 0.95
    end_logits[0][end1] = 0.95
    start_logits[1][start2] = 0.95
    end_logits[1][end2] = 0.95

    # convert to torch tensors
    start_logits = torch.FloatTensor(start_logits)
    end_logits = torch.FloatTensor(end_logits)

    # convert to logits
    start_logits = torch.log(start_logits / (1 - start_logits))
    end_logits = torch.log(end_logits / (1 - end_logits))

    model_outputs = transformers.modeling_outputs.QuestionAnsweringModelOutput(
        start_logits=start_logits,
        end_logits=end_logits,
    )
    return model_outputs


@pytest.fixture()
def taskmodule():
    return ExtractiveQuestionAnsweringTaskModule(
        tokenizer_name_or_path="bert-base-uncased", max_length=128
    )


def test_encode_input(
    taskmodule, document, document_with_no_answer, document_with_multuple_answers
):
    inputs = taskmodule.encode_input(document)
    assert inputs is not None
    assert len(inputs) == 1
    expected_inputs = [
        101,
        2054,
        2003,
        1996,
        2034,
        2773,
        1029,
        102,
        2023,
        2003,
        1037,
        3231,
        6254,
        102,
    ]
    assert inputs[0].inputs == expected_inputs

    inputs = taskmodule.encode_input(document_with_no_answer)
    assert inputs is not None
    assert len(inputs) == 1
    assert inputs[0].inputs == expected_inputs

    inputs = taskmodule.encode_input(document_with_multuple_answers)
    assert inputs is not None
    assert len(inputs) == 1
    assert inputs[0].inputs == expected_inputs


def test_encode_target(taskmodule, document, document_with_no_answer):
    inputs = taskmodule.encode_input(document)
    targets = taskmodule.encode_target(inputs[0])
    assert targets is not None
    assert targets.start_position == 8
    assert targets.end_position == 8

    inputs = taskmodule.encode_input(document_with_no_answer)
    targets = taskmodule.encode_target(inputs[0])
    assert targets is not None
    assert targets.start_position == 0
    assert targets.end_position == 0


def test_get_question_layer(taskmodule, document, document_with_no_answer):
    question_layer = taskmodule.get_question_layer(document)
    assert question_layer is not None
    assert len(question_layer) == 1
    assert type(question_layer) is AnnotationList
    assert type(question_layer[0]) is Question
    assert question_layer[0].text == "What is the first word?"

    question_layer = taskmodule.get_question_layer(document_with_no_answer)
    assert question_layer is not None
    assert len(question_layer) == 1
    assert type(question_layer) is AnnotationList
    assert type(question_layer[0]) is Question
    assert question_layer[0].text == "What is the first word?"


def test_get_answer_layer(taskmodule, document, document_with_no_answer):
    answer_layer = taskmodule.get_answer_layer(document)
    assert answer_layer is not None
    assert len(answer_layer) == 1
    assert type(answer_layer) is AnnotationList
    assert type(answer_layer[0]) is ExtractiveAnswer
    assert answer_layer[0].question.text == "What is the first word?"
    assert answer_layer[0].start == 0
    assert answer_layer[0].end == 3

    answer_layer = taskmodule.get_answer_layer(document_with_no_answer)
    assert answer_layer is not None
    assert len(answer_layer) == 0
    assert type(answer_layer) is AnnotationList


def test_get_context(taskmodule, document, document_with_no_answer):
    context = taskmodule.get_context(document)
    assert context is not None
    assert context == "This is a test document"

    context = taskmodule.get_context(document_with_no_answer)
    assert context is not None
    assert context == "This is a test document"


def test_collate(taskmodule, document, document_with_no_answer):
    task_encodings = taskmodule.encode([document, document_with_no_answer])
    batch_encoding = taskmodule.collate(task_encodings)
    assert batch_encoding is not None
    assert len(batch_encoding) == 2
    inputs, targets = batch_encoding
    assert inputs is not None
    assert targets is None

    task_encodings = taskmodule.encode([document, document_with_no_answer], encode_target=True)
    batch_encoding = taskmodule.collate(task_encodings)
    assert batch_encoding is not None
    assert len(batch_encoding) == 2
    inputs, targets = batch_encoding
    assert inputs is not None
    assert set(inputs.data) == {"input_ids", "token_type_ids", "attention_mask"}
    assert inputs.data["input_ids"].shape == (2, 14)
    assert inputs.data["token_type_ids"].shape == (2, 14)
    assert inputs.data["attention_mask"].shape == (2, 14)
    assert targets is not None
    assert set(targets) == {"start_positions", "end_positions"}
    assert targets["start_positions"].shape == (2,)
    assert targets["end_positions"].shape == (2,)

    expected_inputs_ids = [
        [101, 2054, 2003, 1996, 2034, 2773, 1029, 102, 2023, 2003, 1037, 3231, 6254, 102],
        [101, 2054, 2003, 1996, 2034, 2773, 1029, 102, 2023, 2003, 1037, 3231, 6254, 102],
    ]
    expected_token_type_ids = [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    ]
    expected_attention_mask = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    assert inputs.data["input_ids"].tolist() == expected_inputs_ids
    assert inputs.data["token_type_ids"].tolist() == expected_token_type_ids
    assert inputs.data["attention_mask"].tolist() == expected_attention_mask

    expected_start_positions = [8, 0]
    expected_end_positions = [8, 0]
    assert targets["start_positions"].tolist() == expected_start_positions
    assert targets["end_positions"].tolist() == expected_end_positions


def test_unbatch_output(taskmodule, model_outputs):
    unbatched_output = taskmodule.unbatch_output(model_outputs)
    assert unbatched_output is not None

    result = [
        {
            "start": output.start,
            "end": output.end,
            "start_probability": round(float(output.start_probability), 2),
            "end_probability": round(float(output.end_probability), 2),
        }
        for output in unbatched_output
    ]
    expected_result = [
        {"start": 0, "end": 3, "start_probability": 0.93, "end_probability": 0.93},
        {"start": 13, "end": 18, "start_probability": 0.93, "end_probability": 0.93},
    ]

    assert result == expected_result


def test_create_annotations_from_output(taskmodule, document_batch, model_outputs):
    task_encodings_for_batch = taskmodule.encode(document_batch, encode_target=True)
    unbatched_outputs = taskmodule.unbatch_output(model_outputs)

    named_annotations_per_document = defaultdict(list)
    for task_encoding, task_output in zip(task_encodings_for_batch, unbatched_outputs):
        annotations = taskmodule.create_annotations_from_output(task_encoding, task_output)
        named_annotations_per_document[task_encoding.document.id].extend(list(annotations))

    assert named_annotations_per_document is not None
    assert len(named_annotations_per_document) == 2
