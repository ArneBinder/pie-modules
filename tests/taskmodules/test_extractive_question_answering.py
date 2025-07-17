import pytest
import torch
import transformers
from pie_core import AnnotationLayer

from pie_modules.annotations import ExtractiveAnswer, Question
from pie_modules.documents import TextDocumentWithQuestionsAndExtractiveAnswers
from pie_modules.taskmodules.extractive_question_answering import (
    ExtractiveQuestionAnsweringTaskModule,
)


@pytest.fixture()
def document():
    document = TextDocumentWithQuestionsAndExtractiveAnswers(
        text="This is a test document", id="doc0"
    )
    document.questions.append(Question(text="What is the first word?"))
    document.answers.append(ExtractiveAnswer(question=document.questions[0], start=0, end=4))
    assert str(document.answers[0]) == "This"
    return document


@pytest.fixture()
def document1():
    document1 = TextDocumentWithQuestionsAndExtractiveAnswers(
        text="This is the second document", id="doc1"
    )
    document1.questions.append(Question(text="Which document is this?"))
    document1.answers.append(ExtractiveAnswer(question=document1.questions[0], start=13, end=18))
    assert str(document1.answers[0]) == "second"
    return document1


@pytest.fixture()
def document_with_no_answer():
    document = TextDocumentWithQuestionsAndExtractiveAnswers(
        text="This is a test document", id="document_with_no_answer"
    )
    document.questions.append(Question(text="What is the first word?"))
    return document


@pytest.fixture()
def document_with_multiple_answers():
    document = TextDocumentWithQuestionsAndExtractiveAnswers(
        text="This is a test document", id="document_with_multiple_answers"
    )
    document.questions.append(Question(text="What is the first word?"))
    document.answers.append(ExtractiveAnswer(question=document.questions[0], start=0, end=4))
    assert str(document.answers[0]) == "This"
    document.answers.append(ExtractiveAnswer(question=document.questions[0], start=0, end=7))
    assert str(document.answers[1]) == "This is"
    return document


@pytest.fixture()
def taskmodule():
    return ExtractiveQuestionAnsweringTaskModule(
        tokenizer_name_or_path="bert-base-uncased", max_length=128
    )


def test_encode_input(
    taskmodule, document, document_with_no_answer, document_with_multiple_answers
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

    inputs = taskmodule.encode_input(document_with_multiple_answers)
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
    assert type(question_layer) is AnnotationLayer
    assert type(question_layer[0]) is Question
    assert question_layer[0].text == "What is the first word?"

    question_layer = taskmodule.get_question_layer(document_with_no_answer)
    assert question_layer is not None
    assert len(question_layer) == 1
    assert type(question_layer) is AnnotationLayer
    assert type(question_layer[0]) is Question
    assert question_layer[0].text == "What is the first word?"


def test_get_answer_layer(taskmodule, document, document_with_no_answer):
    answer_layer = taskmodule.get_answer_layer(document)
    assert answer_layer is not None
    assert len(answer_layer) == 1
    assert type(answer_layer) is AnnotationLayer
    assert type(answer_layer[0]) is ExtractiveAnswer
    assert answer_layer[0].question.text == "What is the first word?"
    assert answer_layer[0].start == 0
    assert answer_layer[0].end == 4

    answer_layer = taskmodule.get_answer_layer(document_with_no_answer)
    assert answer_layer is not None
    assert len(answer_layer) == 0
    assert type(answer_layer) is AnnotationLayer


def test_get_context(taskmodule, document, document_with_no_answer):
    context = taskmodule.get_context(document)
    assert context is not None
    assert context == "This is a test document"

    context = taskmodule.get_context(document_with_no_answer)
    assert context is not None
    assert context == "This is a test document"


@pytest.fixture()
def documents(document, document_with_no_answer):
    return [document, document_with_no_answer]


@pytest.fixture()
def batch_without_targets(taskmodule, documents):
    task_encodings = taskmodule.encode(documents)
    batch_encoding = taskmodule.collate(task_encodings)
    return batch_encoding


def test_collate_without_targets(batch_without_targets):
    assert batch_without_targets is not None
    assert len(batch_without_targets) == 2
    inputs, targets = batch_without_targets
    assert inputs is not None
    assert targets is None


@pytest.fixture()
def task_encodings(taskmodule, documents):
    task_encodings = taskmodule.encode(documents, encode_target=True)
    return task_encodings


@pytest.fixture()
def batch(taskmodule, task_encodings):
    batch_encoding = taskmodule.collate(task_encodings)
    return batch_encoding


def test_collate_with_targets(batch):
    assert batch is not None
    assert len(batch) == 2
    inputs, targets = batch
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


@pytest.fixture()
def model_outputs(batch):
    # create probabilities that "perfectly" model the batch targets
    inputs, targets = batch
    start_probs = torch.zeros_like(inputs.input_ids, dtype=torch.float) + 0.05
    end_probs = torch.zeros_like(inputs.input_ids, dtype=torch.float) + 0.05
    # set target positions to 0.95 as a dummy value
    for idx, (start_position, end_position) in enumerate(
        zip(targets["start_positions"], targets["end_positions"])
    ):
        start_probs[idx, start_position] = 0.95
        end_probs[idx, end_position] = 0.95

    # convert probs to logits
    start_logits = torch.log(start_probs / (1 - start_probs))
    end_logits = torch.log(end_probs / (1 - end_probs))

    model_outputs = transformers.modeling_outputs.QuestionAnsweringModelOutput(
        start_logits=start_logits,
        end_logits=end_logits,
    )
    return model_outputs


@pytest.fixture()
def unbatched_output(taskmodule, model_outputs):
    return taskmodule.unbatch_output(model_outputs)


def test_unbatch_output(unbatched_output):
    assert unbatched_output is not None
    assert len(unbatched_output) == 2
    # check first result
    assert unbatched_output[0].start == 8
    assert unbatched_output[0].end == 8
    assert unbatched_output[0].start_probability == pytest.approx(0.9652407)
    assert unbatched_output[0].end_probability == pytest.approx(0.9652407)
    # check second result
    assert unbatched_output[1].start == 0
    assert unbatched_output[1].end == 0
    assert unbatched_output[1].start_probability == pytest.approx(0.9652407)
    assert unbatched_output[1].end_probability == pytest.approx(0.9652407)


def test_create_annotations_from_output(taskmodule, task_encodings, unbatched_output, documents):
    taskmodule.combine_outputs(task_encodings, unbatched_output)
    assert len(documents) > 0
    for doc in documents:
        gold_annotations = doc.answers
        predicted_annotations = doc.answers.predictions
        assert len(predicted_annotations) == len(gold_annotations)
        for predicted_annotation, gold_annotation in zip(predicted_annotations, gold_annotations):
            # we did construct the predicted annotations from the gold annotations, so they should be equal
            assert predicted_annotation == gold_annotation
            assert predicted_annotation.score == pytest.approx(0.9316896200180054)
