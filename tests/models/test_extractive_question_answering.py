import json

import pytest
import torch
import transformers

from pie_modules.annotations import ExtractiveAnswer, Question
from pie_modules.documents import ExtractiveQADocument
from pie_modules.models.simple_extractive_question_answering import (
    SimpleExtractiveQuestionAnsweringModel,
)
from pie_modules.taskmodules.extractive_question_answering import (
    ExtractiveQuestionAnsweringTaskModule,
)
from tests import FIXTURES_ROOT

DUMP_FIXTURES = True
FIXTURES_TASKMODULE_DATA_PATH = FIXTURES_ROOT / "taskmodules" / "extractive_question_answering"


@pytest.fixture
def documents():
    document0 = ExtractiveQADocument(text="This is a test document", id="doc0")
    document0.questions.append(Question(text="What is the first word?"))
    document0.answers.append(ExtractiveAnswer(question=document0.questions[0], start=0, end=3))

    document1 = ExtractiveQADocument(text="Oranges are orange in color.", id="doc1")
    document1.questions.append(Question(text="What color are oranges?"))
    document1.answers.append(ExtractiveAnswer(question=document1.questions[0], start=23, end=27))

    document2 = ExtractiveQADocument(
        text="This is a test document that has two questions attached to it.", id="doc2"
    )
    document2.questions.append(Question(text="What type of document is this?"))
    document2.questions.append(Question(text="How many questions are attached to this document?"))
    document2.answers.append(ExtractiveAnswer(question=document2.questions[0], start=11, end=14))
    document2.answers.append(ExtractiveAnswer(question=document2.questions[1], start=34, end=36))

    documents = [document0, document1, document2]
    return documents


@pytest.mark.skipif(
    condition=not DUMP_FIXTURES,
    reason="Only need to dump the data if taskmodule has changed",
)
def test_dump_fixtures(documents):
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = ExtractiveQuestionAnsweringTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_length=512,
    )

    task_encodings = taskmodule.encode(documents, encode_target=True)
    batch_encoding = taskmodule.collate(task_encodings)

    FIXTURES_TASKMODULE_DATA_PATH.mkdir(parents=True, exist_ok=True)
    filepath = FIXTURES_TASKMODULE_DATA_PATH / "batch_encoding_inputs.json"

    inputs = {key: tensor.tolist() for key, tensor in batch_encoding[0].items()}
    targets = {key: tensor.tolist() for key, tensor in batch_encoding[1].items()}
    converted_batch_encoding = {
        "inputs": inputs,
        "targets": targets,
    }

    with open(filepath, "w") as f:
        json.dump(converted_batch_encoding, f)
    return converted_batch_encoding


@pytest.fixture
def batch(documents):
    filepath = FIXTURES_TASKMODULE_DATA_PATH / "batch_encoding_inputs.json"
    with open(filepath) as f:
        batch_encoding = json.load(f)

    inputs = {key: torch.LongTensor(tensor) for key, tensor in batch_encoding["inputs"].items()}
    targets = {key: torch.LongTensor(tensor) for key, tensor in batch_encoding["targets"].items()}
    return (inputs, targets)


def get_model(
    monkeypatch,
    model_type,
    batch_size,
    seq_len,
    add_dummy_linear=False,
    **model_kwargs,
):
    class MockConfig:
        def __init__(
            self,
            hidden_size: int = 10,
            model_type=model_type,
        ) -> None:
            self.hidden_size = hidden_size
            self.model_type = model_type

    class MockModel(torch.nn.Module):
        def __init__(self, batch_size, seq_len, hidden_size, add_dummy_linear) -> None:
            super().__init__()
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.hidden_size = hidden_size
            if add_dummy_linear:
                self.dummy_linear = torch.nn.Linear(self.hidden_size, 99)

        def __call__(self, *args, **kwargs):
            torch.manual_seed(42)
            start_logits = torch.FloatTensor(torch.rand(self.batch_size, self.seq_len))
            end_logits = torch.FloatTensor(torch.rand(self.batch_size, self.seq_len))
            loss = torch.FloatTensor(torch.rand(1))
            return transformers.modeling_outputs.QuestionAnsweringModelOutput(
                start_logits=start_logits,
                end_logits=end_logits,
                loss=loss,
            )

    hidden_size = 10

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda model_name_or_path: MockConfig(hidden_size=hidden_size, model_type=model_type),
    )
    monkeypatch.setattr(
        transformers.AutoModelForQuestionAnswering,
        "from_pretrained",
        lambda model_name_or_path, config: MockModel(
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_size=hidden_size,
            add_dummy_linear=add_dummy_linear,
        ),
    )

    # set seed to make the classifier deterministic
    torch.manual_seed(42)
    result = SimpleExtractiveQuestionAnsweringModel(
        model_name_or_path=model_type,
        max_input_length=seq_len,
        **model_kwargs,
    )
    assert not result.is_from_pretrained

    return result


@pytest.fixture
def model(monkeypatch, batch):
    inputs, targets = batch
    model = get_model(
        monkeypatch=monkeypatch,
        model_type="bert",
        batch_size=inputs["input_ids"].shape[0],
        seq_len=inputs["input_ids"].shape[1],
        add_dummy_linear=True,
    )
    return model


def test_get_model(monkeypatch, model):
    assert model is not None
    assert isinstance(model, SimpleExtractiveQuestionAnsweringModel)


def test_forward(batch, model):
    inputs, targets = batch
    batch_size, seq_len = inputs["input_ids"].shape

    # set seed to make sure the output is deterministic
    torch.manual_seed(42)
    output = model.forward(inputs)
    assert set(output) == {"start_logits", "end_logits", "loss"}
    start_logits = output["start_logits"]
    end_logits = output["end_logits"]
    loss = output["loss"]
    assert start_logits.shape == (batch_size, seq_len)
    assert end_logits.shape == (batch_size, seq_len)
    assert loss.shape == (1,)
    expected_loss = torch.FloatTensor([0.04587])
    torch.testing.assert_close(output["loss"], expected_loss)


def test_step(batch, model):
    torch.manual_seed(42)
    loss = model.step("train", batch)
    assert loss is not None
    expected_loss = torch.FloatTensor([0.04587])
    torch.testing.assert_close(loss, expected_loss)


def test_training_step(batch, model):
    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None
    expected_loss = torch.FloatTensor([0.04587])
    torch.testing.assert_close(loss, expected_loss)


def test_validation_step(batch, model):
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    expected_loss = torch.FloatTensor([0.04587])
    torch.testing.assert_close(loss, expected_loss)


def test_test_step(batch, model):
    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None
    expected_loss = torch.FloatTensor([0.04587])
    torch.testing.assert_close(loss, expected_loss)


def test_optim(model):
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == 1e-05
