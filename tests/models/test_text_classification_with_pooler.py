import pytest
import torch
import transformers
from transformers.modeling_outputs import BaseModelOutputWithPooling

from pie_models.models import TextClassificationModelWithPooler
from pie_models.taskmodules import RETextClassificationWithIndicesTaskModule


@pytest.fixture(scope="session")
def documents(dataset):
    return dataset["train"]


@pytest.fixture(scope="module")
def taskmodule():
    tokenizer_name_or_path = "bert-base-cased"
    taskmodule = RETextClassificationWithIndicesTaskModule(
        tokenizer_name_or_path=tokenizer_name_or_path,
        add_argument_indices_to_input=True,
    )
    return taskmodule


@pytest.fixture
def prepared_taskmodule(taskmodule, documents):
    taskmodule.prepare(documents)
    return taskmodule


class MockConfig:
    def __init__(self, hidden_size: int = 10, classifier_dropout: float = 1.0) -> None:
        self.hidden_size = hidden_size
        self.classifier_dropout = classifier_dropout


class MockModel:
    def __init__(self, batch_size, seq_len, hidden_size) -> None:
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size

    def __call__(self, *args, **kwargs):
        last_hidden_state = torch.rand(self.batch_size, self.seq_len, self.hidden_size)
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state)

    def resize_token_embeddings(self, new_num_tokens):
        pass


@pytest.fixture(params=["cls_token", "mention_pooling", "start_tokens"])
def mock_model(monkeypatch, documents, prepared_taskmodule, request):
    encodings = prepared_taskmodule.encode(documents, encode_target=True)
    inputs, _ = prepared_taskmodule.collate(encodings)

    batch_size, seq_len = inputs["input_ids"].shape
    hidden_size = 10
    num_classes = 4
    tokenizer_vocab_size = 30000

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda model_name_or_path: MockConfig(hidden_size=hidden_size, classifier_dropout=1.0),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda model_name_or_path, config: MockModel(
            batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size
        ),
    )

    model = TextClassificationModelWithPooler(
        model_name_or_path="some-model-name",
        num_classes=num_classes,
        tokenizer_vocab_size=tokenizer_vocab_size,
        ignore_index=0,
        pooler=request.param,
        # disable warmup because it would require a trainer and a datamodule to get the total number of training steps
        warmup_proportion=0.0,
    )
    assert not model.is_from_pretrained

    return model


def test_forward(documents, prepared_taskmodule, mock_model):
    encodings = prepared_taskmodule.encode(documents, encode_target=True)
    inputs, _ = prepared_taskmodule.collate(encodings)

    output = mock_model.forward(inputs)

    assert set(output.keys()) == {"logits"}
    assert output["logits"].shape[0] == len(encodings)
    assert output["logits"].shape[1] == 4  # num classes is set to 4 in mock_model()
    if mock_model.pooler_config["type"] == "cls_token":
        assert mock_model.classifier.in_features == 10  # hidden size = 10 in mock_model()
    elif mock_model.pooler_config["type"] in ["mention_pooling", "start_tokens"]:
        assert (
            mock_model.classifier.in_features == 20
        )  # hidden size = 10 in mock_model(), *2 for concat
