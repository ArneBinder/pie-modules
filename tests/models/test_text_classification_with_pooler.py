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


def test_pooler(documents, prepared_taskmodule, mock_model):
    encodings = prepared_taskmodule.encode(documents, encode_target=True)
    inputs, _ = prepared_taskmodule.collate(encodings)

    model_inputs = {}
    pooler_inputs = {}
    for k, v in inputs.items():
        if k.startswith("pooler_"):
            pooler_inputs[k[len("pooler_") :]] = v
        else:
            model_inputs[k] = inputs[k]

    # set the seed to make the test deterministic
    torch.manual_seed(42)

    outputs = mock_model.model(**model_inputs)
    hidden_state = outputs.last_hidden_state
    # just as sanity check: compare the shape and the first values of the first batch element of the input
    assert hidden_state.shape == torch.Size([7, 22, 10])
    torch.testing.assert_close(
        hidden_state[0, 0],
        torch.tensor(
            [
                0.8822692632675171,
                0.9150039553642273,
                0.38286375999450684,
                0.9593056440353394,
                0.3904482126235962,
                0.600895345211029,
                0.2565724849700928,
                0.7936413288116455,
                0.9407714605331421,
                0.13318592309951782,
            ],
        ),
    )
    assert set(pooler_inputs) == {"start_indices", "end_indices"}
    torch.testing.assert_close(
        pooler_inputs["start_indices"],
        torch.tensor([[2, 10], [5, 13], [5, 17], [17, 11], [5, 13], [14, 18], [18, 14]]),
    )
    torch.testing.assert_close(
        pooler_inputs["end_indices"],
        torch.tensor([[6, 11], [9, 14], [9, 18], [18, 12], [9, 14], [15, 19], [19, 15]]),
    )

    pooled_output = mock_model.pooler(hidden_state, **pooler_inputs)
    assert pooled_output is not None

    # we compare just the shape and the first and last values of the first batch element
    if mock_model.pooler_config["type"] == "cls_token":
        assert pooled_output.shape == torch.Size([7, 10])
        torch.testing.assert_close(
            pooled_output[0],
            torch.tensor(
                [
                    0.8822692632675171,
                    0.9150039553642273,
                    0.38286375999450684,
                    0.9593056440353394,
                    0.3904482126235962,
                    0.600895345211029,
                    0.2565724849700928,
                    0.7936413288116455,
                    0.9407714605331421,
                    0.13318592309951782,
                ],
            ),
        )
        torch.testing.assert_close(
            pooled_output[-1],
            torch.tensor(
                [
                    0.1607622504234314,
                    0.5514103174209595,
                    0.5479037761688232,
                    0.5692103505134583,
                    0.07835221290588379,
                    0.025107860565185547,
                    0.7300586700439453,
                    0.9287979602813721,
                    0.05631381273269653,
                    0.685197114944458,
                ],
            ),
        )

    elif mock_model.pooler_config["type"] == "mention_pooling":
        assert pooled_output.shape == torch.Size([7, 20])
        torch.testing.assert_close(
            pooled_output[0],
            torch.tensor(
                [
                    0.951554536819458,
                    0.4413635730743408,
                    0.8860136866569519,
                    0.9464110732078552,
                    0.7890297770500183,
                    0.8089749813079834,
                    0.788632333278656,
                    0.9039816856384277,
                    0.8913041353225708,
                    0.34231340885162354,
                    0.21366488933563232,
                    0.6249018311500549,
                    0.43400341272354126,
                    0.13705700635910034,
                    0.5117283463478088,
                    0.15845924615859985,
                    0.07580167055130005,
                    0.2246686816215515,
                    0.06239396333694458,
                    0.1816309690475464,
                ],
            ),
        )
        torch.testing.assert_close(
            pooled_output[-1],
            torch.tensor(
                [
                    0.9258986711502075,
                    0.47120314836502075,
                    0.5970233082771301,
                    0.20084011554718018,
                    0.6932605504989624,
                    0.02057945728302002,
                    0.7358364462852478,
                    0.5917077660560608,
                    0.6536521315574646,
                    0.6830741763114929,
                    0.05204510688781738,
                    0.7196753025054932,
                    0.5311281681060791,
                    0.3370867967605591,
                    0.2904888391494751,
                    0.2421920895576477,
                    0.9047484993934631,
                    0.4137304425239563,
                    0.8606035113334656,
                    0.9463248252868652,
                ],
            ),
        )
    elif mock_model.pooler_config["type"] == "start_tokens":
        assert pooled_output.shape == torch.Size([7, 20])
        torch.testing.assert_close(
            pooled_output[0],
            torch.tensor(
                [
                    0.9345980882644653,
                    0.5935796499252319,
                    0.8694044351577759,
                    0.5677152872085571,
                    0.7410940527915955,
                    0.42940449714660645,
                    0.8854429125785828,
                    0.5739044547080994,
                    0.2665800452232361,
                    0.6274491548538208,
                    0.6057037711143494,
                    0.3725206255912781,
                    0.7980347275733948,
                    0.8399046063423157,
                    0.13741332292556763,
                    0.2330659031867981,
                    0.9578309655189514,
                    0.3312837481498718,
                    0.3227418065071106,
                    0.016202688217163086,
                ],
            ),
        )
        torch.testing.assert_close(
            pooled_output[-1],
            torch.tensor(
                [
                    0.2910900115966797,
                    0.21942031383514404,
                    0.27615296840667725,
                    0.15551382303237915,
                    0.14953309297561646,
                    0.05449223518371582,
                    0.056860387325286865,
                    0.6304993033409119,
                    0.8049269914627075,
                    0.9694236516952515,
                    0.5081672072410583,
                    0.06770873069763184,
                    0.7817272543907166,
                    0.9528661966323853,
                    0.5884308815002441,
                    0.1284925937652588,
                    0.6165931820869446,
                    0.6815068125724792,
                    0.895921528339386,
                    0.23396503925323486,
                ],
            ),
        )
    else:
        raise ValueError(f'Unknown pooler type {mock_model.pooler_config["type"]}')
