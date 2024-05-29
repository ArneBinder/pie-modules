from typing import Dict

import pytest
import torch
from pytorch_lightning import Trainer
from torch import LongTensor, tensor
from torch.optim.lr_scheduler import LambdaLR
from transformers.modeling_outputs import SequenceClassifierOutput

from pie_modules.models import SequenceClassificationModelWithPooler
from pie_modules.models.sequence_classification_with_pooler import OutputType
from tests.models import trunc_number

NUM_CLASSES = 4
POOLER = "start_tokens"


@pytest.fixture
def inputs() -> Dict[str, LongTensor]:
    result_dict = {
        "input_ids": torch.tensor(
            [
                [
                    101,
                    28998,
                    13832,
                    3121,
                    2340,
                    138,
                    28996,
                    1759,
                    1120,
                    28999,
                    139,
                    28997,
                    119,
                    102,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    28998,
                    13832,
                    3121,
                    2340,
                    144,
                    28996,
                    1759,
                    1120,
                    28999,
                    145,
                    28997,
                    119,
                    1262,
                    1771,
                    146,
                    119,
                    102,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    28998,
                    13832,
                    3121,
                    2340,
                    144,
                    28996,
                    1759,
                    1120,
                    145,
                    119,
                    1262,
                    1771,
                    28999,
                    146,
                    28997,
                    119,
                    102,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    13832,
                    3121,
                    2340,
                    144,
                    1759,
                    1120,
                    28999,
                    145,
                    28997,
                    119,
                    1262,
                    1771,
                    28998,
                    146,
                    28996,
                    119,
                    102,
                    0,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    28998,
                    13832,
                    3121,
                    2340,
                    150,
                    28996,
                    1759,
                    1120,
                    28999,
                    151,
                    28997,
                    119,
                    1262,
                    1122,
                    1771,
                    152,
                    119,
                    102,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    13832,
                    3121,
                    2340,
                    150,
                    1759,
                    1120,
                    151,
                    119,
                    1262,
                    28998,
                    1122,
                    28996,
                    1771,
                    28999,
                    152,
                    28997,
                    119,
                    102,
                ],
                [
                    101,
                    1752,
                    5650,
                    119,
                    13832,
                    3121,
                    2340,
                    150,
                    1759,
                    1120,
                    151,
                    119,
                    1262,
                    28999,
                    1122,
                    28997,
                    1771,
                    28998,
                    152,
                    28996,
                    119,
                    102,
                ],
            ]
        ).to(torch.long),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ).to(torch.long),
        "pooler_start_indices": torch.tensor(
            [[2, 10], [5, 13], [5, 17], [17, 11], [5, 13], [14, 18], [18, 14]]
        ).to(torch.long),
        "pooler_end_indices": torch.tensor(
            [[6, 11], [9, 14], [9, 18], [18, 12], [9, 14], [15, 19], [19, 15]]
        ).to(torch.long),
    }

    return result_dict


@pytest.fixture
def targets() -> Dict[str, LongTensor]:
    return {"labels": torch.tensor([0, 1, 2, 3, 1, 2, 3]).to(torch.long)}


@pytest.fixture
def model() -> SequenceClassificationModelWithPooler:
    torch.manual_seed(42)
    result = SequenceClassificationModelWithPooler(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
        pooler=POOLER,
    )
    return result


def test_model(model):
    assert model is not None
    named_parameters = dict(model.named_parameters())
    parameter_means = {k: trunc_number(v.mean().item(), 7) for k, v in named_parameters.items()}
    parameter_means_expected = {
        "classifier.bias": -0.0253964,
        "classifier.weight": -0.000511,
        "model.embeddings.LayerNorm.bias": -0.0294608,
        "model.embeddings.LayerNorm.weight": 1.312345,
        "model.embeddings.position_embeddings.weight": 5.5e-05,
        "model.embeddings.token_type_embeddings.weight": -0.0015419,
        "model.embeddings.word_embeddings.weight": 0.0031152,
        "model.encoder.layer.0.attention.output.LayerNorm.bias": 0.0608714,
        "model.encoder.layer.0.attention.output.LayerNorm.weight": 1.199831,
        "model.encoder.layer.0.attention.output.dense.bias": 0.0007209,
        "model.encoder.layer.0.attention.output.dense.weight": 3.01e-05,
        "model.encoder.layer.0.attention.self.key.bias": 0.0020557,
        "model.encoder.layer.0.attention.self.key.weight": 0.0003863,
        "model.encoder.layer.0.attention.self.query.bias": 0.0185744,
        "model.encoder.layer.0.attention.self.query.weight": -0.0003949,
        "model.encoder.layer.0.attention.self.value.bias": 0.0065417,
        "model.encoder.layer.0.attention.self.value.weight": 4.22e-05,
        "model.encoder.layer.0.intermediate.dense.bias": -0.1219958,
        "model.encoder.layer.0.intermediate.dense.weight": -0.0011731,
        "model.encoder.layer.0.output.LayerNorm.bias": 0.005295,
        "model.encoder.layer.0.output.LayerNorm.weight": 1.2419648,
        "model.encoder.layer.0.output.dense.bias": -0.0013031,
        "model.encoder.layer.0.output.dense.weight": -0.0002212,
        "model.encoder.layer.1.attention.output.LayerNorm.bias": 0.0443237,
        "model.encoder.layer.1.attention.output.LayerNorm.weight": 1.0377343,
        "model.encoder.layer.1.attention.output.dense.bias": 0.0041446,
        "model.encoder.layer.1.attention.output.dense.weight": -2.43e-05,
        "model.encoder.layer.1.attention.self.key.bias": 0.0045062,
        "model.encoder.layer.1.attention.self.key.weight": 0.0001333,
        "model.encoder.layer.1.attention.self.query.bias": -0.0358397,
        "model.encoder.layer.1.attention.self.query.weight": -0.0007321,
        "model.encoder.layer.1.attention.self.value.bias": -0.0007094,
        "model.encoder.layer.1.attention.self.value.weight": 0.0001012,
        "model.encoder.layer.1.intermediate.dense.bias": -0.1247257,
        "model.encoder.layer.1.intermediate.dense.weight": -0.001344,
        "model.encoder.layer.1.output.LayerNorm.bias": -0.0474442,
        "model.encoder.layer.1.output.LayerNorm.weight": 1.017162,
        "model.encoder.layer.1.output.dense.bias": 0.000677,
        "model.encoder.layer.1.output.dense.weight": -5.32e-05,
        "model.pooler.dense.bias": -0.0052078,
        "model.pooler.dense.weight": 0.0001295,
        "pooler.pooler.missing_embeddings": 0.0630417,
    }
    assert parameter_means == parameter_means_expected


def test_model_pickleable(model):
    import pickle

    pickle.dumps(model)


@pytest.fixture
def model_output(model, inputs) -> OutputType:
    # set seed to make sure the output is deterministic
    torch.manual_seed(42)
    return model(inputs, return_hidden_states=True)


def test_forward_hidden_states(model_output):
    last_hidden_state = model_output["hidden_states"][0]
    # check first token of first batch ...
    torch.testing.assert_close(
        last_hidden_state[0, 0],
        tensor(
            [
                -0.646003007888794,
                0.7487114071846008,
                -3.576923370361328,
                -1.6957248449325562,
                0.44349995255470276,
                1.2073384523391724,
                -0.9055725932121277,
                0.2955082654953003,
                -0.4938056170940399,
                -0.07763966172933578,
                -0.5820843577384949,
                -0.35837045311927795,
                -2.177708387374878,
                -0.10517356544733047,
                1.8452434539794922,
                -1.3576951026916504,
                0.33492806553840637,
                -0.6974215507507324,
                -1.853518009185791,
                1.769407868385315,
                1.0600963830947876,
                0.42672356963157654,
                2.0446269512176514,
                1.3517223596572876,
                1.24276602268219,
                -0.9950742721557617,
                -0.13340961933135986,
                0.9109700918197632,
                0.1269163191318512,
                -0.1406235694885254,
                -0.23685386776924133,
                -1.4826629161834717,
                -2.1277058124542236,
                1.1538283824920654,
                0.5683708190917969,
                -2.498945713043213,
                0.21515989303588867,
                0.46334323287010193,
                -2.0521392822265625,
                -1.1138701438903809,
                1.3039685487747192,
                -0.7592583894729614,
                1.4672906398773193,
                -2.51831316947937,
                0.488108366727829,
                -0.877996563911438,
                -1.1584198474884033,
                -1.3187131881713867,
                0.283611923456192,
                -0.1542758047580719,
                -0.9190185070037842,
                1.7347131967544556,
                1.1241217851638794,
                0.45560094714164734,
                0.03938984498381615,
                -0.4590461254119873,
                -0.04643985256552696,
                -0.6134224534034729,
                -1.085078239440918,
                1.70975923538208,
                0.38798195123672485,
                0.11232337355613708,
                -1.5158600807189941,
                -0.500819981098175,
                -0.8203051686286926,
                0.0022824169136583805,
                0.8608205318450928,
                0.17795918881893158,
                -0.1826174557209015,
                1.0026092529296875,
                0.4817505180835724,
                0.643688440322876,
                0.16746684908866882,
                -0.002742352895438671,
                -0.4707050621509552,
                -0.0668899193406105,
                1.3160650730133057,
                1.6299465894699097,
                2.6769676208496094,
                0.682036280632019,
                0.9359247088432312,
                -0.5984869003295898,
                -0.002743073273450136,
                -0.08123790472745895,
                0.014478045515716076,
                -0.5866737365722656,
                0.07180003076791763,
                -0.193366140127182,
                2.07020902633667,
                -0.45512548089027405,
                1.1995497941970825,
                0.23864911496639252,
                -0.8044520020484924,
                0.14578185975551605,
                -0.11076415330171585,
                0.8898897767066956,
                -1.457258939743042,
                -0.033318862318992615,
                -1.253710150718689,
                -0.3255557119846344,
                0.14961537718772888,
                -0.901416003704071,
                0.3102920353412628,
                -0.29155492782592773,
                0.889419436454773,
                -0.7135364413261414,
                -0.5418419241905212,
                0.5407666563987732,
                0.26518434286117554,
                -1.5459678173065186,
                0.6059889793395996,
                0.8830941319465637,
                0.6813007593154907,
                -1.1159330606460571,
                -0.5445968508720398,
                -0.2346498966217041,
                0.53174889087677,
                -0.6501120328903198,
                0.7147217988967896,
                0.34584733843803406,
                -0.08642642199993134,
                0.5532509684562683,
                0.9094955325126648,
                -0.2911924123764038,
                -0.8527982831001282,
                -2.9052340984344482,
                -1.6688522100448608,
                1.507781982421875,
            ]
        ),
    )
    # ... and the sum of the embedding dimension
    torch.testing.assert_close(
        last_hidden_state.sum(-1),
        tensor(
            [
                [
                    -7.613239765167236,
                    -7.064064025878906,
                    -7.7010674476623535,
                    -6.935450077056885,
                    -7.054670333862305,
                    -6.008757591247559,
                    -7.029489040374756,
                    -7.008832931518555,
                    -7.471133708953857,
                    -7.772115707397461,
                    -7.775435924530029,
                    -7.523985385894775,
                    -7.0200982093811035,
                    -6.846195697784424,
                    -7.525077819824219,
                    -7.593684196472168,
                    -7.653444290161133,
                    -7.480998992919922,
                    -7.153192520141602,
                    -7.131610870361328,
                    -7.413901329040527,
                    -7.367512226104736,
                ],
                [
                    -7.593530178070068,
                    -8.282629013061523,
                    -7.988146781921387,
                    -6.976714134216309,
                    -6.852786064147949,
                    -7.677435398101807,
                    -6.948968887329102,
                    -7.162008285522461,
                    -6.212387561798096,
                    -7.4167799949646,
                    -7.329948425292969,
                    -7.504671096801758,
                    -7.921816825866699,
                    -8.045998573303223,
                    -8.182007789611816,
                    -7.608787536621094,
                    -7.661788463592529,
                    -7.653861045837402,
                    -6.988530158996582,
                    -6.864537239074707,
                    -6.819007396697998,
                    -7.751153469085693,
                ],
                [
                    -7.596049785614014,
                    -8.256193161010742,
                    -7.9658002853393555,
                    -6.924246311187744,
                    -6.830434799194336,
                    -7.673945903778076,
                    -6.896265029907227,
                    -7.133776664733887,
                    -6.182246208190918,
                    -7.3677568435668945,
                    -7.296995162963867,
                    -7.285590171813965,
                    -7.371635437011719,
                    -7.44062614440918,
                    -7.439065933227539,
                    -7.833099842071533,
                    -7.881043434143066,
                    -7.903671741485596,
                    -7.817512512207031,
                    -7.303112983703613,
                    -7.013047218322754,
                    -7.770578861236572,
                ],
                [
                    -7.56870698928833,
                    -8.274161338806152,
                    -8.09821891784668,
                    -6.741868019104004,
                    -7.753932952880859,
                    -6.892941951751709,
                    -7.2075886726379395,
                    -6.621621608734131,
                    -7.439785957336426,
                    -7.559423446655273,
                    -7.866125583648682,
                    -7.865529537200928,
                    -7.927544593811035,
                    -7.4975385665893555,
                    -7.76503849029541,
                    -7.889374732971191,
                    -7.09637451171875,
                    -6.663265228271484,
                    -7.2689337730407715,
                    -6.775190353393555,
                    -6.681921005249023,
                    -7.542079925537109,
                ],
                [
                    -7.456245422363281,
                    -8.241959571838379,
                    -7.944664478302002,
                    -6.9324951171875,
                    -6.792070388793945,
                    -7.64109468460083,
                    -6.853059768676758,
                    -7.107330799102783,
                    -6.233638763427734,
                    -7.3772783279418945,
                    -7.261654853820801,
                    -7.46744441986084,
                    -7.869386672973633,
                    -7.995800018310547,
                    -8.146440505981445,
                    -7.4717912673950195,
                    -7.571569442749023,
                    -7.651631832122803,
                    -7.440743446350098,
                    -7.066323757171631,
                    -6.921482086181641,
                    -6.787400245666504,
                ],
                [
                    -7.463903427124023,
                    -8.204326629638672,
                    -7.997576713562012,
                    -6.6558990478515625,
                    -7.677325248718262,
                    -6.727311134338379,
                    -7.121250152587891,
                    -6.576358318328857,
                    -7.312101364135742,
                    -7.244853973388672,
                    -7.218317031860352,
                    -7.387792587280273,
                    -7.39663028717041,
                    -7.058981895446777,
                    -6.972137451171875,
                    -7.464954376220703,
                    -7.6565752029418945,
                    -7.70589542388916,
                    -7.917507648468018,
                    -7.814467906951904,
                    -7.260939121246338,
                    -6.888543128967285,
                ],
                [
                    -7.467811584472656,
                    -8.222318649291992,
                    -8.015830993652344,
                    -6.691221714019775,
                    -7.698590278625488,
                    -6.754973411560059,
                    -7.13471794128418,
                    -6.588311672210693,
                    -7.356595993041992,
                    -7.328269004821777,
                    -7.302258491516113,
                    -7.333678722381592,
                    -7.495467185974121,
                    -7.806238174438477,
                    -8.272199630737305,
                    -7.996397972106934,
                    -7.960136413574219,
                    -7.02586555480957,
                    -6.658778667449951,
                    -7.27991247177124,
                    -6.813238143920898,
                    -6.6815595626831055,
                ],
            ]
        ),
    )


def test_forward_logits(model_output, inputs):
    batch_size, seq_len = inputs["input_ids"].shape

    assert isinstance(model_output, SequenceClassifierOutput)

    logits = model_output.logits

    assert logits.shape == (batch_size, NUM_CLASSES)

    torch.testing.assert_close(
        logits,
        torch.tensor(
            [
                [
                    -0.29492446780204773,
                    -0.804599940776825,
                    -0.19659805297851562,
                    -1.0868580341339111,
                ],
                [
                    -0.3601434826850891,
                    -0.7454482316970825,
                    0.4882846474647522,
                    -1.0253472328186035,
                ],
                [
                    -0.40172430872917175,
                    -1.2119399309158325,
                    0.5856620669364929,
                    -1.0999149084091187,
                ],
                [
                    -0.09260234981775284,
                    -1.0742112398147583,
                    0.3299948275089264,
                    -0.5182554125785828,
                ],
                [
                    -0.40149545669555664,
                    -0.7731614708900452,
                    0.4616103768348694,
                    -1.0583568811416626,
                ],
                [
                    -0.14356234669685364,
                    -1.2634986639022827,
                    0.31660008430480957,
                    -0.7487461566925049,
                ],
                [
                    -0.11717557162046432,
                    -0.971996009349823,
                    0.3477852940559387,
                    -0.5993944406509399,
                ],
            ]
        ),
    )


def test_decode(model, model_output, inputs):
    decoded = model.decode(inputs=inputs, outputs=model_output)
    assert isinstance(decoded, dict)
    assert set(decoded) == {"labels", "probabilities"}
    labels = decoded["labels"]
    assert labels.shape == (inputs["input_ids"].shape[0],)
    torch.testing.assert_close(
        labels,
        torch.tensor([2, 2, 2, 2, 2, 2, 2]),
    )
    probabilities = decoded["probabilities"]
    assert probabilities.shape == (inputs["input_ids"].shape[0], NUM_CLASSES)
    torch.testing.assert_close(
        probabilities.round(decimals=4),
        torch.tensor(
            [
                [0.3168, 0.1903, 0.3495, 0.1435],
                [0.2207, 0.1502, 0.5156, 0.1135],
                [0.2161, 0.0961, 0.5802, 0.1075],
                [0.2814, 0.1054, 0.4294, 0.1838],
                [0.2184, 0.1506, 0.5177, 0.1132],
                [0.2893, 0.0944, 0.4583, 0.1580],
                [0.2751, 0.1170, 0.4380, 0.1699],
            ]
        ),
    )


def test_decode_with_multi_label(model_output, inputs):
    torch.manual_seed(42)
    model = SequenceClassificationModelWithPooler(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
        pooler=POOLER,
        multi_label=True,
    )
    decoded = model.decode(inputs=inputs, outputs=model_output)
    assert isinstance(decoded, dict)
    assert set(decoded) == {"labels", "probabilities"}
    labels = decoded["labels"]
    assert labels.shape == (inputs["input_ids"].shape[0], NUM_CLASSES)
    torch.testing.assert_close(
        labels,
        torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0],
            ]
        ),
    )
    probabilities = decoded["probabilities"]
    assert probabilities.shape == (inputs["input_ids"].shape[0], NUM_CLASSES)
    torch.testing.assert_close(
        probabilities.round(decimals=4),
        torch.tensor(
            [
                [0.4268, 0.3090, 0.4510, 0.2522],
                [0.4109, 0.3218, 0.6197, 0.2640],
                [0.4009, 0.2294, 0.6424, 0.2498],
                [0.4769, 0.2546, 0.5818, 0.3733],
                [0.4010, 0.3158, 0.6134, 0.2576],
                [0.4642, 0.2204, 0.5785, 0.3211],
                [0.4707, 0.2745, 0.5861, 0.3545],
            ]
        ),
    )


@pytest.fixture
def batch(inputs, targets):
    return inputs, targets


def test_training_step(batch, model):
    # set the seed to make sure the loss is deterministic
    torch.manual_seed(42)
    loss = model.training_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.3899686336517334))


def test_validation_step(batch, model):
    # set the seed to make sure the loss is deterministic
    torch.manual_seed(42)
    loss = model.validation_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.3899686336517334))


def test_test_step(batch, model):
    # set the seed to make sure the loss is deterministic
    torch.manual_seed(42)
    loss = model.test_step(batch, batch_idx=0)
    assert loss is not None
    torch.testing.assert_close(loss, torch.tensor(1.3899686336517334))


def test_base_model_named_parameters(model):
    base_model_named_parameters = dict(model.base_model_named_parameters())
    assert set(base_model_named_parameters) == {
        "model.pooler.dense.bias",
        "model.encoder.layer.0.intermediate.dense.weight",
        "model.encoder.layer.0.intermediate.dense.bias",
        "model.encoder.layer.1.attention.output.dense.weight",
        "model.encoder.layer.1.attention.output.LayerNorm.weight",
        "model.encoder.layer.1.attention.self.query.weight",
        "model.encoder.layer.1.output.dense.weight",
        "model.encoder.layer.0.output.dense.bias",
        "model.encoder.layer.1.intermediate.dense.bias",
        "model.encoder.layer.1.attention.self.value.bias",
        "model.encoder.layer.0.attention.output.dense.weight",
        "model.encoder.layer.0.attention.self.query.bias",
        "model.encoder.layer.0.attention.self.value.bias",
        "model.encoder.layer.1.output.dense.bias",
        "model.encoder.layer.1.attention.self.query.bias",
        "model.encoder.layer.1.attention.output.LayerNorm.bias",
        "model.encoder.layer.0.attention.self.query.weight",
        "model.encoder.layer.0.attention.output.LayerNorm.bias",
        "model.encoder.layer.0.attention.self.key.bias",
        "model.encoder.layer.1.intermediate.dense.weight",
        "model.encoder.layer.1.output.LayerNorm.bias",
        "model.encoder.layer.1.output.LayerNorm.weight",
        "model.encoder.layer.0.attention.self.key.weight",
        "model.encoder.layer.1.attention.output.dense.bias",
        "model.encoder.layer.0.attention.output.dense.bias",
        "model.embeddings.LayerNorm.bias",
        "model.encoder.layer.0.attention.self.value.weight",
        "model.encoder.layer.0.attention.output.LayerNorm.weight",
        "model.embeddings.token_type_embeddings.weight",
        "model.encoder.layer.0.output.LayerNorm.weight",
        "model.embeddings.position_embeddings.weight",
        "model.encoder.layer.1.attention.self.key.bias",
        "model.embeddings.LayerNorm.weight",
        "model.encoder.layer.0.output.LayerNorm.bias",
        "model.encoder.layer.1.attention.self.key.weight",
        "model.pooler.dense.weight",
        "model.encoder.layer.0.output.dense.weight",
        "model.embeddings.word_embeddings.weight",
        "model.encoder.layer.1.attention.self.value.weight",
    }


def test_task_named_parameters(model):
    task_named_parameters = dict(model.task_named_parameters())
    assert set(task_named_parameters) == {
        "classifier.weight",
        "pooler.pooler.missing_embeddings",
        "classifier.bias",
    }


def test_configure_optimizers_with_warmup():
    model = SequenceClassificationModelWithPooler(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
    )
    model.trainer = Trainer(max_epochs=10)
    optimizers_and_schedulers = model.configure_optimizers()
    assert len(optimizers_and_schedulers) == 2
    optimizers, schedulers = optimizers_and_schedulers
    assert len(optimizers) == 1
    assert len(schedulers) == 1
    optimizer = optimizers[0]
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 1e-05
    assert optimizer.defaults["weight_decay"] == 0.01
    assert optimizer.defaults["eps"] == 1e-08

    scheduler = schedulers[0]
    assert isinstance(scheduler, dict)
    assert set(scheduler) == {"scheduler", "interval"}
    assert isinstance(scheduler["scheduler"], LambdaLR)


def test_configure_optimizers_with_task_learning_rate(monkeypatch):
    model = SequenceClassificationModelWithPooler(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
        learning_rate=1e-5,
        task_learning_rate=1e-3,
        # disable warmup to make sure the scheduler is not added which would set the learning rate
        # to 0
        warmup_proportion=0.0,
    )
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2
    # base model parameters
    param_group = optimizer.param_groups[0]
    assert len(param_group["params"]) == 39
    assert param_group["lr"] == 1e-5
    # classifier head parameters
    param_group = optimizer.param_groups[1]
    assert len(param_group["params"]) == 2
    assert param_group["lr"] == 1e-3
    # ensure that all parameters are covered
    assert set(optimizer.param_groups[0]["params"] + optimizer.param_groups[1]["params"]) == set(
        model.parameters()
    )


def test_freeze_base_model(monkeypatch, inputs, targets):
    model = SequenceClassificationModelWithPooler(
        model_name_or_path="prajjwal1/bert-tiny",
        num_classes=NUM_CLASSES,
        freeze_base_model=True,
        # disable warmup to make sure the scheduler is not added which would set the learning rate
        # to 0
        warmup_proportion=0.0,
    )
    base_model_params = [param for name, param in model.base_model_named_parameters()]
    task_params = [param for name, param in model.task_named_parameters()]
    assert len(base_model_params) + len(task_params) == len(list(model.parameters()))
    for param in base_model_params:
        assert not param.requires_grad
    for param in task_params:
        assert param.requires_grad
