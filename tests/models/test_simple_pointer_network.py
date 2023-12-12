import torch
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

ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)
MODEL_NAME_OR_PATH = "sshleifer/distilbart-xsum-12-1"


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


def test_bart_pointer_network_generate():
    model_name_or_path = MODEL_NAME_OR_PATH
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = BartAsPointerNetwork.from_pretrained(
        model_name_or_path,
        label_ids=[2, 3, 4, 5, 6],
        target_token_ids=[0, 2, 50266, 50269, 50268, 50265, 50267],
        eos_id=1,
        pad_id=1,
    )
    input_text = ARTICLE_TO_SUMMARIZE
    inputs = tokenizer([input_text], max_length=1024, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], num_beams=3, min_length=5, max_length=20)
    result = tokenizer.batch_decode(
        summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    assert result == [" power lines in California have been shut down on Friday."]


def test_bart_pointer_network_beam_search():
    # tokenizer = AutoTokenizer.from_pretrained("t5-base")
    # model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

    model_name_or_path = MODEL_NAME_OR_PATH  # "sshleifer/distilbart-xsum-12-1"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = BartAsPointerNetwork.from_pretrained(
        model_name_or_path,
        label_ids=[2, 3, 4, 5, 6],
        target_token_ids=[0, 2, 50266, 50269, 50268, 50265, 50267],
        eos_id=1,
        pad_id=1,
    )

    encoder_input_str = ARTICLE_TO_SUMMARIZE  # "translate English to German: How old are you?"
    encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

    # lets run beam search using 3 beams
    num_beams = 3
    # define decoder start token ids
    input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
    input_ids = input_ids * model.config.decoder_start_token_id

    # add encoder_outputs to model keyword arguments
    encoder = model.get_encoder()
    encoder_input = encoder_input_ids.repeat_interleave(num_beams, dim=0)
    model_kwargs = {
        "encoder_outputs": encoder(encoder_input, return_dict=True),
        "encoder_input_ids": encoder_input_ids,
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
        input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs
    )

    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    assert result == [
        " power lines in California have been shut down after a power provider said it was due to high winds."
    ]
