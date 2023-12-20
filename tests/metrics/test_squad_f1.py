import logging

from pie_modules.annotations import ExtractiveAnswer, Question
from pie_modules.documents import TextDocumentWithQuestionsAndExtractiveAnswers
from pie_modules.metrics import SQuADF1


def test_squad_f1_exact_match(caplog):
    metric = SQuADF1(show_as_markdown=True)

    # create a test document
    # sample edit
    doc = TextDocumentWithQuestionsAndExtractiveAnswers(text="This is a test document.")
    # add a question
    q1 = Question(text="What is this?")
    doc.questions.append(q1)
    # add a gold answer
    doc.answers.append(ExtractiveAnswer(question=q1, start=8, end=23))
    assert str(doc.answers[0]) == "a test document"
    # add a predicted answer
    doc.answers.predictions.append(ExtractiveAnswer(question=q1, start=8, end=23, score=0.9))
    assert str(doc.answers.predictions[0]) == str(doc.answers[0])

    metric._update(doc)

    # assert internal state
    assert metric.exact_scores == {"text=This is a test document.,question=What is this?": 1}
    assert metric.f1_scores == {"text=This is a test document.,question=What is this?": 1.0}
    assert metric.has_answer_qids == ["text=This is a test document.,question=What is this?"]
    assert metric.no_answer_qids == []
    assert metric.qas_id_to_has_answer == {
        "text=This is a test document.,question=What is this?": True
    }

    caplog.clear()
    with caplog.at_level(logging.INFO):
        metric_values = metric._compute()
    assert len(caplog.records) == 1
    assert str(caplog.records[0].message) == (
        "\n"
        "|              |   0 |\n"
        "|:-------------|----:|\n"
        "| exact        | 100 |\n"
        "| f1           | 100 |\n"
        "| total        |   1 |\n"
        "| HasAns_exact | 100 |\n"
        "| HasAns_f1    | 100 |\n"
        "| HasAns_total |   1 |"
    )

    assert metric_values == {
        "HasAns_exact": 100.0,
        "HasAns_f1": 100.0,
        "HasAns_total": 1,
        "exact": 100.0,
        "f1": 100.0,
        "total": 1,
    }


def test_squad_f1_exact_match_added_article():
    metric = SQuADF1()

    # create a test document
    doc = TextDocumentWithQuestionsAndExtractiveAnswers(
        text="This is a test document.", id="eqa_doc_with_exact_match_added_article"
    )
    # add a question
    q1 = Question(text="What is this?")
    doc.questions.append(q1)
    # add a gold answer for q1
    doc.answers.append(ExtractiveAnswer(question=q1, start=8, end=23))
    assert str(doc.answers[0]) == "a test document"
    # add a predicted answer for q1
    doc.answers.predictions.append(ExtractiveAnswer(question=q1, start=10, end=23, score=0.9))
    assert str(doc.answers.predictions[0]) == "test document"
    # the spans are not the same!
    assert str(doc.answers.predictions[0]) != str(doc.answers[0])

    metric._update(doc)
    # assert internal state
    assert metric.exact_scores == {"eqa_doc_with_exact_match_added_article_0": 1}
    assert metric.f1_scores == {"eqa_doc_with_exact_match_added_article_0": 1.0}
    assert metric.has_answer_qids == ["eqa_doc_with_exact_match_added_article_0"]
    assert metric.no_answer_qids == []
    assert metric.qas_id_to_has_answer == {"eqa_doc_with_exact_match_added_article_0": True}

    metric_values = metric._compute()
    assert metric_values == {
        "HasAns_exact": 100.0,
        "HasAns_f1": 100.0,
        "HasAns_total": 1,
        "exact": 100.0,
        "f1": 100.0,
        "total": 1,
    }


def test_squad_f1_partly_span_mismatch():
    metric = SQuADF1()

    # create a test document
    doc = TextDocumentWithQuestionsAndExtractiveAnswers(
        text="This is a test document.", id="eqa_doc_with_partly_span_mismatch"
    )
    # add a question
    q1 = Question(text="What is this?")
    doc.questions.append(q1)
    # add a gold answer for q1
    doc.answers.append(ExtractiveAnswer(question=q1, start=8, end=23))
    assert str(doc.answers[0]) == "a test document"
    # add a predicted answer for q1
    doc.answers.predictions.append(ExtractiveAnswer(question=q1, start=15, end=23, score=0.9))
    assert str(doc.answers.predictions[0]) == "document"
    # the spans are not the same!
    assert str(doc.answers.predictions[0]) != str(doc.answers[0])

    metric._update(doc)
    # assert internal state
    assert metric.exact_scores == {"eqa_doc_with_partly_span_mismatch_0": 0}
    assert metric.f1_scores == {"eqa_doc_with_partly_span_mismatch_0": 0.6666666666666666}
    assert metric.has_answer_qids == ["eqa_doc_with_partly_span_mismatch_0"]
    assert metric.no_answer_qids == []
    assert metric.qas_id_to_has_answer == {"eqa_doc_with_partly_span_mismatch_0": True}

    metric_values = metric._compute()
    assert metric_values == {
        "HasAns_exact": 0.0,
        "HasAns_f1": 66.66666666666666,
        "HasAns_total": 1,
        "exact": 0.0,
        "f1": 66.66666666666666,
        "total": 1,
    }


def test_squad_f1_full_span_mismatch():
    metric = SQuADF1()

    # create a test document
    doc = TextDocumentWithQuestionsAndExtractiveAnswers(
        text="This is a test document.", id="eqa_doc_with_full_span_mismatch"
    )
    # add a question
    q1 = Question(text="What is this?")
    doc.questions.append(q1)
    # add a gold answer for q1
    doc.answers.append(ExtractiveAnswer(question=q1, start=8, end=23))
    assert str(doc.answers[0]) == "a test document"
    # add a predicted answer for q1
    doc.answers.predictions.append(ExtractiveAnswer(question=q1, start=0, end=4, score=0.9))
    assert str(doc.answers.predictions[0]) == "This"
    # the spans are not the same!
    assert str(doc.answers.predictions[0]) != str(doc.answers[0])

    metric._update(doc)
    # assert internal state
    assert metric.exact_scores == {"eqa_doc_with_full_span_mismatch_0": 0}
    assert metric.f1_scores == {"eqa_doc_with_full_span_mismatch_0": 0.0}
    assert metric.has_answer_qids == ["eqa_doc_with_full_span_mismatch_0"]
    assert metric.no_answer_qids == []
    assert metric.qas_id_to_has_answer == {"eqa_doc_with_full_span_mismatch_0": True}

    metric_values = metric._compute()
    assert metric_values == {
        "HasAns_exact": 0.0,
        "HasAns_f1": 0.0,
        "HasAns_total": 1,
        "exact": 0.0,
        "f1": 0.0,
        "total": 1,
    }


def test_squad_f1_no_predicted_answers():
    metric = SQuADF1()

    # create a test document
    doc = TextDocumentWithQuestionsAndExtractiveAnswers(
        text="This is a test document.", id="eqa_doc_without_predicted_answers"
    )
    # add a question
    q1 = Question(text="What is this?")
    doc.questions.append(q1)
    # add a gold answer for q1
    doc.answers.append(ExtractiveAnswer(question=q1, start=8, end=23))
    assert str(doc.answers[0]) == "a test document"

    metric._update(doc)
    # assert internal state
    assert metric.exact_scores == {"eqa_doc_without_predicted_answers_0": 0}
    assert metric.f1_scores == {"eqa_doc_without_predicted_answers_0": 0}
    assert metric.has_answer_qids == ["eqa_doc_without_predicted_answers_0"]
    assert metric.no_answer_qids == []
    assert metric.qas_id_to_has_answer == {"eqa_doc_without_predicted_answers_0": True}

    metric_values = metric._compute()
    assert metric_values == {
        "HasAns_exact": 0.0,
        "HasAns_f1": 0.0,
        "HasAns_total": 1,
        "exact": 0.0,
        "f1": 0.0,
        "total": 1,
    }


def test_squad_f1_no_gold_answers():
    metric = SQuADF1()

    # create a test document
    doc = TextDocumentWithQuestionsAndExtractiveAnswers(
        text="This is a test document.", id="eqa_doc_without_gold_answers"
    )
    # add a question
    q1 = Question(text="What is this?")
    doc.questions.append(q1)
    # add a predicted answer for q1
    doc.answers.predictions.append(ExtractiveAnswer(question=q1, start=8, end=23, score=0.9))
    assert str(doc.answers.predictions[0]) == "a test document"

    metric._update(doc)
    # assert internal state
    assert metric.exact_scores == {"eqa_doc_without_gold_answers_0": 0}
    assert metric.f1_scores == {"eqa_doc_without_gold_answers_0": 0}
    assert metric.has_answer_qids == []
    assert metric.no_answer_qids == ["eqa_doc_without_gold_answers_0"]
    assert metric.qas_id_to_has_answer == {"eqa_doc_without_gold_answers_0": False}

    metric_values = metric._compute()
    assert metric_values == {
        "NoAns_exact": 0.0,
        "NoAns_f1": 0.0,
        "NoAns_total": 1,
        "exact": 0.0,
        "f1": 0.0,
        "total": 1,
    }


def test_squad_f1_empty_document():
    metric = SQuADF1()

    # create a test document
    doc = TextDocumentWithQuestionsAndExtractiveAnswers(text="", id="eqa_doc_with_empty_text")
    # add a question
    q1 = Question(text="What is this?")
    doc.questions.append(q1)

    metric._update(doc)

    values = metric._compute()
    assert values == {
        "NoAns_exact": 100.0,
        "NoAns_f1": 100.0,
        "NoAns_total": 1,
        "exact": 100.0,
        "f1": 100.0,
        "total": 1,
    }
