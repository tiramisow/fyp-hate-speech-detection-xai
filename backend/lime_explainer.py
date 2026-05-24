import numpy as np
from lime.lime_text import LimeTextExplainer

LIME_NUM_SAMPLES = 300
LIME_NUM_FEATURES = 10
MIN_POSITIVE_WEIGHT = 0.01

NO_TOXICITY_EXPLANATION = (
    "No XAI explanation is generated because no toxic label was predicted."
)

STOPWORDS = {
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "am",
    "is",
    "are",
    "was",
    "were",
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
}

LABEL_CLASSIFICATION_PHRASE = {
    "toxic": "toxic",
    "severe_toxic": "severely toxic",
    "obscene": "obscene",
    "threat": "threatening",
    "insult": "insulting",
    "identity_hate": "identity-hate related",
}


def select_label_to_explain(
    predicted_labels: list[str],
    probabilities: dict[str, float],
) -> str:
    return max(predicted_labels, key=lambda label: probabilities[label])


def _effect_from_weight(weight: float) -> str:
    if weight > 0:
        return "supports"
    if weight < 0:
        return "opposes"
    return "neutral"


def _normalize_word(word: str) -> str:
    return word.strip().lower().strip("'\"")


def _is_stopword(word: str) -> bool:
    return _normalize_word(word) in STOPWORDS


def _meaningful_supporting_words(
    words: list[str],
    weights: list[float],
    effects: list[str],
    *,
    max_words: int = 3,
    exclude_stopwords: bool = True,
) -> list[str]:
    candidates: list[tuple[str, float]] = []

    for word, weight, effect in zip(words, weights, effects):
        if effect != "supports" or weight <= MIN_POSITIVE_WEIGHT:
            continue
        cleaned = word.strip()
        if exclude_stopwords and _is_stopword(cleaned):
            continue
        candidates.append((cleaned, weight))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [word for word, _ in candidates[:max_words]]


def _format_words_phrase(words: list[str]) -> str:
    if not words:
        return ""
    if len(words) == 1:
        return f"words such as '{words[0]}'"
    if len(words) == 2:
        return f"words such as '{words[0]}' and '{words[1]}'"
    joined = ", ".join(f"'{word}'" for word in words[:-1])
    return f"words such as {joined}, and '{words[-1]}'"


def _classification_phrase(label: str) -> str:
    return LABEL_CLASSIFICATION_PHRASE.get(label, label.replace("_", " "))


def _build_explanation_sentence(
    label: str,
    words: list[str],
    weights: list[float],
    effects: list[str],
) -> str:
    display_label = label.replace("_", " ")
    classification_phrase = _classification_phrase(label)

    meaningful_words = _meaningful_supporting_words(
        words, weights, effects, exclude_stopwords=True
    )
    if not meaningful_words:
        meaningful_words = _meaningful_supporting_words(
            words, weights, effects, exclude_stopwords=False
        )

    if meaningful_words:
        word_phrase = _format_words_phrase(meaningful_words)
        return (
            f"LIME indicates that {word_phrase} increased the {display_label} prediction, "
            f"which explains why the model classified the comment as {classification_phrase}."
        )

    opposers = [
        word.strip()
        for word, effect in zip(words, effects)
        if effect == "opposes" and not _is_stopword(word)
    ]
    if opposers:
        word_phrase = _format_words_phrase(opposers[:3])
        return (
            f"LIME indicates that {word_phrase} decreased the {display_label} prediction, "
            f"although the model still classified the comment as {classification_phrase}."
        )

    return (
        f"LIME indicates that no distinctive words strongly increased the {display_label} "
        f"prediction, although the model classified the comment as {classification_phrase}."
    )


def explain_prediction(
    text: str,
    label: str,
    predict_proba_fn,
) -> dict:
    explainer = LimeTextExplainer(class_names=["not_label", "positive"])

    explanation = explainer.explain_instance(
        text,
        predict_proba_fn,
        labels=[1],
        num_features=LIME_NUM_FEATURES,
        num_samples=LIME_NUM_SAMPLES,
    )

    feature_weights = explanation.as_list(label=1)
    important_words = [word.strip() for word, _ in feature_weights]
    weights = [float(weight) for _, weight in feature_weights]
    effects = [_effect_from_weight(weight) for weight in weights]

    return {
        "explained_label": label,
        "important_words": important_words,
        "weights": weights,
        "effects": effects,
        "explanation": _build_explanation_sentence(label, important_words, weights, effects),
    }
