import { useState } from "react";
import "./App.css";

const LABELS = [
  "toxic",
  "severe_toxic",
  "obscene",
  "threat",
  "insult",
  "identity_hate",
];

const LABEL_DISPLAY = {
  toxic: "Toxic",
  severe_toxic: "Severe Toxic",
  obscene: "Obscene",
  threat: "Threat",
  insult: "Insult",
  identity_hate: "Identity Hate",
};

const NO_TOXICITY_MESSAGE =
  "No toxicity label was detected. All model confidence scores are below the tuned decision thresholds.";

const NO_XAI_MESSAGE =
  "No XAI explanation is generated because no toxic label was predicted.";

const API_URL = import.meta.env.VITE_API_URL ?? "/api/predict";

function formatLabel(label) {
  return LABEL_DISPLAY[label] ?? label.replace(/_/g, " ");
}

function formatPercent(value) {
  const percent = value * 100;
  if (percent < 0.1 && percent > 0) {
    return `${percent.toFixed(2)}%`;
  }
  return `${percent.toFixed(1)}%`;
}

function formatWeight(value) {
  return value >= 0 ? `+${value.toFixed(4)}` : value.toFixed(4);
}

function effectClass(effect) {
  if (effect === "supports") return "effect-supports";
  if (effect === "opposes") return "effect-opposes";
  return "effect-neutral";
}

export default function App() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const hasPredictions = (result?.predicted_labels?.length ?? 0) > 0;

  async function handleDetect() {
    const trimmed = text.trim();
    if (!trimmed) {
      setError("Please enter some text to analyze.");
      setResult(null);
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed }),
      });

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail ?? `Request failed (${response.status})`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err.message || "Failed to connect to the backend. Is it running on port 8000?",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Bullying and Hate Speech Detection with XAI</h1>
        <p>
          BERT-based multi-label bullying and hate speech classification with
          post-hoc LIME explanations.
        </p>
      </header>

      <section className="card input-section">
        <label htmlFor="comment-text">Input text</label>
        <textarea
          id="comment-text"
          className="text-input"
          placeholder="Enter a comment to classify..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={loading}
        />
        <button
          type="button"
          className="detect-btn"
          onClick={handleDetect}
          disabled={loading}
        >
          {loading ? "Analyzing…" : "Detect Toxicity"}
        </button>
      </section>

      {loading && (
        <div className="loading-box" role="status" aria-live="polite">
          <div className="spinner" aria-hidden="true" />
          <div>
            <p>
              <strong>Running BERT classification…</strong>
            </p>
            <p>
              If a toxic label is predicted, LIME explanation will follow (may take
              30–90 seconds).
            </p>
          </div>
        </div>
      )}

      {error && <div className="error-box" role="alert">{error}</div>}

      {result && !loading && (
        <>
          <section className="card">
            <h2 className="section-title">Prediction Result</h2>
            {hasPredictions ? (
              <div className="badges">
                {result.predicted_labels.map((label) => (
                  <span key={label} className="badge">
                    {formatLabel(label)}
                  </span>
                ))}
              </div>
            ) : (
              <p className="result-message">{NO_TOXICITY_MESSAGE}</p>
            )}
          </section>

          <section className="card">
            <h2 className="section-title">Label Probabilities</h2>
            <div className="prob-list">
              {LABELS.map((label) => {
                const score = result.probabilities?.[label] ?? 0;
                const threshold = result.thresholds?.[label] ?? 0;
                const isActive = result.predictions?.[label] === 1;
                return (
                  <div key={label} className="prob-row">
                    <div className="prob-row-header">
                      <span className="prob-label">{formatLabel(label)}</span>
                    </div>
                    <div className="prob-bar-track">
                      <div
                        className={`prob-bar-fill${isActive ? " active" : ""}`}
                        style={{ width: `${Math.min(score * 100, 100)}%` }}
                      />
                    </div>
                    <p className="prob-meta">
                      Probability: {formatPercent(score)}, Threshold:{" "}
                      {formatPercent(threshold)}
                    </p>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="card">
            <h2 className="section-title">XAI Explanation</h2>
            {hasPredictions && result.explained_label && (
              <p className="explained-label">
                Explaining highest predicted label:{" "}
                <strong>{formatLabel(result.explained_label)}</strong>
              </p>
            )}
            <p className="explanation-text">
              {hasPredictions ? result.explanation : NO_XAI_MESSAGE}
            </p>
          </section>

          {hasPredictions && result.important_words?.length > 0 && (
            <section className="card">
              <h2 className="section-title">LIME Important Words</h2>
              <div className="lime-table-wrap">
                <table className="lime-table">
                  <thead>
                    <tr>
                      <th>Word</th>
                      <th>Weight</th>
                      <th>Effect</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.important_words.map((word, index) => (
                      <tr key={`${word}-${index}`}>
                        <td>{word}</td>
                        <td>{formatWeight(result.weights[index])}</td>
                        <td className={effectClass(result.effects[index])}>
                          {result.effects[index]}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}

      <footer className="disclaimer">
        <strong>Disclaimer</strong>
        This system is a proof-of-concept and should not be used as a final moderation
        decision tool.
      </footer>
    </div>
  );
}
