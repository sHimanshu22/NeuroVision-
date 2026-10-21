import { Info } from "lucide-react";
import { getSegmentationWarning } from "../services/report";
import { api } from "../services/api";
import { useState } from "react";

const shown = (value, fallback = "Unavailable") => (
  value === undefined || value === null || value === "" ? fallback : value
);
const percent = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1 ? `${(value * 100).toFixed(2)}%` : "Unavailable";
const titleCase = (value) => String(shown(value)).replaceAll("_", " ");

function ExplanationSection({ title, children, className = "" }) {
  return <section className={`panel ${className}`} style={{ display: "grid", gap: 10 }}><h3>{title}</h3>{children}</section>;
}

function MissingResult() {
  return <p className="muted" style={{ margin: 0 }}>Not performed. Run this workflow from the MRI analysis page to see its result.</p>;
}

export function ResultsExplanation({ classification, segmentation, gradcam }) {
  const segmentationWarning = getSegmentationWarning(segmentation);
  const hasSegmentationMetadata = Boolean(segmentation && Number.isFinite(segmentation.processed_size) && Number.isFinite(segmentation.foreground_pixels));
  const [disclosureOpen, setDisclosureOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [gemini, setGemini] = useState(null);
  const hasResults = Boolean(classification || segmentation || gradcam);
  const requestGemini = async () => {
    setBusy(true);
    setError("");
    try {
      const result = await api.geminiExplanation({
        classification: classification ? {
          predicted_class: classification.predicted_class ?? null,
          confidence: classification.confidence ?? null,
          class_probabilities: classification.class_probabilities || {},
        } : null,
        segmentation: segmentation ? {
          status: "Completed",
          original_width: segmentation.original_width ?? null,
          original_height: segmentation.original_height ?? null,
          processed_size: segmentation.processed_size ?? null,
          threshold: segmentation.threshold ?? null,
          foreground_pixels: segmentation.foreground_pixels ?? null,
          sparse_warning: segmentationWarning || null,
        } : null,
        gradcam: gradcam ? {
          predicted_class: gradcam.predicted_class ?? null,
          confidence: gradcam.confidence ?? null,
        } : null,
      });
      setGemini(result);
      setDisclosureOpen(false);
    } catch (reason) {
      setError(reason.message || "The optional Gemini explanation is unavailable.");
    } finally {
      setBusy(false);
    }
  };

  return <section style={{ display: "grid", gap: 16, marginTop: 16 }} aria-label="Plain-language AI results explanation">
    <ExplanationSection title="What the AI model predicted">
      {!classification ? <MissingResult/> : <>
        <div className="metrics list">
          <div><span>Predicted class</span><b>{titleCase(classification.predicted_class)}</b></div>
          <div><span>Model confidence</span><b>{percent(classification.confidence)}</b></div>
        </div>
        <p className="muted" style={{ margin: 0 }}>A label such as “pituitary” is the AI model&apos;s predicted category. It does not confirm a pituitary tumor or any disease.</p>
        {classification.class_probabilities && Object.keys(classification.class_probabilities).length ? <div>
          <strong>Class probabilities</strong>
          <div className="metrics list" style={{ marginTop: 8 }}>
            {Object.entries(classification.class_probabilities).map(([name, probability]) => <div key={name}><span>{titleCase(name)}</span><b>{percent(probability)}</b></div>)}
          </div>
        </div> : <p className="muted" style={{ margin: 0 }}>Class probabilities: Unavailable.</p>}
        <div className="notice"><Info size={17}/><p>Confidence describes how strongly this model selected one of its trained categories. It is not the probability that a person has a disease.</p></div>
      </>}
    </ExplanationSection>

    <ExplanationSection title="What the classification result means">
      <p className="muted" style={{ margin: 0 }}>The classification model assigns the uploaded image to one of the categories it was trained to recognize. This is a computer-generated prediction, not a confirmed diagnosis.</p>
    </ExplanationSection>

    <ExplanationSection title="What the segmentation result means">
      {!segmentation ? <MissingResult/> : <>
        <p className="muted" style={{ margin: 0 }}>The mask highlights image pixels selected by the model. It does not show tumor size or location, and it cannot prove that a tumor is present or absent.</p>
        {segmentationWarning ? <div className="notice warning"><Info size={17}/><p>{segmentationWarning} This makes the result inconclusive; it is not proof that no tumor exists.</p></div> : hasSegmentationMetadata ? <p className="muted" style={{ margin: 0 }}>The available metadata does not flag this mask as sparse. This is not a diagnosis and does not rule a tumor in or out.</p> : <p className="muted" style={{ margin: 0 }}>There is not enough segmentation metadata to assess whether the mask is sparse. This result is not a diagnosis and does not rule a tumor in or out.</p>}
        <details>
          <summary className="muted">Technical details</summary>
          <div className="metrics list" style={{ marginTop: 10 }}>
            <div><span>Status</span><b>Completed</b></div>
            <div><span>Foreground pixels</span><b>{shown(segmentation.foreground_pixels)}</b></div>
            <div><span>Processed dimensions</span><b>{shown(segmentation.processed_size)} x {shown(segmentation.processed_size)}</b></div>
            <div><span>Threshold</span><b>{shown(segmentation.threshold)}</b></div>
          </div>
        </details>
      </>}
    </ExplanationSection>

    <ExplanationSection title="What the Grad-CAM heatmap means">
      {!gradcam ? <MissingResult/> : <>
        <div className="metrics list">
          <div><span>Status</span><b>Completed</b></div>
          <div><span>Predicted class</span><b>{titleCase(gradcam.predicted_class)}</b></div>
          <div><span>Model confidence</span><b>{percent(gradcam.confidence)}</b></div>
        </div>
        <p className="muted" style={{ margin: 0 }}>The heatmap highlights image regions that influenced the classification model. Highlighted regions are not automatically tumors, and Grad-CAM does not prove that the prediction is correct.</p>
      </>}
    </ExplanationSection>

    <ExplanationSection title="What to do with this information" className="follow-up-panel">
      <p className="muted" style={{ margin: 0 }}>NeuroVision is an educational research prototype, not a clinical diagnostic system. Discuss your MRI and the official radiology report with a qualified healthcare professional. Do not use these model outputs alone to make health decisions.</p>
    </ExplanationSection>

    <ExplanationSection title="Optional Gemini explanation">
      <p className="muted" style={{ margin: 0 }}>Ask Gemini to restate the available structured model results in simpler language. This is optional and does not replace the explanations above.</p>
      <button className="button secondary" disabled={!hasResults || busy} onClick={() => { setError(""); setDisclosureOpen(true); }}>
        {busy ? "Requesting explanation…" : "Explain results with Gemini"}
      </button>
      {!hasResults && <p className="muted" style={{ margin: 0 }}>Run at least one local workflow before requesting an explanation.</p>}
      {error && <div className="error" role="alert">{error}</div>}
      {disclosureOpen && <div className="panel" role="dialog" aria-modal="true" aria-labelledby="gemini-disclosure-title" style={{ display: "grid", gap: 10, padding: 16 }}>
        <h4 id="gemini-disclosure-title" style={{ margin: 0 }}>Before sending structured results</h4>
        <p className="muted" style={{ margin: 0 }}>If you continue, NeuroVision will send only the available predicted class, model confidence, class probabilities, segmentation metadata, sparse-mask warning, and Grad-CAM predicted class. It will not send the MRI image, mask, heatmap, filename, name, or patient identifiers.</p>
        <p className="muted" style={{ margin: 0 }}>This information will be sent to Google Gemini. Google&apos;s data-handling terms apply, and data sent through the free tier may be used to improve Google products. Cancel if you do not agree.</p>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button className="button primary" disabled={busy} onClick={requestGemini}>Continue and send</button>
          <button className="button secondary" disabled={busy} onClick={() => setDisclosureOpen(false)}>Cancel</button>
        </div>
      </div>}
      {gemini && <div className="panel" style={{ display: "grid", gap: 10, padding: 16 }}>
        <strong>AI-generated patient-friendly explanation</strong>
        <p className="muted" style={{ margin: 0 }}>{gemini.plain_language_summary}</p>
        <h4 style={{ margin: "6px 0 0" }}>What the AI found</h4>
        {classification && <p className="muted" style={{ margin: 0 }}>The label “{titleCase(classification.predicted_class)}” is the AI model&apos;s predicted category. It does not confirm a tumor or any disease.</p>}
        <p className="muted" style={{ margin: 0 }}>{gemini.classification_explanation}</p>
        <h4 style={{ margin: "6px 0 0" }}>How to understand the confidence</h4>
        <div className="metrics list" style={{ margin: 0 }}><div><span>Model confidence</span><b>{classification ? percent(classification.confidence) : "Not performed"}</b></div></div>
        <p className="muted" style={{ margin: 0 }}>This percentage describes how strongly the model selected its category. It is not the probability that a person has a disease.</p>
        <h4 style={{ margin: "6px 0 0" }}>What the segmentation means</h4>
        <p className="muted" style={{ margin: 0 }}>{gemini.segmentation_explanation}</p>
        <h4 style={{ margin: "6px 0 0" }}>What the highlighted regions mean</h4>
        <p className="muted" style={{ margin: 0 }}>{gemini.gradcam_explanation}</p>
        <h4 style={{ margin: "6px 0 0" }}>What to do next</h4>
        <p className="muted" style={{ margin: 0 }}>{gemini.next_steps}</p>
        <p className="muted" style={{ margin: 0 }}><b>Limitations:</b> {gemini.limitations}</p>
        {Array.isArray(gemini.educational_terms) && gemini.educational_terms.length > 0 && <p className="muted" style={{ margin: 0 }}><b>Terms:</b> {gemini.educational_terms.join(", ")}</p>}
      </div>}
    </ExplanationSection>
  </section>;
}
