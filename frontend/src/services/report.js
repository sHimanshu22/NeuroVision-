const PAGE_HEIGHT = 280;
const LEFT = 18;
const WIDTH = 174;

const display = (value, fallback = "Unavailable") => {
  if (value === undefined || value === null || value === "") return fallback;
  return String(value);
};
const percentage = (value) => typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "Unavailable";

export const getSegmentationWarning = (segmentation) => {
  if (!segmentation) return "";
  const area = segmentation.processed_size ** 2;
  if (!area || segmentation.foreground_pixels / area >= 0.01) return "";
  return "The prediction is very small (under 1% of the processed image area), so the mask may appear nearly black. Treat this output as inconclusive.";
};

export async function downloadAnalysisReport(state = {}) {
  const { jsPDF } = await import("jspdf");
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const generatedAt = new Date();
  let y = 20;

  const ensureSpace = (height = 8) => {
    if (y + height <= PAGE_HEIGHT) return;
    doc.addPage();
    y = 20;
  };

  const paragraph = (value, options = {}) => {
    const lines = doc.splitTextToSize(display(value), options.width || WIDTH);
    ensureSpace(lines.length * (options.lineHeight || 5) + 2);
    doc.setFont(options.font || "helvetica", options.style || "normal");
    doc.setFontSize(options.size || 10);
    doc.setTextColor(...(options.color || [55, 65, 81]));
    doc.text(lines, options.x || LEFT, y);
    y += lines.length * (options.lineHeight || 5);
  };

  const heading = (value) => {
    ensureSpace(14);
    y += 5;
    doc.setFont("helvetica", "bold");
    doc.setFontSize(13);
    doc.setTextColor(35, 82, 155);
    doc.text(value, LEFT, y);
    y += 7;
  };

  const row = (label, value) => {
    const lines = doc.splitTextToSize(display(value), WIDTH - 43);
    ensureSpace(Math.max(7, lines.length * 5) + 2);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(9.5);
    doc.setTextColor(75, 85, 99);
    doc.text(label, LEFT, y);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(31, 41, 55);
    doc.text(lines, LEFT + 43, y);
    y += Math.max(5, lines.length * 5) + 2;
  };

  doc.setFont("helvetica", "bold");
  doc.setFontSize(22);
  doc.setTextColor(35, 82, 155);
  doc.text("NeuroVision", LEFT, y);
  y += 8;
  doc.setFont("helvetica", "normal");
  doc.setFontSize(11);
  doc.setTextColor(75, 85, 99);
  doc.text("Downloadable analysis report", LEFT, y);
  y += 8;
  row("Generated", generatedAt.toLocaleString());
  row("Input image", state.file?.name || "Unavailable");

  heading("Classification");
  if (!state.classification) {
    paragraph("Not performed.");
  } else {
    row("Status", "Completed");
    row("Predicted class", state.classification.predicted_class);
    row("Confidence", percentage(state.classification.confidence));
    const probabilities = state.classification.class_probabilities;
    if (probabilities && Object.keys(probabilities).length) {
      row("Class probabilities", "Listed below");
      Object.entries(probabilities).forEach(([name, probability]) => {
        row(`  ${name}`, percentage(probability));
      });
    } else {
      row("Class probabilities", "Unavailable");
    }
  }

  heading("Segmentation");
  if (!state.segmentation) {
    paragraph("Not performed.");
  } else {
    const segmentation = state.segmentation;
    row("Status", "Completed");
    row("Foreground pixels", segmentation.foreground_pixels);
    row("Processed dimensions", `${display(segmentation.processed_size)} × ${display(segmentation.processed_size)}`);
    row("Threshold", segmentation.threshold);
    const warning = getSegmentationWarning(segmentation);
    if (warning) {
      paragraph(`Warning: ${warning}`, { color: [148, 101, 27] });
    }
  }

  heading("Grad-CAM");
  if (!state.gradcam) {
    paragraph("Not performed.");
  } else {
    row("Status", "Completed");
    row("Predicted class", state.gradcam.predicted_class);
    row("Confidence", percentage(state.gradcam.confidence));
    paragraph("Grad-CAM highlights image regions that influenced the classifier prediction.");
  }

  heading("Limitations and disclaimer");
  paragraph("NeuroVision is an educational/research prototype. Model outputs may be incorrect and are not medical diagnoses. Segmentation results are thresholded model predictions, not ground-truth or clinical boundaries. Grad-CAM highlights model attention and does not establish causation or diagnosis.");

  const pageCount = doc.getNumberOfPages();
  for (let page = 1; page <= pageCount; page += 1) {
    doc.setPage(page);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(8);
    doc.setTextColor(107, 114, 128);
    doc.text(`NeuroVision · Research prototype · Page ${page} of ${pageCount}`, LEFT, 289);
  }

  const date = generatedAt.toISOString().slice(0, 10);
  doc.save(`neurovision-analysis-report-${date}.pdf`);
}
