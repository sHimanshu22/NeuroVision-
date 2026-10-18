import { useRef, useState } from "react";
import axios from "axios";
import {
  Activity,
  Brain,
  CheckCircle2,
  CircleHelp,
  Download,
  FileImage,
  ImagePlus,
  LoaderCircle,
  ScanLine,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

function App() {
  const fileInputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [mask, setMask] = useState("");
  const [overlay, setOverlay] = useState("");
  const [groundTruth, setGroundTruth] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);

  function chooseFile(selectedFile) {
    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    if (selectedFile.size > MAX_UPLOAD_BYTES) {
      setError("Please choose an image smaller than 10 MB.");
      return;
    }

    if (preview) URL.revokeObjectURL(preview);

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setMask("");
    setOverlay("");
    setResult(null);
    setError("");
    setGroundTruth("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function clearFile() {
    if (preview) URL.revokeObjectURL(preview);

    setFile(null);
    setPreview("");
    setMask("");
    setOverlay("");
    setResult(null);
    setError("");
    setGroundTruth("");

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function createOverlay(originalFile, maskDataUrl) {
    const [originalImage, maskImage] = await Promise.all([
      createImageBitmap(originalFile),
      new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () =>
          reject(new Error("Could not read the predicted mask."));
        image.src = maskDataUrl;
      }),
    ]);

    const canvas = document.createElement("canvas");
    canvas.width = originalImage.width;
    canvas.height = originalImage.height;

    const context = canvas.getContext("2d");
    if (!context) {
      originalImage.close();
      throw new Error("Your browser could not create the overlay preview.");
    }

    // Draw the original MRI first, at its original resolution.
    context.drawImage(originalImage, 0, 0, canvas.width, canvas.height);

    // Resize the 256 × 256 mask to match the original MRI dimensions.
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = canvas.width;
    maskCanvas.height = canvas.height;
    const maskContext = maskCanvas.getContext("2d");

    if (!maskContext) {
      originalImage.close();
      throw new Error("Your browser could not process the mask.");
    }

    maskContext.imageSmoothingEnabled = false;
    maskContext.drawImage(maskImage, 0, 0, canvas.width, canvas.height);

    const maskPixels = maskContext.getImageData(
      0,
      0,
      canvas.width,
      canvas.height,
    );
    const overlayPixels = context.getImageData(
      0,
      0,
      canvas.width,
      canvas.height,
    );

    // Tint predicted foreground pixels green while retaining MRI detail.
    for (let i = 0; i < maskPixels.data.length; i += 4) {
      if (maskPixels.data[i] > 127) {
        const alpha = 0.48;
        overlayPixels.data[i] =
          overlayPixels.data[i] * (1 - alpha) + 35 * alpha;
        overlayPixels.data[i + 1] =
          overlayPixels.data[i + 1] * (1 - alpha) + 230 * alpha;
        overlayPixels.data[i + 2] =
          overlayPixels.data[i + 2] * (1 - alpha) + 150 * alpha;
      }
    }

    context.putImageData(overlayPixels, 0, 0);
    originalImage.close();

    return canvas.toDataURL("image/png");
  }

  async function runSegmentation() {
    if (!file) {
      setError("Please upload an MRI image first.");
      return;
    }

    setLoading(true);
    setError("");
    setMask("");
    setOverlay("");
    setResult(null);
    setGroundTruth("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(`${API_URL}/api/segment`, formData, {
        timeout: 120000,
      });

      const data = response.data;
      const maskDataUrl = `data:image/png;base64,${data.mask_png_base64}`;

      setMask(maskDataUrl);
      setResult(data);

      if (file.name === "brisc2025_test_00004_gl_ax_t1.jpg") {
        try {
          const referenceResponse = await axios.get(
            `${API_URL}/api/reference-mask`,
            {
              params: { filename: file.name },
              timeout: 30000,
            },
          );

          setGroundTruth(
            `data:image/png;base64,${referenceResponse.data.mask_png_base64}`,
          );
        } catch (referenceError) {
          console.error("Reference mask could not be loaded:", referenceError);
          setError(
            "Segmentation succeeded, but the reference mask could not be loaded.",
          );
        }
      }

      try {
        const overlayDataUrl = await createOverlay(file, maskDataUrl);
        setOverlay(overlayDataUrl);
      } catch (overlayError) {
        console.error("Overlay preview could not be created:", overlayError);
        setError(
          "Segmentation succeeded, but the overlay preview could not be created in this browser.",
        );
      }
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Could not connect to NeuroVision. Check that the backend is running.",
      );
    } finally {
      setLoading(false);
    }
  }

  function downloadMask() {
    if (!mask) return;

    const link = document.createElement("a");
    link.href = mask;
    link.download = "neurovision-segmentation-mask.png";
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  function downloadOverlay() {
    if (!overlay) return;

    const link = document.createElement("a");
    link.href = overlay;
    link.download = "neurovision-mri-overlay.png";
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="NeuroVision home">
          <span className="brand-mark">
            <Brain size={25} />
          </span>
          <span>
            <strong>NeuroVision</strong>
            <small>AI RESEARCH PLATFORM</small>
          </span>
        </a>

        <div className="topbar-right">
          <span className="status-pill">
            <span className="status-dot" />
            Local AI engine
          </span>
          <span className="version-label">Research v1.0</span>
        </div>
      </header>

      <main className="main-content">
        <section className="hero-section">
          <div className="eyebrow">
            <Activity size={15} />
            BRAIN MRI SEGMENTATION
          </div>

          <h1>
            See beyond the
            <span> image.</span>
          </h1>

          <p className="hero-description">
            Upload a brain MRI scan and explore the segmentation generated by
            your trained U-Net research model.
          </p>

          <div className="feature-row">
            <span>
              <ScanLine size={16} /> U-Net segmentation
            </span>
            <span>
              <ShieldCheck size={16} /> Local processing
            </span>
            <span>
              <FileImage size={16} /> PNG mask export
            </span>
          </div>
        </section>

        <section className="workspace">
          <div className="section-heading">
            <div>
              <h2>Segmentation workspace</h2>
              <p>Upload an image to begin your analysis.</p>
            </div>
            <span className="resolution-tag">INPUT 256 × 256</span>
          </div>

          <div
            className={`upload-zone ${dragging ? "dragging" : ""} ${
              file ? "has-file" : ""
            }`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              chooseFile(event.dataTransfer.files?.[0]);
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(event) => chooseFile(event.target.files?.[0])}
            />

            {!file ? (
              <>
                <div className="upload-icon">
                  <ImagePlus size={27} />
                </div>
                <h3>Upload a brain MRI image</h3>
                <p>Drag and drop your image here, or browse your computer.</p>
                <button
                  className="button button-secondary"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload size={17} /> Choose MRI image
                </button>
                <span className="file-hint">Image files · Maximum 10 MB</span>
              </>
            ) : (
              <div className="selected-file">
                <div className="file-icon">
                  <FileImage size={24} />
                </div>
                <div className="file-details">
                  <strong>{file.name}</strong>
                  <span>
                    {(file.size / 1024).toFixed(1)} KB · Ready for processing
                  </span>
                </div>
                <button
                  className="icon-button"
                  onClick={clearFile}
                  aria-label="Remove selected image"
                  title="Remove image"
                >
                  <X size={19} />
                </button>
              </div>
            )}
          </div>

          {error && (
            <div className="error-message" role="alert">
              <CircleHelp size={18} /> {error}
            </div>
          )}

          <div className="action-row">
            <span className="action-note">
              {file ? "Image selected and ready." : "No image selected yet."}
            </span>
            <button
              className="button button-primary"
              onClick={runSegmentation}
              disabled={!file || loading}
            >
              {loading ? (
                <>
                  <LoaderCircle className="spin" size={18} /> Processing MRI…
                </>
              ) : (
                <>
                  <ScanLine size={18} /> Run segmentation
                </>
              )}
            </button>
          </div>

          <section className="results-section">
            <div className="results-heading">
              <div>
                <h2>Results</h2>
                <p>Compare the MRI, predicted mask, and overlay preview.</p>
              </div>
              {result && (
                <span className="success-pill">
                  <CheckCircle2 size={15} /> Processing complete
                </span>
              )}
            </div>

            <div className="image-grid">
              <article className="image-card">
                <div className="image-card-heading">
                  <div>
                    <span className="panel-number">01</span>
                    <strong>Original MRI</strong>
                  </div>
                  <span className="panel-tag">INPUT</span>
                </div>

                <div className="image-view">
                  {preview ? (
                    <img src={preview} alt="Uploaded MRI scan" />
                  ) : (
                    <div className="empty-state">
                      <FileImage size={31} />
                      <span>Your MRI preview will appear here</span>
                    </div>
                  )}
                </div>

                {result && (
                  <div className="image-card-footer">
                    <span>Original dimensions</span>
                    <strong>
                      {result.original_width} × {result.original_height}
                    </strong>
                  </div>
                )}
              </article>

              <article className="image-card">
                <div className="image-card-heading">
                  <div>
                    <span className="panel-number">02</span>
                    <strong>Predicted mask</strong>
                  </div>
                  <span className="panel-tag mask-tag">U-NET</span>
                </div>

                <div className="image-view mask-view">
                  {mask ? (
                    <img src={mask} alt="Predicted binary segmentation mask" />
                  ) : (
                    <div className="empty-state">
                      <ScanLine size={31} />
                      <span>Segmentation mask will appear here</span>
                    </div>
                  )}
                </div>

                {result && (
                  <div className="image-card-footer">
                    <span>Foreground pixels</span>
                    <strong>{result.foreground_pixels.toLocaleString()}</strong>
                  </div>
                )}
              </article>

              <article className="image-card">
                <div className="image-card-heading">
                  <div>
                    <span className="panel-number">03</span>
                    <strong>MRI overlay</strong>
                  </div>
                  <span className="panel-tag overlay-tag">PREVIEW</span>
                </div>

                <div className="image-view overlay-view">
                  {overlay ? (
                    <img
                      src={overlay}
                      alt="Original MRI with predicted foreground highlighted in green"
                    />
                  ) : (
                    <div className="empty-state">
                      <ScanLine size={31} />
                      <span>Overlay will appear after segmentation</span>
                    </div>
                  )}
                </div>

                <div className="image-card-footer">
                  <span>Green = predicted foreground</span>
                  <strong>Threshold 0.25</strong>
                </div>
              </article>

              <article className="image-card">
                <div className="image-card-heading">
                  <div>
                    <span className="panel-number">04</span>
                    <strong>Ground-truth mask</strong>
                  </div>
                  <span className="panel-tag">REFERENCE</span>
                </div>

                <div className="image-view mask-view">
                  {groundTruth ? (
                    <img
                      src={groundTruth}
                      alt="Ground-truth segmentation mask"
                    />
                  ) : (
                    <div className="empty-state">
                      <FileImage size={31} />
                      <span>
                        Reference available for the configured test example only
                      </span>
                    </div>
                  )}
                </div>

                <div className="image-card-footer">
                  <span>Reference mask</span>
                  <strong>Threshold 128</strong>
                </div>
              </article>
            </div>

            {result && (
              <div className="result-summary">
                <div className="summary-icon">
                  <CheckCircle2 size={21} />
                </div>
                <div className="summary-copy">
                  <strong>Segmentation generated</strong>
                  <span>
                    Threshold: {result.threshold} · Model input:{" "}
                    {result.processed_size} × {result.processed_size}
                    {" · "}
                    {(
                      (result.foreground_pixels /
                        (result.processed_size * result.processed_size)) *
                      100
                    ).toFixed(3)}
                    % foreground
                  </span>
                </div>
                <button
                  className="button button-secondary"
                  onClick={downloadMask}
                >
                  <Download size={17} /> Download mask
                </button>
                <button
                  className="button button-secondary"
                  onClick={downloadOverlay}
                  disabled={!overlay}
                >
                  <Download size={17} /> Download overlay
                </button>
              </div>
            )}
          </section>
        </section>

        <aside className="notice">
          <ShieldCheck size={20} />
          <div>
            <strong>Research use only</strong>
            <p>
              This prototype displays model-generated segmentation, not a
              medical diagnosis. Results may be inaccurate and must not be used
              to make clinical decisions.
            </p>
          </div>
        </aside>

        <footer className="footer">
          <span>NeuroVision · AI-Powered Research Prototype</span>
          <span>React · FastAPI · PyTorch</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
