"use client";

import { useState, useRef } from "react";
import { Upload, Loader2, ImageIcon, AlertCircle, CheckCircle2, X, RotateCcw, Sparkles, Cpu, Eye, Layers, ShieldCheck, AlertTriangle } from "lucide-react";
import { Section, SectionHeading } from "./Section";
import { type Breed } from "@/data/breeds";

interface HybridResult {
  primary: {
    source: "yolo" | "vlm" | "consensus" | "none";
    breed: string;
    breedId: string | null;
    confidence: number;
    yoloConfidence?: number;
    vlmConfidence?: number;
  };
  vlmResult: {
    breed: string;
    breedId: string | null;
    confidence: number;
    characteristics: string[];
    notes: string;
  };
  yoloResult: {
    available: boolean;
    primary: { class: string; classId: number; confidence: number; bbox: number[] } | null;
    allDetections: { class: string; classId: number; confidence: number; bbox: number[] }[];
    annotatedImage: string | null;
    classesAvailable: string[];
  };
  agreement: boolean;
  characteristics: string[];
  notes: string;
  breedInfo: Partial<Breed> | null;
}

export function BreedClassifier() {
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<HybridResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [yoloStatus, setYoloStatus] = useState<"unknown" | "online" | "offline">("unknown");
  const inputRef = useRef<HTMLInputElement>(null);

  // Check YOLO service status once on mount
  useState(() => {
    fetch("/api/yolo-health")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setYoloStatus(d?.modelLoaded ? "online" : "offline"))
      .catch(() => setYoloStatus("offline"));
  });

  const onFileSelect = (f: File) => {
    if (!f.type.startsWith("image/")) {
      setError("Please upload an image file (PNG, JPG, WebP).");
      return;
    }
    if (f.size > 10 * 1024 * 1024) {
      setError("Image is too large. Maximum allowed size is 10 MB.");
      return;
    }
    setError(null);
    setFile(f);
    setResult(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) onFileSelect(f);
  };

  const classify = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const formData = new FormData();
      formData.append("image", file);
      const res = await fetch("/api/classify", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Classification failed.");
      setResult(data as HybridResult);
      // Update YOLO status based on result
      setYoloStatus((data as HybridResult).yoloResult.available ? "online" : "offline");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to classify image.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <Section bg="mist">
      <SectionHeading
        eyebrow="Hybrid AI · YOLO + VLM"
        title="Identify any Indian bovine breed"
        subtitle="Your trained YOLO model (version4.pt) runs as the primary detector for fast, high-accuracy breed identification. A vision-language model then refines the result, validates the prediction, and adds observed visual characteristics — giving you the best of both worlds."
      />

      {/* Pipeline status banner */}
      <div className="mb-6 p-4 rounded-md bg-brand-blue-50 border border-brand-blue-100">
        <div className="flex items-start gap-3">
          <Layers className="h-5 w-5 text-brand-blue shrink-0 mt-0.5" />
          <div className="flex-1">
            <div
              className="text-sm font-semibold text-brand-navy mb-1.5"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              Hybrid Detection Pipeline
            </div>
            <div className="grid sm:grid-cols-2 gap-3 text-xs">
              <div className="flex items-start gap-2">
                <span
                  className={`h-2 w-2 rounded-full mt-1.5 shrink-0 ${
                    yoloStatus === "online"
                      ? "bg-brand-green"
                      : yoloStatus === "offline"
                      ? "bg-red-400"
                      : "bg-slate-300"
                  }`}
                />
                <div>
                  <div className="font-semibold text-brand-navy flex items-center gap-1.5">
                    <Cpu className="h-3.5 w-3.5" /> YOLO Model (Primary)
                  </div>
                  <div className="text-slate-600">
                    {yoloStatus === "online"
                      ? "Online · version4.pt loaded · fast detection with bbox"
                      : yoloStatus === "offline"
                      ? "Offline — drop version4.pt at python-services/yolo-detector/models/ and restart. Falling back to VLM-only."
                      : "Checking service status..."}
                  </div>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <span className="h-2 w-2 rounded-full bg-brand-green mt-1.5 shrink-0" />
                <div>
                  <div className="font-semibold text-brand-navy flex items-center gap-1.5">
                    <Eye className="h-3.5 w-3.5" /> VLM Refinement (Secondary)
                  </div>
                  <div className="text-slate-600">
                    Online · validates YOLO prediction · adds visual features & context
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-5 gap-6">
        {/* Upload area */}
        <div className="lg:col-span-3">
          <div className="card-soft p-6">
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) onFileSelect(f);
              }}
            />

            {!preview ? (
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={onDrop}
                onClick={() => inputRef.current?.click()}
                className="border-2 border-dashed border-brand-line rounded-lg p-10 md:p-14 text-center cursor-pointer hover:border-brand-blue hover:bg-white transition-colors"
              >
                <div className="h-14 w-14 mx-auto rounded-full bg-brand-blue-50 flex items-center justify-center mb-4">
                  <Upload className="h-6 w-6 text-brand-blue" />
                </div>
                <h3
                  className="text-base font-semibold text-brand-navy mb-1"
                  style={{ fontFamily: "var(--font-montserrat)" }}
                >
                  Drop an image here, or click to browse
                </h3>
                <p className="text-sm text-slate-500">
                  Supports PNG, JPG, WebP up to 10 MB. Best results with clear, well-lit side-profile photos.
                </p>
              </div>
            ) : (
              <div>
                <div className="relative rounded-lg overflow-hidden bg-slate-100 border border-brand-line">
                  <img
                    src={preview}
                    alt="Uploaded bovine"
                    className="w-full max-h-[420px] object-contain bg-slate-50"
                  />
                  <button
                    onClick={reset}
                    className="absolute top-3 right-3 h-8 w-8 rounded-full bg-black/60 backdrop-blur text-white flex items-center justify-center hover:bg-black/80"
                    aria-label="Remove image"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
                <div className="mt-4 flex items-center justify-between gap-3 flex-wrap">
                  <div className="text-xs text-slate-500 flex items-center gap-1.5">
                    <ImageIcon className="h-3.5 w-3.5" />
                    {file?.name} · {(file ? file.size / 1024 / 1024 : 0).toFixed(2)} MB
                  </div>
                  <div className="flex items-center gap-2">
                    <button onClick={reset} className="btn-secondary text-xs" disabled={loading}>
                      <RotateCcw className="h-3.5 w-3.5" /> Change
                    </button>
                    <button onClick={classify} disabled={loading} className="btn-primary">
                      {loading ? (
                        <>
                          <Loader2 className="h-4 w-4 animate-spin" /> Detecting...
                        </>
                      ) : (
                        <>
                          <Sparkles className="h-4 w-4" /> Run Hybrid Detection
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className="mt-4 flex items-start gap-2 p-3 rounded-md bg-red-50 border border-red-200 text-sm text-red-700">
                <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
                {error}
              </div>
            )}
          </div>

          {/* Tips card */}
          <div className="mt-4 card-soft p-5">
            <h4
              className="text-sm font-semibold text-brand-navy mb-2"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              How the hybrid pipeline works
            </h4>
            <ol className="text-xs text-slate-600 space-y-2">
              <li className="flex gap-2">
                <span className="numbered-badge !h-5 !w-5 !text-[10px] shrink-0 mt-0.5">1</span>
                <span>
                  <strong className="text-brand-navy">YOLO inference:</strong> Your trained version4.pt
                  model runs first, returning the breed with highest detection confidence and a
                  bounding box on the animal.
                </span>
              </li>
              <li className="flex gap-2">
                <span className="numbered-badge !h-5 !w-5 !text-[10px] shrink-0 mt-0.5">2</span>
                <span>
                  <strong className="text-brand-navy">VLM refinement:</strong> The vision-language model
                  receives the YOLO prediction as a hint, then independently verifies it against the
                  image and lists observed visual characteristics.
                </span>
              </li>
              <li className="flex gap-2">
                <span className="numbered-badge !h-5 !w-5 !text-[10px] shrink-0 mt-0.5">3</span>
                <span>
                  <strong className="text-brand-navy">Consensus scoring:</strong> If both models agree,
                  confidence is boosted. If they disagree, the higher-confidence prediction wins, and
                  both results are shown for transparency.
                </span>
              </li>
            </ol>
          </div>
        </div>

        {/* Results column */}
        <div className="lg:col-span-2">
          <div className="card-soft p-6 sticky top-24">
            <h3
              className="text-base font-semibold text-brand-navy mb-4"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              Detection Result
            </h3>

            {!result && !loading && (
              <div className="text-center py-10 text-slate-400">
                <div className="h-14 w-14 mx-auto rounded-full bg-brand-mist flex items-center justify-center mb-3">
                  <Layers className="h-6 w-6" />
                </div>
                <p className="text-sm">Hybrid result will appear here.</p>
              </div>
            )}

            {loading && (
              <div className="text-center py-10">
                <Loader2 className="h-10 w-10 mx-auto text-brand-blue animate-spin mb-3" />
                <p className="text-sm text-slate-600 font-medium">Running hybrid detection...</p>
                <div className="mt-3 text-xs text-slate-500 space-y-1">
                  <div className="flex items-center justify-center gap-1.5">
                    <Cpu className="h-3 w-3" /> YOLO inference
                  </div>
                  <div className="flex items-center justify-center gap-1.5">
                    <Eye className="h-3 w-3" /> VLM refinement
                  </div>
                  <div className="flex items-center justify-center gap-1.5">
                    <ShieldCheck className="h-3 w-3" /> Consensus scoring
                  </div>
                </div>
              </div>
            )}

            {result && !loading && (
              <div className="space-y-4">
                {/* Primary result */}
                <div
                  className={`p-4 rounded-md border ${
                    result.primary.source === "consensus"
                      ? "bg-green-50 border-green-200"
                      : result.primary.source === "yolo"
                      ? "bg-blue-50 border-blue-200"
                      : "bg-amber-50 border-amber-200"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <div className="text-[11px] uppercase tracking-wider text-slate-600 font-semibold mb-0.5 flex items-center gap-1.5">
                        <SourceBadge source={result.primary.source} agreement={result.agreement} />
                      </div>
                      <div
                        className="text-lg font-bold text-brand-navy"
                        style={{ fontFamily: "var(--font-montserrat)" }}
                      >
                        {result.primary.breed}
                      </div>
                      {result.breedInfo?.type && (
                        <div className="text-xs text-slate-600 mt-0.5">
                          {result.breedInfo.type === "buffalo" ? "Buffalo" : "Cattle"} ·{" "}
                          {result.breedInfo.category}
                        </div>
                      )}
                    </div>
                    <ConfidenceMeter value={result.primary.confidence} />
                  </div>

                  {/* Source breakdown */}
                  <div className="mt-3 pt-3 border-t border-white/40 grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold flex items-center gap-1">
                        <Cpu className="h-3 w-3" /> YOLO
                      </div>
                      <div className="font-bold text-brand-navy">
                        {result.primary.yoloConfidence != null
                          ? `${result.primary.yoloConfidence}%`
                          : "—"}
                      </div>
                      <div className="text-[10px] text-slate-500 truncate">
                        {result.yoloResult.primary?.class || "No detection"}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold flex items-center gap-1">
                        <Eye className="h-3 w-3" /> VLM
                      </div>
                      <div className="font-bold text-brand-navy">
                        {result.primary.vlmConfidence != null
                          ? `${result.primary.vlmConfidence}%`
                          : "—"}
                      </div>
                      <div className="text-[10px] text-slate-500 truncate">
                        {result.vlmResult.breed || "No identification"}
                      </div>
                    </div>
                  </div>
                </div>

                {/* YOLO annotated image */}
                {result.yoloResult.annotatedImage && (
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1.5 flex items-center gap-1.5">
                      <Cpu className="h-3 w-3" /> YOLO Detection Overlay
                    </div>
                    <div className="rounded-md overflow-hidden border border-brand-line bg-slate-100">
                      <img
                        src={result.yoloResult.annotatedImage}
                        alt="YOLO annotated"
                        className="w-full max-h-48 object-contain bg-slate-50"
                      />
                    </div>
                  </div>
                )}

                {/* All YOLO detections */}
                {result.yoloResult.allDetections.length > 1 && (
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1.5">
                      All YOLO Detections
                    </div>
                    <div className="space-y-1">
                      {result.yoloResult.allDetections.slice(0, 5).map((d, i) => (
                        <div
                          key={i}
                          className="flex items-center justify-between text-xs p-1.5 rounded bg-brand-mist"
                        >
                          <span className="font-medium text-brand-navy">{d.class}</span>
                          <span className="text-slate-500">{(d.confidence * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* VLM notes */}
                {result.notes && (
                  <div className="text-sm text-slate-700 leading-relaxed p-3 rounded-md bg-slate-50 border border-brand-line">
                    <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1 flex items-center gap-1">
                      <Eye className="h-3 w-3" /> VLM Analysis
                    </div>
                    {result.notes}
                  </div>
                )}

                {/* Visual characteristics */}
                {result.characteristics.length > 0 && (
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-2">
                      Observed Features
                    </div>
                    <ul className="space-y-1.5">
                      {result.characteristics.map((c, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                          <CheckCircle2 className="h-4 w-4 text-brand-green shrink-0 mt-0.5" />
                          {c}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Breed profile */}
                {result.breedInfo && (
                  <div className="pt-3 border-t border-brand-line">
                    <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-2">
                      Breed Profile
                    </div>
                    <dl className="grid grid-cols-2 gap-2 text-xs">
                      {result.breedInfo.origin && (
                        <div>
                          <dt className="text-slate-500">Origin</dt>
                          <dd className="font-medium text-brand-navy">{result.breedInfo.origin}</dd>
                        </div>
                      )}
                      {result.breedInfo.milkYieldKgPerLactation && (
                        <div>
                          <dt className="text-slate-500">Milk / lactation</dt>
                          <dd className="font-medium text-brand-navy">
                            {result.breedInfo.milkYieldKgPerLactation} kg
                          </dd>
                        </div>
                      )}
                      {result.breedInfo.fatContent && (
                        <div>
                          <dt className="text-slate-500">Fat content</dt>
                          <dd className="font-medium text-brand-navy">{result.breedInfo.fatContent}%</dd>
                        </div>
                      )}
                      {result.breedInfo.heatTolerance && (
                        <div>
                          <dt className="text-slate-500">Heat tolerance</dt>
                          <dd className="font-medium text-brand-navy capitalize">
                            {result.breedInfo.heatTolerance}
                          </dd>
                        </div>
                      )}
                    </dl>
                    {result.breedInfo.description && (
                      <p className="text-xs text-slate-600 mt-3 leading-relaxed">
                        {result.breedInfo.description.slice(0, 200)}...
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </Section>
  );
}

function SourceBadge({
  source,
  agreement,
}: {
  source: "yolo" | "vlm" | "consensus" | "none";
  agreement: boolean;
}) {
  if (source === "consensus") {
    return (
      <span className="flex items-center gap-1.5 text-brand-green-dark">
        <ShieldCheck className="h-3.5 w-3.5" />
        YOLO + VLM Consensus · High confidence
      </span>
    );
  }
  if (source === "yolo") {
    return (
      <span className="flex items-center gap-1.5 text-brand-blue">
        <Cpu className="h-3.5 w-3.5" />
        YOLO Primary {agreement === false ? "· VLM disagrees" : ""}
      </span>
    );
  }
  if (source === "vlm") {
    return (
      <span className="flex items-center gap-1.5 text-amber-700">
        <Eye className="h-3.5 w-3.5" />
        VLM Only {agreement === false ? "· YOLO disagrees" : "· YOLO unavailable"}
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1.5 text-slate-500">
      <AlertTriangle className="h-3.5 w-3.5" />
      No detection
    </span>
  );
}

function ConfidenceMeter({ value }: { value: number }) {
  const color =
    value >= 75
      ? "bg-brand-green"
      : value >= 50
      ? "bg-brand-amber"
      : value > 0
      ? "bg-orange-400"
      : "bg-slate-300";
  const label =
    value >= 75 ? "High" : value >= 50 ? "Medium" : value > 0 ? "Low" : "Uncertain";
  return (
    <div className="text-right shrink-0">
      <div className="text-[11px] uppercase tracking-wider text-slate-500 font-semibold">
        Confidence
      </div>
      <div
        className="text-xl font-bold text-brand-navy"
        style={{ fontFamily: "var(--font-montserrat)" }}
      >
        {value}%
      </div>
      <div className="text-[10px] text-slate-500 mb-1">{label}</div>
      <div className="h-1.5 w-20 bg-slate-200 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}
