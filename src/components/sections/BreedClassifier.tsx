"use client";

import { useState, useRef } from "react";
import { Upload, Loader2, ImageIcon, AlertCircle, CheckCircle2, X, RotateCcw, Sparkles } from "lucide-react";
import { Section, SectionHeading } from "./Section";
import { getBreedById, type Breed } from "@/data/breeds";

interface ClassificationResult {
  breed: string;
  breedId: string | null;
  confidence: number;
  characteristics: string[];
  notes: string;
  breedInfo: Partial<Breed> | null;
}

export function BreedClassifier() {
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

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
      setResult(data);
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
        eyebrow="AI · Computer Vision"
        title="Identify any Indian bovine breed"
        subtitle="Upload a clear photo of the cattle or buffalo and our AI vision model identifies the breed, lists distinguishing features, and pulls up the full breed profile from our database."
      />

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
                    <button
                      onClick={reset}
                      className="btn-secondary text-xs"
                      disabled={loading}
                    >
                      <RotateCcw className="h-3.5 w-3.5" /> Change
                    </button>
                    <button onClick={classify} disabled={loading} className="btn-primary">
                      {loading ? (
                        <>
                          <Loader2 className="h-4 w-4 animate-spin" /> Classifying...
                        </>
                      ) : (
                        <>
                          <Sparkles /> Identify Breed
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
              Photo tips for best accuracy
            </h4>
            <ul className="text-xs text-slate-600 space-y-1.5">
              <li>• Capture the full side profile including horns, hump, and udder.</li>
              <li>• Ensure good natural lighting — avoid harsh shadows or backlit shots.</li>
              <li>• Frame the animal centrally with at least 1 m of surrounding context.</li>
              <li>• For buffalo, ensure the horn shape is clearly visible from above or side.</li>
              <li>• Avoid photos with multiple animals — focus on a single individual.</li>
            </ul>
          </div>
        </div>

        {/* Results column */}
        <div className="lg:col-span-2">
          <div className="card-soft p-6 sticky top-24">
            <h3
              className="text-base font-semibold text-brand-navy mb-4"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              Classification Result
            </h3>

            {!result && !loading && (
              <div className="text-center py-10 text-slate-400">
                <div className="h-14 w-14 mx-auto rounded-full bg-brand-mist flex items-center justify-center mb-3">
                  <ImageIcon className="h-6 w-6" />
                </div>
                <p className="text-sm">Your result will appear here.</p>
              </div>
            )}

            {loading && (
              <div className="text-center py-10">
                <Loader2 className="h-10 w-10 mx-auto text-brand-blue animate-spin mb-3" />
                <p className="text-sm text-slate-600">Analysing image with vision AI...</p>
                <p className="text-xs text-slate-400 mt-1">
                  Matching against {32}+ Indian bovine breed characteristics.
                </p>
              </div>
            )}

            {result && !loading && (
              <div className="space-y-4">
                {/* Breed name & confidence */}
                <div className="p-4 rounded-md bg-brand-blue-50 border border-brand-blue-100">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <div className="text-[11px] uppercase tracking-wider text-brand-blue font-semibold mb-0.5">
                        Identified Breed
                      </div>
                      <div
                        className="text-lg font-bold text-brand-navy"
                        style={{ fontFamily: "var(--font-montserrat)" }}
                      >
                        {result.breed}
                      </div>
                      {result.breedInfo?.type && (
                        <div className="text-xs text-slate-600 mt-0.5">
                          {result.breedInfo.type === "buffalo" ? "Buffalo" : "Cattle"} ·{" "}
                          {result.breedInfo.category}
                        </div>
                      )}
                    </div>
                    <ConfidenceMeter value={result.confidence} />
                  </div>
                </div>

                {/* Notes */}
                {result.notes && (
                  <div className="text-sm text-slate-700 leading-relaxed p-3 rounded-md bg-slate-50 border border-brand-line">
                    {result.notes}
                  </div>
                )}

                {/* Visual characteristics */}
                {result.characteristics.length > 0 && (
                  <div>
                    <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold mb-2">
                      Observed features
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

                {/* Breed profile link */}
                {result.breedId && result.breedInfo && (
                  <div className="pt-3 border-t border-brand-line">
                    <div className="text-xs uppercase tracking-wider text-slate-500 font-semibold mb-2">
                      Breed Profile
                    </div>
                    <dl className="grid grid-cols-2 gap-2 text-xs">
                      {result.breedInfo.origin && (
                        <div>
                          <dt className="text-slate-500">Origin</dt>
                          <dd className="font-medium text-brand-navy">
                            {result.breedInfo.origin}
                          </dd>
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
                          <dd className="font-medium text-brand-navy">
                            {result.breedInfo.fatContent}%
                          </dd>
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
                        {result.breedInfo.description.slice(0, 180)}...
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


