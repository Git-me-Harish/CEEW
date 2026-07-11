import { NextRequest, NextResponse } from "next/server";
import ZAI from "z-ai-web-dev-sdk";
import { breeds } from "@/data/breeds";

export const runtime = "nodejs";
export const maxDuration = 60;

interface YoloDetection {
  class: string;
  classId: number;
  confidence: number;
  bbox: number[];
}

interface YoloResponse {
  success: boolean;
  primary: YoloDetection | null;
  allDetections: YoloDetection[];
  annotatedImage: string | null;
  modelLoaded: boolean;
  classesAvailable: string[];
  thresholds: { conf: number; iou: number };
}

interface VlmResult {
  breed: string;
  breedId: string | null;
  confidence: number;
  characteristics: string[];
  notes: string;
}

interface HybridResult {
  primary: {
    source: "yolo" | "vlm" | "consensus" | "none";
    breed: string;
    breedId: string | null;
    confidence: number;
    yoloConfidence?: number;
    vlmConfidence?: number;
  };
  vlmResult: VlmResult;
  yoloResult: {
    available: boolean;
    primary: YoloDetection | null;
    allDetections: YoloDetection[];
    annotatedImage: string | null;
    classesAvailable: string[];
  };
  agreement: boolean;
  characteristics: string[];
  notes: string;
  breedInfo: ReturnType<typeof getBreedInfo> | null;
}

function getBreedInfo(name: string | null) {
  if (!name) return null;
  const matched = breeds.find(
    (b) =>
      b.name.toLowerCase() === name.toLowerCase() ||
      b.name.toLowerCase().includes(name.toLowerCase()) ||
      name.toLowerCase().includes(b.name.toLowerCase())
  );
  if (!matched) return null;
  return {
    id: matched.id,
    name: matched.name,
    type: matched.type,
    category: matched.category,
    origin: matched.origin,
    milkYieldKgPerLactation: matched.milkYieldKgPerLactation,
    fatContent: matched.fatContent,
    description: matched.description,
    distinguishingFeatures: matched.distinguishingFeatures,
    heatTolerance: matched.heatTolerance,
    diseaseResistance: matched.diseaseResistance,
  };
}

// --- YOLO call --------------------------------------------------------------
const YOLO_URL = process.env.YOLO_URL || "http://localhost:8501";

async function callYolo(buffer: Buffer, mimeType: string): Promise<YoloResponse | null> {
  try {
    const formData = new FormData();
    const blob = new Blob([buffer], { type: mimeType });
    formData.append("image", blob, "upload.jpg");

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 25000); // 25s timeout

    const res = await fetch(`${YOLO_URL}/detect?conf=0.25&iou=0.45`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (res.status === 503) {
      // Model not loaded — graceful fallback
      console.log("[classify] YOLO service returned 503 — model not loaded");
      return null;
    }
    if (!res.ok) {
      console.warn(`[classify] YOLO service returned ${res.status}`);
      return null;
    }
    return (await res.json()) as YoloResponse;
  } catch (err) {
    console.warn(
      "[classify] YOLO service unreachable:",
      err instanceof Error ? err.message : "unknown error"
    );
    return null;
  }
}

// --- VLM call (secondary refinement) ----------------------------------------
async function callVlm(base64Image: string, yoloHint?: string): Promise<VlmResult> {
  const zai = await ZAI.create();

  const breedNames = breeds.map((b) => b.name).join(", ");
  const hintClause = yoloHint
    ? `A YOLO detection model has preliminarily identified this as "${yoloHint}". Verify this prediction against the visual features. If you strongly disagree (clearly different visual features), say so.`
    : "";

  const prompt = `You are an expert bovine classifier specialised in Indian cattle and buffalo breeds. Look at this image carefully and identify the breed from this list (or indicate if not in list): ${breedNames}

${hintClause}

Return a STRICT JSON response with this exact schema:
{
  "breed": "exact breed name from list or 'Unknown'",
  "confidence": 0-100 integer (your confidence in this identification),
  "characteristics": ["3-4 visual features you observed"],
  "notes": "1-sentence explanation of identification rationale. If not a bovine, say so. If you agree/disagree with any YOLO hint, mention it briefly."
}

Rules:
- If the image is not a bovine/cow/buffalo/cattle, set breed to "Unknown" and confidence to 0.
- Confidence reflects how sure you are that this is the breed (not just a bovine).
- Be conservative: only assign confidence > 75 for clear, distinctive breed features.
- Do not include any text outside the JSON.`;

  const response = await zai.chat.completions.createVision({
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: prompt },
          { type: "image_url", image_url: { url: base64Image } },
        ],
      },
    ],
    thinking: { type: "disabled" },
  });

  const content = response.choices[0]?.message?.content || "";
  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    return {
      breed: "Unknown",
      breedId: null,
      confidence: 0,
      characteristics: [],
      notes: "Unable to parse VLM output.",
    };
  }
  try {
    const parsed = JSON.parse(jsonMatch[0]);
    const breedName = (parsed.breed || "Unknown").toString();
    const matched = breeds.find(
      (b) => b.name.toLowerCase() === breedName.toLowerCase()
    );
    return {
      breed: breedName,
      breedId: matched?.id ?? null,
      confidence: Math.min(100, Math.max(0, parseInt(parsed.confidence, 10) || 0)),
      characteristics: Array.isArray(parsed.characteristics)
        ? parsed.characteristics.slice(0, 5).map(String)
        : [],
      notes: parsed.notes || "",
    };
  } catch {
    return {
      breed: "Unknown",
      breedId: null,
      confidence: 0,
      characteristics: [],
      notes: "Unable to parse VLM classification result.",
    };
  }
}

// --- Main hybrid handler ----------------------------------------------------
export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("image") as File | null;

    if (!file) {
      return NextResponse.json(
        { error: "No image file provided. Use 'image' as the form field name." },
        { status: 400 }
      );
    }
    if (!file.type.startsWith("image/")) {
      return NextResponse.json(
        { error: "Uploaded file is not an image." },
        { status: 400 }
      );
    }
    if (file.size > 10 * 1024 * 1024) {
      return NextResponse.json(
        { error: "Image too large. Maximum allowed size is 10 MB." },
        { status: 400 }
      );
    }

    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    const base64 = `data:${file.type};base64,${buffer.toString("base64")}`;

    // --- Step 1: YOLO (primary) ---
    const yoloResult = await callYolo(buffer, file.type);

    // --- Step 2: VLM (secondary refinement, with YOLO hint if available) ---
    const vlmResult = await callVlm(base64, yoloResult?.primary?.class);

    // --- Step 3: Combine into final result ---
    const yoloBreed = yoloResult?.primary?.class ?? null;
    const yoloConf = yoloResult?.primary?.confidence ?? 0;
    const vlmBreed = vlmResult.breed;
    const vlmConf = vlmResult.confidence;

    let primarySource: "yolo" | "vlm" | "consensus" | "none" = "none";
    let primaryBreed = "Unknown";
    let primaryBreedId: string | null = null;
    let primaryConfidence = 0;

    if (yoloBreed && vlmBreed && vlmBreed !== "Unknown") {
      // Both available — check agreement
      const normalise = (s: string) => s.toLowerCase().replace(/[^a-z]/g, "");
      const agreement =
        normalise(yoloBreed) === normalise(vlmBreed) ||
        normalise(yoloBreed).includes(normalise(vlmBreed)) ||
        normalise(vlmBreed).includes(normalise(yoloBreed));

      if (agreement) {
        // Consensus — highest confidence
        primarySource = "consensus";
        primaryBreed = vlmBreed; // VLM name is more precise (matches our breed DB)
        primaryBreedId = vlmResult.breedId;
        primaryConfidence = Math.min(100, Math.round((yoloConf * 100 + vlmConf) / 2 + 10)); // boost for consensus
      } else {
        // Disagreement — pick higher confidence
        if (yoloConf * 100 >= vlmConf) {
          primarySource = "yolo";
          primaryBreed = yoloBreed;
          primaryBreedId = getBreedInfo(yoloBreed)?.id ?? null;
          primaryConfidence = Math.round(yoloConf * 100);
        } else {
          primarySource = "vlm";
          primaryBreed = vlmBreed;
          primaryBreedId = vlmResult.breedId;
          primaryConfidence = vlmConf;
        }
      }
    } else if (yoloBreed) {
      // Only YOLO detected something
      primarySource = "yolo";
      primaryBreed = yoloBreed;
      primaryBreedId = getBreedInfo(yoloBreed)?.id ?? null;
      primaryConfidence = Math.round(yoloConf * 100);
    } else if (vlmBreed && vlmBreed !== "Unknown") {
      // Only VLM identified (YOLO unavailable or didn't detect)
      primarySource = "vlm";
      primaryBreed = vlmBreed;
      primaryBreedId = vlmResult.breedId;
      primaryConfidence = vlmConf;
    }

    const agreement =
      yoloBreed && vlmBreed && vlmBreed !== "Unknown"
        ? (() => {
            const n = (s: string) => s.toLowerCase().replace(/[^a-z]/g, "");
            return n(yoloBreed) === n(vlmBreed) || n(yoloBreed).includes(n(vlmBreed)) || n(vlmBreed).includes(n(yoloBreed));
          })()
        : false;

    const breedInfo = getBreedInfo(primaryBreed);

    const result: HybridResult = {
      primary: {
        source: primarySource,
        breed: primaryBreed,
        breedId: primaryBreedId,
        confidence: primaryConfidence,
        yoloConfidence: yoloBreed ? Math.round(yoloConf * 100) : undefined,
        vlmConfidence: vlmBreed !== "Unknown" ? vlmConf : undefined,
      },
      vlmResult,
      yoloResult: {
        available: yoloResult?.modelLoaded ?? false,
        primary: yoloResult?.primary ?? null,
        allDetections: yoloResult?.allDetections ?? [],
        annotatedImage: yoloResult?.annotatedImage ?? null,
        classesAvailable: yoloResult?.classesAvailable ?? [],
      },
      agreement,
      characteristics: vlmResult.characteristics,
      notes: vlmResult.notes,
      breedInfo,
    };

    return NextResponse.json(result);
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("Classification API error:", message);
    return NextResponse.json(
      { error: `Classification failed: ${message}` },
      { status: 500 }
    );
  }
}
