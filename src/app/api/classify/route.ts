import { NextRequest, NextResponse } from "next/server";
import fs from "fs/promises";
import path from "path";
import os from "os";
import { breeds } from "@/data/breeds";

export const runtime = "nodejs";
export const maxDuration = 60;

// --- Z.ai config loading (mirrors the SDK's loadConfig) --------------------
interface ZaiConfig {
  baseUrl: string;
  apiKey: string;
  token?: string;
  chatId?: string;
  userId?: string;
}

async function loadZaiConfig(): Promise<ZaiConfig> {
  const homeDir = os.homedir();
  const configPaths = [
    path.join(process.cwd(), ".z-ai-config"),
    path.join(homeDir, ".z-ai-config"),
    "/etc/.z-ai-config",
  ];
  for (const filePath of configPaths) {
    try {
      const configStr = await fs.readFile(filePath, "utf-8");
      const config = JSON.parse(configStr);
      if (config.baseUrl && config.apiKey) {
        return {
          baseUrl: config.baseUrl,
          apiKey: config.apiKey,
          token: config.token,
          chatId: config.chatId,
          userId: config.userId,
        };
      }
    } catch {
      // continue to next path
    }
  }
  throw new Error(
    "Z.ai configuration not found. Create a .z-ai-config file with { \"apiKey\": \"...\", \"baseUrl\": \"https://api.z.ai/api/paas/v4\" } in the project root."
  );
}

// Build the standard set of headers the Z.ai API expects
function buildHeaders(config: ZaiConfig): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${config.apiKey}`,
    "X-Z-AI-From": "Z",
  };
  if (config.chatId) headers["X-Chat-Id"] = config.chatId;
  if (config.userId) headers["X-User-Id"] = config.userId;
  if (config.token) headers["X-Token"] = config.token;
  return headers;
}

// --- YOLO types -------------------------------------------------------------
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

// --- Result types -----------------------------------------------------------
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
  vlmStatus: {
    available: boolean;
    error: string | null;
  };
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
    const timeout = setTimeout(() => controller.abort(), 25000);

    const res = await fetch(`${YOLO_URL}/detect?conf=0.25&iou=0.45`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (res.status === 503) {
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

interface VlmFailure {
  ok: false;
  error: string;
}
interface VlmSuccess {
  ok: true;
  result: VlmResult;
}
type VlmOutcome = VlmSuccess | VlmFailure;

async function callVlm(base64Image: string, yoloHint?: string): Promise<VlmOutcome> {
  let config: ZaiConfig;
  try {
    config = await loadZaiConfig();
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : "Config load failed" };
  }
  const headers = buildHeaders(config);

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

  const multimodalBody = {
    model: "glm-4.6v",
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
  };

  let content = "";
  let lastError = "";

  // --- Attempt 1: try /chat/completions/vision (internal API) ---
  try {
    const visionUrl = `${config.baseUrl}/chat/completions/vision`;
    const res = await fetch(visionUrl, {
      method: "POST",
      headers,
      body: JSON.stringify(multimodalBody),
    });
    if (res.ok) {
      const data = await res.json();
      content = data.choices?.[0]?.message?.content || "";
    } else if (res.status !== 404) {
      const errBody = await res.text();
      lastError = `vision endpoint ${res.status}: ${errBody.slice(0, 200)}`;
      console.warn(`[classify] /chat/completions/vision returned ${res.status}`);
    }
  } catch (err) {
    lastError = err instanceof Error ? err.message : "vision endpoint network error";
    console.warn("[classify] /chat/completions/vision failed:", lastError);
  }

  // --- Attempt 2: fall back to /chat/completions with multimodal content (public API) ---
  if (!content) {
    try {
      const standardUrl = `${config.baseUrl}/chat/completions`;
      const res = await fetch(standardUrl, {
        method: "POST",
        headers,
        body: JSON.stringify(multimodalBody),
      });
      if (!res.ok) {
        const errBody = await res.text();
        lastError = `standard endpoint ${res.status}: ${errBody.slice(0, 200)}`;
        console.warn(`[classify] /chat/completions returned ${res.status}`);
      } else {
        const data = await res.json();
        content = data.choices?.[0]?.message?.content || "";
      }
    } catch (err) {
      lastError = err instanceof Error ? err.message : "standard endpoint network error";
      console.warn("[classify] /chat/completions failed:", lastError);
    }
  }

  // --- If both attempts failed, return failure (DO NOT throw) ---
  if (!content) {
    return {
      ok: false,
      error: lastError || "VLM returned empty response. Likely insufficient Z.ai balance or invalid API key.",
    };
  }

  // --- Parse the JSON response from the model ---
  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    return { ok: false, error: "VLM response did not contain valid JSON" };
  }
  try {
    const parsed = JSON.parse(jsonMatch[0]);
    const breedName = (parsed.breed || "Unknown").toString();
    const matched = breeds.find(
      (b) => b.name.toLowerCase() === breedName.toLowerCase()
    );
    return {
      ok: true,
      result: {
        breed: breedName,
        breedId: matched?.id ?? null,
        confidence: Math.min(100, Math.max(0, parseInt(parsed.confidence, 10) || 0)),
        characteristics: Array.isArray(parsed.characteristics)
          ? parsed.characteristics.slice(0, 5).map(String)
          : [],
        notes: parsed.notes || "",
      },
    };
  } catch {
    return { ok: false, error: "Failed to parse VLM JSON response" };
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

    // --- Step 1: YOLO (primary, REQUIRED) ---
    const yoloResult = await callYolo(buffer, file.type);

    // --- Step 2: VLM (secondary refinement, BEST-EFFORT) ---
    // If VLM fails (no Z.ai balance, network error, etc.), we still return
    // a valid YOLO-only result. The VLM is a bonus, not a requirement.
    const vlmOutcome = await callVlm(base64, yoloResult?.primary?.class);
    const vlmOk = vlmOutcome.ok;
    const vlmResult: VlmResult = vlmOk
      ? vlmOutcome.result
      : { breed: "Unknown", breedId: null, confidence: 0, characteristics: [], notes: "" };
    const vlmError: string | null = vlmOk ? null : vlmOutcome.error;

    // --- Step 3: Combine into final result ---
    const yoloBreed = yoloResult?.primary?.class ?? null;
    const yoloConf = yoloResult?.primary?.confidence ?? 0;
    const vlmBreed = vlmResult.breed;
    const vlmConf = vlmResult.confidence;

    let primarySource: "yolo" | "vlm" | "consensus" | "none" = "none";
    let primaryBreed = "Unknown";
    let primaryBreedId: string | null = null;
    let primaryConfidence = 0;

    const normalise = (s: string) => s.toLowerCase().replace(/[^a-z]/g, "");

    // Only consider VLM for consensus if it actually succeeded
    const vlmAvailable = vlmOk && vlmBreed && vlmBreed !== "Unknown";

    if (yoloBreed && vlmAvailable) {
      const agreement =
        normalise(yoloBreed) === normalise(vlmBreed) ||
        normalise(yoloBreed).includes(normalise(vlmBreed)) ||
        normalise(vlmBreed).includes(normalise(yoloBreed));

      if (agreement) {
        primarySource = "consensus";
        primaryBreed = vlmBreed;
        primaryBreedId = vlmResult.breedId;
        primaryConfidence = Math.min(100, Math.round((yoloConf * 100 + vlmConf) / 2 + 10));
      } else {
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
      // YOLO-only result (VLM either failed or returned Unknown)
      primarySource = "yolo";
      primaryBreed = yoloBreed;
      primaryBreedId = getBreedInfo(yoloBreed)?.id ?? null;
      primaryConfidence = Math.round(yoloConf * 100);
    } else if (vlmAvailable) {
      // VLM-only result (YOLO unavailable)
      primarySource = "vlm";
      primaryBreed = vlmBreed;
      primaryBreedId = vlmResult.breedId;
      primaryConfidence = vlmConf;
    }

    const agreement =
      yoloBreed && vlmAvailable
        ? normalise(yoloBreed) === normalise(vlmBreed) ||
          normalise(yoloBreed).includes(normalise(vlmBreed)) ||
          normalise(vlmBreed).includes(normalise(yoloBreed))
        : false;

    const breedInfo = getBreedInfo(primaryBreed);

    const result: HybridResult = {
      primary: {
        source: primarySource,
        breed: primaryBreed,
        breedId: primaryBreedId,
        confidence: primaryConfidence,
        yoloConfidence: yoloBreed ? Math.round(yoloConf * 100) : undefined,
        vlmConfidence: vlmAvailable ? vlmConf : undefined,
      },
      vlmResult,
      vlmStatus: {
        available: vlmOk,
        error: vlmError,
      },
      yoloResult: {
        available: yoloResult?.modelLoaded ?? false,
        primary: yoloResult?.primary ?? null,
        allDetections: yoloResult?.allDetections ?? [],
        annotatedImage: yoloResult?.annotatedImage ?? null,
        classesAvailable: yoloResult?.classesAvailable ?? [],
      },
      agreement,
      characteristics: vlmResult.characteristics,
      notes: vlmOk ? vlmResult.notes : (vlmError || ""),
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
