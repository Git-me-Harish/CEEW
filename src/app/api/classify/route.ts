import { NextRequest, NextResponse } from "next/server";
import ZAI from "z-ai-web-dev-sdk";
import { breeds } from "@/data/breeds";

export const runtime = "nodejs";
export const maxDuration = 60;

interface ClassificationResult {
  breed: string;
  breedId: string | null;
  confidence: number;
  characteristics: string[];
  notes: string;
}

// Result is always returned, even on partial failure
async function classifyImage(base64Image: string): Promise<ClassificationResult> {
  const zai = await ZAI.create();

  const breedNames = breeds.map((b) => b.name).join(", ");
  const prompt = `You are an expert bovine classifier specialised in Indian cattle and buffalo breeds. Look at this image carefully and identify the breed from this list (or indicate if not in list): ${breedNames}

Return a STRICT JSON response with this exact schema:
{
  "breed": "exact breed name from list or 'Unknown'",
  "confidence": 0-100 integer,
  "characteristics": ["3-4 visual features you observed"],
  "notes": "1-sentence explanation of identification rationale. If not a bovine, say so."
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

  // Extract JSON from response
  let jsonMatch = content.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    return {
      breed: "Unknown",
      breedId: null,
      confidence: 0,
      characteristics: [],
      notes: "Unable to parse model output. Please try a clearer image.",
    };
  }

  try {
    const parsed = JSON.parse(jsonMatch[0]);
    const breedName = (parsed.breed || "Unknown").toString();
    const matchedBreed = breeds.find(
      (b) => b.name.toLowerCase() === breedName.toLowerCase()
    );
    return {
      breed: breedName,
      breedId: matchedBreed?.id ?? null,
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
      notes: "Unable to parse classification result.",
    };
  }
}

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

    const result = await classifyImage(base64);

    // Enrich with breed info if matched
    const breedInfo = result.breedId
      ? breeds.find((b) => b.id === result.breedId)
      : null;

    return NextResponse.json({
      ...result,
      breedInfo: breedInfo
        ? {
            id: breedInfo.id,
            name: breedInfo.name,
            type: breedInfo.type,
            category: breedInfo.category,
            origin: breedInfo.origin,
            milkYieldKgPerLactation: breedInfo.milkYieldKgPerLactation,
            fatContent: breedInfo.fatContent,
            description: breedInfo.description,
            distinguishingFeatures: breedInfo.distinguishingFeatures,
            heatTolerance: breedInfo.heatTolerance,
            diseaseResistance: breedInfo.diseaseResistance,
          }
        : null,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("Classification API error:", message);
    return NextResponse.json(
      { error: `Classification failed: ${message}` },
      { status: 500 }
    );
  }
}
