import { NextResponse } from "next/server";

export const runtime = "nodejs";

// Proxies to the Python YOLO service's /health endpoint
export async function GET() {
  const YOLO_URL = process.env.YOLO_URL || "http://localhost:8501";
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    const res = await fetch(`${YOLO_URL}/health`, { signal: controller.signal });
    clearTimeout(timeout);
    if (!res.ok) {
      return NextResponse.json(
        { available: false, modelLoaded: false, error: `YOLO service returned ${res.status}` },
        { status: 200 }
      );
    }
    const data = await res.json();
    return NextResponse.json({
      available: true,
      modelLoaded: data.modelLoaded === true,
      classes: data.classes || [],
      error: data.error || null,
    });
  } catch (err) {
    return NextResponse.json(
      {
        available: false,
        modelLoaded: false,
        error: err instanceof Error ? err.message : "YOLO service unreachable",
      },
      { status: 200 }
    );
  }
}
