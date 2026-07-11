import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";

export const runtime = "nodejs";

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const cattleId = searchParams.get("cattleId");

    const where: { cattleId?: string } = {};
    if (cattleId) where.cattleId = cattleId;

    const records = await db.healthRecord.findMany({
      where,
      orderBy: { date: "desc" },
      include: { cattle: { select: { name: true, tagNumber: true } } },
    });
    return NextResponse.json({ records });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { cattleId, date, type, event, description, cost } = body;

    if (!cattleId || !type || !event) {
      return NextResponse.json(
        { error: "cattleId, type, and event are required." },
        { status: 400 }
      );
    }

    const record = await db.healthRecord.create({
      data: {
        cattleId,
        date: date ? new Date(date) : new Date(),
        type,
        event,
        description: description || null,
        cost: cost ? parseFloat(cost) : null,
      },
    });
    return NextResponse.json({ record });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
