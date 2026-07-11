import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";

export const runtime = "nodejs";

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const cattleId = searchParams.get("cattleId");
    const days = parseInt(searchParams.get("days") || "30", 10);

    const where: { cattleId?: string } = {};
    if (cattleId) where.cattleId = cattleId;

    const since = new Date();
    since.setDate(since.getDate() - days);

    const logs = await db.milkLog.findMany({
      where: { ...where, date: { gte: since } },
      orderBy: { date: "desc" },
      include: { cattle: { select: { name: true, tagNumber: true, breed: true } } },
    });
    return NextResponse.json({ logs, count: logs.length });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { cattleId, date, morningKg, eveningKg, fatPct, snfPct, notes } = body;

    if (!cattleId || morningKg == null || eveningKg == null) {
      return NextResponse.json(
        { error: "cattleId, morningKg, eveningKg are required." },
        { status: 400 }
      );
    }

    const log = await db.milkLog.create({
      data: {
        cattleId,
        date: date ? new Date(date) : new Date(),
        morningKg: parseFloat(morningKg),
        eveningKg: parseFloat(eveningKg),
        fatPct: fatPct ? parseFloat(fatPct) : null,
        snfPct: snfPct ? parseFloat(snfPct) : null,
        notes: notes || null,
      },
    });
    return NextResponse.json({ log });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
