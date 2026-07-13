import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { getSession, canManageContent } from "@/lib/auth";
import { Role } from "@prisma/client";

export const runtime = "nodejs";

export async function GET(req: NextRequest) {
  try {
    const session = await getSession();
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { searchParams } = new URL(req.url);
    const cattleId = searchParams.get("cattleId");
    const days = parseInt(searchParams.get("days") || "30", 10);

    // Build where clause — scope to cattle the user owns (or all for vets/admins)
    const since = new Date();
    since.setDate(since.getDate() - days);

    const where: { date: { gte: Date }; cattleId?: string; cattle?: { ownerId?: string } } = {
      date: { gte: since },
    };

    if (cattleId) where.cattleId = cattleId;

    // Farmers only see their own cattle's logs
    if (!canManageContent(session.role as Role)) {
      where.cattle = { ownerId: session.id };
    }

    const logs = await db.milkLog.findMany({
      where,
      orderBy: { date: "desc" },
      include: {
        cattle: { select: { name: true, tagNumber: true, breed: true, ownerId: true } },
      },
    });
    return NextResponse.json({ logs, count: logs.length });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const session = await getSession();
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const { cattleId, date, morningKg, eveningKg, fatPct, snfPct, notes } = body;

    if (!cattleId || morningKg == null || eveningKg == null) {
      return NextResponse.json(
        { error: "cattleId, morningKg, eveningKg are required." },
        { status: 400 }
      );
    }

    // Verify cattle ownership (or vet/admin)
    const cattle = await db.cattle.findUnique({ where: { id: cattleId } });
    if (!cattle) {
      return NextResponse.json({ error: "Cattle not found." }, { status: 404 });
    }
    if (cattle.ownerId !== session.id && !canManageContent(session.role as Role)) {
      return NextResponse.json({ error: "Forbidden: not your cattle." }, { status: 403 });
    }

    const log = await db.milkLog.create({
      data: {
        cattleId,
        recordedById: session.id,
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
