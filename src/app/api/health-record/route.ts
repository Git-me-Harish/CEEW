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

    const where: { cattleId?: string; cattle?: { ownerId?: string } } = {};
    if (cattleId) where.cattleId = cattleId;
    if (!canManageContent(session.role as Role)) {
      where.cattle = { ownerId: session.id };
    }

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
    const session = await getSession();
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const { cattleId, date, type, event, description, cost } = body;

    if (!cattleId || !type || !event) {
      return NextResponse.json(
        { error: "cattleId, type, and event are required." },
        { status: 400 }
      );
    }

    const cattle = await db.cattle.findUnique({ where: { id: cattleId } });
    if (!cattle) {
      return NextResponse.json({ error: "Cattle not found." }, { status: 404 });
    }
    if (cattle.ownerId !== session.id && !canManageContent(session.role as Role)) {
      return NextResponse.json({ error: "Forbidden: not your cattle." }, { status: 403 });
    }

    const record = await db.healthRecord.create({
      data: {
        cattleId,
        recordedById: session.id,
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
