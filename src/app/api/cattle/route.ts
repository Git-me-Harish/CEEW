import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { getSession, canManageContent } from "@/lib/auth";
import { Role } from "@prisma/client";

export const runtime = "nodejs";

// GET /api/cattle — list cattle (farmers see own; vets/admins see all)
export async function GET() {
  try {
    const session = await getSession();
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const where = canManageContent(session.role as Role) ? {} : { ownerId: session.id };

    const cattle = await db.cattle.findMany({
      where,
      orderBy: { createdAt: "desc" },
      include: {
        owner: { select: { id: true, name: true } },
        _count: { select: { milkLogs: true, healthRecords: true, vaccinationRequests: true } },
      },
    });
    return NextResponse.json({ cattle, canManage: canManageContent(session.role as Role) });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

// POST /api/cattle — create a cattle record (authenticated)
export async function POST(req: NextRequest) {
  try {
    const session = await getSession();
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const { tagNumber, name, breed, species, sex, birthDate, weightKg, source, notes } = body;

    if (!tagNumber || !name || !breed || !species) {
      return NextResponse.json(
        { error: "tagNumber, name, breed, and species are required." },
        { status: 400 }
      );
    }

    // Check tag uniqueness within owner's herd
    const existing = await db.cattle.findUnique({
      where: { ownerId_tagNumber: { ownerId: session.id, tagNumber } },
    });
    if (existing) {
      return NextResponse.json(
        { error: `You already have a cattle with tag number ${tagNumber}.` },
        { status: 409 }
      );
    }

    const cattle = await db.cattle.create({
      data: {
        ownerId: session.id,
        tagNumber,
        name,
        breed,
        species,
        sex: sex || "female",
        birthDate: birthDate ? new Date(birthDate) : new Date(),
        weightKg: weightKg ? parseFloat(weightKg) : 0,
        source: source || null,
        notes: notes || null,
      },
    });
    return NextResponse.json({ cattle });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
