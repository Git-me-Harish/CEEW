import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";

export const runtime = "nodejs";

export async function GET() {
  try {
    const cattle = await db.cattle.findMany({
      orderBy: { createdAt: "desc" },
      include: {
        _count: { select: { milkLogs: true, healthRecords: true } },
      },
    });
    return NextResponse.json({ cattle });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { tagNumber, name, breed, species, sex, birthDate, weightKg, source, notes } = body;

    if (!tagNumber || !name || !breed || !species) {
      return NextResponse.json(
        { error: "tagNumber, name, breed, and species are required." },
        { status: 400 }
      );
    }

    const cattle = await db.cattle.create({
      data: {
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
