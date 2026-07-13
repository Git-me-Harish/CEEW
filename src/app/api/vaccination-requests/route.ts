import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { getSession, canManageContent } from "@/lib/auth";
import { Role, VaccinationStatus } from "@prisma/client";
import { vaccineSchedule } from "@/data/health";

export const runtime = "nodejs";

// GET /api/vaccination-requests — list (farmers see own; vets/admins see all)
export async function GET(req: NextRequest) {
  try {
    const session = await getSession();
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { searchParams } = new URL(req.url);
    const status = searchParams.get("status") as VaccinationStatus | null;
    const mine = searchParams.get("mine") === "true";

    const where: {
      requesterId?: string;
      status?: VaccinationStatus;
    } = {};

    if (session.role === Role.FARMER || mine) {
      where.requesterId = session.id;
    }
    if (status) where.status = status;

    const requests = await db.vaccinationRequest.findMany({
      where,
      orderBy: { createdAt: "desc" },
      include: {
        requester: {
          select: { id: true, name: true, phone: true, location: true },
        },
        cattle: {
          select: { id: true, name: true, tagNumber: true, breed: true, species: true },
        },
        reviewer: {
          select: { id: true, name: true },
        },
      },
      take: 100,
    });

    return NextResponse.json({
      requests,
      canManage: canManageContent(session.role),
      vaccineOptions: vaccineSchedule.map((v) => ({
        name: v.vaccineName,
        disease: v.disease,
        timing: v.timing,
      })),
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

// POST /api/vaccination-requests — farmer creates a new vaccination request
export async function POST(req: NextRequest) {
  try {
    const session = await getSession();
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const { cattleId, vaccineName, requestedDate, preferredTime, notes } = body;

    if (!cattleId || !vaccineName || !requestedDate) {
      return NextResponse.json(
        { error: "cattleId, vaccineName, and requestedDate are required." },
        { status: 400 }
      );
    }

    // Verify cattle belongs to the farmer (unless vet/admin creating on behalf)
    const cattle = await db.cattle.findUnique({ where: { id: cattleId } });
    if (!cattle) {
      return NextResponse.json({ error: "Cattle not found." }, { status: 404 });
    }
    if (session.role === Role.FARMER && cattle.ownerId !== session.id) {
      return NextResponse.json({ error: "Forbidden: not your cattle." }, { status: 403 });
    }

    const request = await db.vaccinationRequest.create({
      data: {
        requesterId: cattle.ownerId,
        cattleId,
        vaccineName,
        requestedDate: new Date(requestedDate),
        preferredTime: preferredTime || null,
        notes: notes || null,
      },
      include: {
        cattle: { select: { name: true, tagNumber: true, breed: true } },
      },
    });

    // Notify all vets/admins about the new vaccination request
    const managers = await db.user.findMany({
      where: { role: { in: [Role.VET, Role.ADMIN] }, active: true },
      select: { id: true },
    });
    if (managers.length > 0) {
      await db.notification.createMany({
        data: managers.map((m) => ({
          userId: m.id,
          type: "vaccination_new",
          title: `New vaccination request`,
          message: `${session.name} requested ${vaccineName} for ${cattle.name} (${cattle.tagNumber}).`,
          link: `/vaccination-requests`,
        })),
      });
    }

    return NextResponse.json({ request });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
