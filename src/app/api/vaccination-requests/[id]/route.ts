import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { requireRole } from "@/lib/auth";
import { Role, VaccinationStatus } from "@prisma/client";

export const runtime = "nodejs";

// PATCH /api/vaccination-requests/[id] — vet/admin reviews (approve/reject/schedule/complete)
export async function PATCH(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const session = await requireRole(Role.VET, Role.ADMIN);

    const request = await db.vaccinationRequest.findUnique({
      where: { id: params.id },
      include: { cattle: { select: { name: true, tagNumber: true } } },
    });
    if (!request) {
      return NextResponse.json({ error: "Request not found" }, { status: 404 });
    }

    const body = await req.json();
    const { status, reviewNotes, scheduledDate, vetAssigned } = body;

    if (!status || !Object.values(VaccinationStatus).includes(status)) {
      return NextResponse.json({ error: "Invalid status." }, { status: 400 });
    }

    const data: {
      status: VaccinationStatus;
      reviewedById: string;
      reviewedAt: Date;
      reviewNotes?: string | null;
      scheduledDate?: Date | null;
      vetAssigned?: string | null;
    } = {
      status,
      reviewedById: session.id,
      reviewedAt: new Date(),
      reviewNotes: reviewNotes ?? null,
    };

    if (scheduledDate) data.scheduledDate = new Date(scheduledDate);
    if (vetAssigned) data.vetAssigned = vetAssigned;

    const updated = await db.vaccinationRequest.update({
      where: { id: params.id },
      data,
      include: {
        cattle: { select: { name: true, tagNumber: true, breed: true } },
        requester: { select: { id: true, name: true } },
      },
    });

    // If approved and scheduled, also create a health record for the cattle
    if (status === VaccinationStatus.COMPLETED) {
      await db.healthRecord.create({
        data: {
          cattleId: request.cattleId,
          recordedById: session.id,
          date: new Date(),
          type: "vaccination",
          event: `${request.vaccineName} (completed via request #${request.id.slice(-6)})`,
          description: reviewNotes || `Vaccination completed by ${session.name}`,
        },
      });
    }

    // Notify the farmer about the status update
    const statusMessages: Record<VaccinationStatus, string> = {
      PENDING: "Your request is back to pending.",
      APPROVED: `Your vaccination request for ${request.cattle.name} has been approved!`,
      REJECTED: `Your vaccination request for ${request.cattle.name} was rejected. See notes.`,
      SCHEDULED: `Vaccination scheduled for ${request.cattle.name}. Check the scheduled date.`,
      COMPLETED: `Vaccination completed for ${request.cattle.name}. Health record updated.`,
      CANCELLED: `Your vaccination request for ${request.cattle.name} was cancelled.`,
    };

    await db.notification.create({
      data: {
        userId: request.requesterId,
        type: `vaccination_${status.toLowerCase()}`,
        title: `Vaccination update: ${request.cattle.name}`,
        message: statusMessages[status as VaccinationStatus],
        link: `/vaccination-requests`,
      },
    });

    return NextResponse.json({ request: updated });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    const status = message.includes("Unauthorized") ? 401 : message.includes("Forbidden") ? 403 : 500;
    return NextResponse.json({ error: message }, { status });
  }
}
