import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { getSession, canManageContent } from "@/lib/auth";
import { Role, TicketStatus } from "@prisma/client";

export const runtime = "nodejs";

// POST /api/tickets/[id]/reply — add a reply to a ticket
export async function POST(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const session = await getSession();
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const ticket = await db.ticket.findUnique({ where: { id: params.id } });
    if (!ticket) {
      return NextResponse.json({ error: "Ticket not found" }, { status: 404 });
    }

    // Farmers can only reply to their own tickets
    if (session.role === Role.FARMER && ticket.authorId !== session.id) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const body = await req.json();
    const { body: replyBody, internal, markInProgress, markResolved } = body;

    if (!replyBody || !replyBody.trim()) {
      return NextResponse.json({ error: "Reply body is required." }, { status: 400 });
    }

    // Only vets/admins can post internal notes
    const isInternal = internal && canManageContent(session.role);

    const reply = await db.ticketReply.create({
      data: {
        ticketId: params.id,
        authorId: session.id,
        body: replyBody.trim(),
        internal: isInternal || false,
      },
      include: {
        author: { select: { id: true, name: true, role: true } },
      },
    });

    // Auto-update ticket status if requested (vet/admin action)
    if (canManageContent(session.role)) {
      if (markInProgress && ticket.status === "OPEN") {
        await db.ticket.update({
          where: { id: params.id },
          data: { status: TicketStatus.IN_PROGRESS },
        });
      }
      if (markResolved) {
        await db.ticket.update({
          where: { id: params.id },
          data: { status: TicketStatus.RESOLVED },
        });
      }
    }

    // Notify the ticket author (if reply author is not the author)
    if (ticket.authorId !== session.id) {
      await db.notification.create({
        data: {
          userId: ticket.authorId,
          type: "ticket_reply",
          title: `New reply on "${ticket.title.slice(0, 40)}"`,
          message: `${session.name} responded to your problem.`,
          link: `/tickets`,
        },
      });
    } else {
      // If farmer replied, notify vets/admins
      const managers = await db.user.findMany({
        where: { role: { in: [Role.VET, Role.ADMIN] }, active: true },
        select: { id: true },
      });
      if (managers.length > 0) {
        await db.notification.createMany({
          data: managers.map((m) => ({
            userId: m.id,
            type: "ticket_reply",
            title: `Farmer replied on "${ticket.title.slice(0, 40)}"`,
            message: `${session.name} added information to their ticket.`,
            link: `/tickets`,
          })),
        });
      }
    }

    return NextResponse.json({ reply });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
