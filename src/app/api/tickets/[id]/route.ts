import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { getSession, canManageContent } from "@/lib/auth";
import { Role, TicketStatus } from "@prisma/client";

export const runtime = "nodejs";

// GET /api/tickets/[id] — fetch single ticket with replies
export async function GET(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const session = await getSession();
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const ticket = await db.ticket.findUnique({
      where: { id: params.id },
      include: {
        author: { select: { id: true, name: true, role: true, location: true } },
        replies: {
          orderBy: { createdAt: "asc" },
          include: {
            author: { select: { id: true, name: true, role: true } },
          },
        },
      },
    });

    if (!ticket) {
      return NextResponse.json({ error: "Ticket not found" }, { status: 404 });
    }

    // Farmers can only see their own tickets
    if (session.role === Role.FARMER && ticket.authorId !== session.id) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    // Filter internal notes for farmers
    const visibleReplies =
      session.role === Role.FARMER
        ? ticket.replies.filter((r) => !r.internal)
        : ticket.replies;

    return NextResponse.json({
      ticket: { ...ticket, replies: visibleReplies },
      canManage: canManageContent(session.role),
      currentUserId: session.id,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

// PATCH /api/tickets/[id] — update status (vet/admin only, or farmer closing own)
export async function PATCH(
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

    const body = await req.json();
    const { status, priority } = body;

    // Farmers can only close their own resolved tickets
    if (session.role === Role.FARMER) {
      if (ticket.authorId !== session.id) {
        return NextResponse.json({ error: "Forbidden" }, { status: 403 });
      }
      if (status && status !== "CLOSED") {
        return NextResponse.json(
          { error: "Farmers can only close tickets." },
          { status: 403 }
        );
      }
    } else if (!canManageContent(session.role)) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }

    const data: { status?: TicketStatus; priority?: string } = {};
    if (status) data.status = status as TicketStatus;
    if (priority && canManageContent(session.role)) data.priority = priority;

    const updated = await db.ticket.update({
      where: { id: params.id },
      data,
    });

    // Notify the ticket author about status change
    if (status && ticket.authorId !== session.id) {
      await db.notification.create({
        data: {
          userId: ticket.authorId,
          type: "ticket_status",
          title: `Ticket "${ticket.title.slice(0, 40)}" updated`,
          message: `Status changed to ${status.toLowerCase().replace("_", " ")}.`,
          link: `/tickets`,
        },
      });
    }

    return NextResponse.json({ ticket: updated });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
