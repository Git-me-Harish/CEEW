import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { getSession, canManageContent } from "@/lib/auth";
import { Role, TicketStatus } from "@prisma/client";

export const runtime = "nodejs";

// GET /api/tickets — list tickets (farmers see own; vets/admins see all)
export async function GET(req: NextRequest) {
  try {
    const session = await getSession();
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const { searchParams } = new URL(req.url);
    const status = searchParams.get("status") as TicketStatus | null;
    const mine = searchParams.get("mine") === "true";

    const where: {
      authorId?: string;
      status?: TicketStatus;
    } = {};

    // Farmers only see their own tickets; vets/admins see all (unless mine=true)
    if (session.role === Role.FARMER || mine) {
      where.authorId = session.id;
    }
    if (status) where.status = status;

    const tickets = await db.ticket.findMany({
      where,
      orderBy: { updatedAt: "desc" },
      include: {
        author: {
          select: { id: true, name: true, role: true, location: true },
        },
        _count: { select: { replies: true } },
      },
      take: 100,
    });

    return NextResponse.json({ tickets, canManage: canManageContent(session.role) });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

// POST /api/tickets — create a new ticket (farmers, vets, admins can all create)
export async function POST(req: NextRequest) {
  try {
    const session = await getSession();
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const { title, body: ticketBody, category, priority, breedTag, cattleTag } = body;

    if (!title || !ticketBody) {
      return NextResponse.json(
        { error: "Title and body are required." },
        { status: 400 }
      );
    }

    const ticket = await db.ticket.create({
      data: {
        authorId: session.id,
        title: title.trim(),
        body: ticketBody.trim(),
        category: category || "general",
        priority: priority || "MEDIUM",
        breedTag: breedTag || null,
        cattleTag: cattleTag || null,
      },
      include: {
        author: { select: { id: true, name: true, role: true, location: true } },
      },
    });

    // Notify all vets/admins about the new ticket
    const managers = await db.user.findMany({
      where: { role: { in: [Role.VET, Role.ADMIN] }, active: true },
      select: { id: true },
    });
    if (managers.length > 0) {
      await db.notification.createMany({
        data: managers.map((m) => ({
          userId: m.id,
          type: "ticket_new",
          title: `New ticket: ${title.trim().slice(0, 60)}`,
          message: `${session.name} posted a ${priority || "MEDIUM"} priority problem.`,
          link: `/tickets`,
        })),
      });
    }

    return NextResponse.json({ ticket });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
