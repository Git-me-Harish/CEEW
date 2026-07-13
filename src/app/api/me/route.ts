import { NextRequest, NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import { getSession } from "@/lib/auth";
import { db } from "@/lib/db";

export const runtime = "nodejs";

export async function GET() {
  try {
    const session = await getSession();
    if (!session) {
      return NextResponse.json({ user: null }, { status: 200 });
    }

    const user = await db.user.findUnique({
      where: { id: session.id },
      select: {
        id: true,
        email: true,
        name: true,
        role: true,
        phone: true,
        location: true,
        avatarUrl: true,
        active: true,
        createdAt: true,
        _count: {
          select: {
            cattle: true,
            tickets: true,
            vaccinationRequests: true,
            notifications: { where: { read: false } },
          },
        },
      },
    });

    if (!user) {
      return NextResponse.json({ user: null }, { status: 200 });
    }

    return NextResponse.json({ user });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

// PATCH /api/me — update profile (name, phone, location, avatarUrl) and/or change password
export async function PATCH(req: NextRequest) {
  try {
    const session = await getSession();
    if (!session) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const { name, phone, location, avatarUrl, currentPassword, newPassword } = body;

    // If changing password, verify current password first
    if (newPassword) {
      if (typeof newPassword !== "string" || newPassword.length < 6) {
        return NextResponse.json(
          { error: "New password must be at least 6 characters." },
          { status: 400 }
        );
      }
      if (!currentPassword) {
        return NextResponse.json(
          { error: "Current password is required to change password." },
          { status: 400 }
        );
      }

      const currentUser = await db.user.findUnique({
        where: { id: session.id },
        select: { passwordHash: true },
      });
      if (!currentUser) {
        return NextResponse.json({ error: "User not found." }, { status: 404 });
      }

      const valid = await bcrypt.compare(currentPassword, currentUser.passwordHash);
      if (!valid) {
        return NextResponse.json(
          { error: "Current password is incorrect." },
          { status: 400 }
        );
      }
    }

    // Build update data (only include fields that were provided)
    const data: {
      name?: string;
      phone?: string | null;
      location?: string | null;
      avatarUrl?: string | null;
      passwordHash?: string;
    } = {};

    if (typeof name === "string" && name.trim()) {
      data.name = name.trim();
    }
    if (phone !== undefined) {
      data.phone = typeof phone === "string" && phone.trim() ? phone.trim() : null;
    }
    if (location !== undefined) {
      data.location = typeof location === "string" && location.trim() ? location.trim() : null;
    }
    if (avatarUrl !== undefined) {
      data.avatarUrl = typeof avatarUrl === "string" && avatarUrl.trim() ? avatarUrl.trim() : null;
    }
    if (newPassword) {
      data.passwordHash = await bcrypt.hash(newPassword, 10);
    }

    if (Object.keys(data).length === 0) {
      return NextResponse.json({ error: "No fields to update." }, { status: 400 });
    }

    const updated = await db.user.update({
      where: { id: session.id },
      data,
      select: {
        id: true,
        email: true,
        name: true,
        role: true,
        phone: true,
        location: true,
        avatarUrl: true,
        active: true,
        createdAt: true,
        _count: {
          select: {
            cattle: true,
            tickets: true,
            vaccinationRequests: true,
            notifications: { where: { read: false } },
          },
        },
      },
    });

    return NextResponse.json({ user: updated, ok: true });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("Update profile API error:", message);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
