import { getServerSession } from "next-auth";
import { authOptions } from "./auth-options";
import { Role } from "@prisma/client";

export type SessionUser = {
  id: string;
  email: string;
  name: string;
  role: Role;
};

export async function getSession(): Promise<SessionUser | null> {
  const session = await getServerSession(authOptions);
  if (!session?.user) return null;
  const u = session.user as { id?: string; email?: string; name?: string; role?: string };
  if (!u.id || !u.role) return null;
  return {
    id: u.id,
    email: u.email || "",
    name: u.name || "",
    role: u.role as Role,
  };
}

export async function requireAuth(): Promise<SessionUser> {
  const s = await getSession();
  if (!s) throw new Error("Unauthorized");
  return s;
}

export async function requireRole(...roles: Role[]): Promise<SessionUser> {
  const s = await requireAuth();
  if (!roles.includes(s.role)) throw new Error("Forbidden: insufficient role");
  return s;
}

export function canManageContent(role: Role): boolean {
  return role === Role.VET || role === Role.ADMIN;
}

export function isAdmin(role: Role): boolean {
  return role === Role.ADMIN;
}
