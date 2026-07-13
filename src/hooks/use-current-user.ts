"use client";

import { useEffect, useState, useCallback } from "react";
import { useSession } from "next-auth/react";

export interface CurrentUser {
  id: string;
  email: string;
  name: string;
  role: "FARMER" | "VET" | "ADMIN";
  phone: string | null;
  location: string | null;
  avatarUrl: string | null;
  active: boolean;
  createdAt: string;
  _count: {
    cattle: number;
    tickets: number;
    vaccinationRequests: number;
    notifications: number; // unread count
  };
}

export function useCurrentUser() {
  const { status } = useSession();
  const [user, setUser] = useState<CurrentUser | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    if (status !== "authenticated") {
      setUser(null);
      setLoading(false);
      return;
    }
    try {
      const res = await fetch("/api/me");
      if (res.ok) {
        const data = await res.json();
        setUser(data.user);
      } else {
        setUser(null);
      }
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, [status]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return {
    user,
    loading: loading || status === "loading",
    isAuthenticated: !!user,
    isFarmer: user?.role === "FARMER",
    isVet: user?.role === "VET",
    isAdmin: user?.role === "ADMIN",
    canManage: user?.role === "VET" || user?.role === "ADMIN",
    refresh,
  };
}
