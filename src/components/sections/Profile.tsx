"use client";

import { Section, SectionHeading } from "./Section";
import { useCurrentUser } from "@/hooks/use-current-user";
import { Users, MessageSquare, Syringe, Bell, Settings, HelpCircle, LogOut } from "lucide-react";
import { signOut } from "next-auth/react";

export function Profile({ onNavigate }: { onNavigate: (tab: "tickets" | "vaccination" | "management" | "milk" | "dashboard" | "settings") => void }) {
  const { user, loading } = useCurrentUser();

  if (loading) {
    return (
      <Section bg="mist">
        <div className="text-center py-16">
          <div className="h-8 w-8 mx-auto border-2 border-brand-blue border-t-transparent rounded-full animate-spin" />
        </div>
      </Section>
    );
  }

  if (!user) {
    return (
      <Section bg="mist">
        <SectionHeading
          eyebrow="Authentication required"
          title="Please sign in to view your profile"
        />
        <div className="card-soft p-6 text-center">
          <p className="text-sm text-slate-600 mb-4">
            You need to be signed in to view your dashboard, post problems, request vaccinations, and manage your cattle.
          </p>
          <a href="/auth/signin" className="btn-primary inline-flex">
            Sign In
          </a>
        </div>
      </Section>
    );
  }

  const roleBadge = {
    FARMER: "pill-green",
    VET: "pill-blue",
    ADMIN: "pill-navy",
  }[user.role];

  const roleLabel = {
    FARMER: "Farmer",
    VET: "Veterinarian",
    ADMIN: "Administrator",
  }[user.role];

  return (
    <Section bg="mist">
      <SectionHeading
        eyebrow="My Profile"
        title={`Welcome, ${user.name.split(" ")[0]}`}
        subtitle="Your personal dashboard — cattle, problems, vaccination requests, and notifications."
      />

      {/* Profile header */}
      <div className="card-soft p-6 mb-6">
        <div className="flex items-start gap-4 flex-wrap">
          <div
            className="h-16 w-16 rounded-full bg-brand-navy flex items-center justify-center text-white text-2xl font-bold shrink-0"
            style={{ fontFamily: "var(--font-montserrat)" }}
          >
            {user.name.charAt(0).toUpperCase()}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap mb-1">
              <h2
                className="text-xl font-bold text-brand-navy"
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                {user.name}
              </h2>
              <span className={`${roleBadge}`}>{roleLabel}</span>
            </div>
            <div className="text-sm text-slate-600">{user.email}</div>
            <div className="text-xs text-slate-500 mt-1 flex items-center gap-4 flex-wrap">
              {user.phone && <span>{user.phone}</span>}
              {user.location && <span>{user.location}</span>}
              <span>
                Joined{" "}
                {new Date(user.createdAt).toLocaleDateString("en-IN", {
                  day: "numeric",
                  month: "short",
                  year: "numeric",
                })}
              </span>
            </div>
          </div>
          <button
            onClick={() => signOut({ callbackUrl: "/" })}
            className="btn-secondary text-xs"
          >
            <LogOut className="h-3.5 w-3.5" /> Sign Out
          </button>
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard label="My Cattle" value={user._count.cattle} icon={Users} color="bg-brand-navy" />
        <StatCard label="My Problems" value={user._count.tickets} icon={MessageSquare} color="bg-brand-blue" />
        <StatCard label="Vaccination Requests" value={user._count.vaccinationRequests} icon={Syringe} color="bg-brand-green" />
        <StatCard label="Unread Alerts" value={user._count.notifications} icon={Bell} color="bg-brand-amber" />
      </div>

      {/* Quick links */}
      <div className="grid md:grid-cols-2 gap-4">
        <div className="card-soft p-5">
          <h3
            className="text-sm font-semibold text-brand-navy mb-3 flex items-center gap-2"
            style={{ fontFamily: "var(--font-montserrat)" }}
          >
            <Settings className="h-4 w-4 text-brand-blue" /> Account Settings
          </h3>
          <p className="text-xs text-slate-600 mb-3">
            Update your profile, change password, or manage notification preferences.
          </p>
          <button onClick={() => onNavigate("settings")} className="btn-primary text-xs">
            <Settings className="h-3.5 w-3.5" /> Edit Profile & Settings
          </button>
        </div>

        <div className="card-soft p-5">
          <h3
            className="text-sm font-semibold text-brand-navy mb-3 flex items-center gap-2"
            style={{ fontFamily: "var(--font-montserrat)" }}
          >
            <HelpCircle className="h-4 w-4 text-brand-blue" /> Need Help?
          </h3>
          <p className="text-xs text-slate-600 mb-3">
            Post a problem to the community or contact a veterinarian directly.
          </p>
          <button onClick={() => onNavigate("tickets")} className="btn-primary text-xs">
            Post a Problem
          </button>
        </div>
      </div>
    </Section>
  );
}

function StatCard({
  label,
  value,
  icon: Icon,
  color,
}: {
  label: string;
  value: number;
  icon: React.ElementType;
  color: string;
}) {
  return (
    <div className="card-soft p-4 flex items-center gap-3">
      <div className={`h-10 w-10 rounded-md ${color} flex items-center justify-center shrink-0`}>
        <Icon className="h-5 w-5 text-white" />
      </div>
      <div>
        <div
          className="text-2xl font-bold text-brand-navy"
          style={{ fontFamily: "var(--font-montserrat)" }}
        >
          {value}
        </div>
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
          {label}
        </div>
      </div>
    </div>
  );
}
