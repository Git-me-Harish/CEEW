"use client";

import { useState, useEffect } from "react";
import { Loader2, MessageSquare, Syringe, Users, Clock, AlertTriangle, CheckCircle2, ArrowRight, Bell, TrendingUp } from "lucide-react";
import { Section, SectionHeading } from "./Section";
import { useCurrentUser } from "@/hooks/use-current-user";

interface Ticket {
  id: string;
  title: string;
  status: "OPEN" | "IN_PROGRESS" | "RESOLVED" | "CLOSED";
  priority: "LOW" | "MEDIUM" | "HIGH" | "URGENT";
  createdAt: string;
  author: { name: string; location: string | null };
}

interface VaccinationRequest {
  id: string;
  vaccineName: string;
  status: "PENDING" | "APPROVED" | "REJECTED" | "SCHEDULED" | "COMPLETED" | "CANCELLED";
  requestedDate: string;
  requester: { name: string; phone: string | null; location: string | null };
  cattle: { name: string; tagNumber: string; breed: string };
}

interface Notification {
  id: string;
  type: string;
  title: string;
  message: string;
  read: boolean;
  createdAt: string;
  link: string | null;
}

interface ManagementProps {
  onNavigate: (tab: "tickets" | "vaccination") => void;
}

export function ManagementDashboard({ onNavigate }: ManagementProps) {
  const { user, loading, canManage } = useCurrentUser();
  const [pendingTickets, setPendingTickets] = useState<Ticket[]>([]);
  const [pendingVaccinations, setPendingVaccinations] = useState<VaccinationRequest[]>([]);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [stats, setStats] = useState({
    totalTickets: 0,
    openTickets: 0,
    urgentTickets: 0,
    pendingVaccinations: 0,
    totalUsers: 0,
    totalCattle: 0,
  });
  const [dataLoading, setDataLoading] = useState(true);

  const load = async () => {
    setDataLoading(true);
    try {
      const [tRes, vRes, nRes] = await Promise.all([
        fetch("/api/tickets?status=OPEN"),
        fetch("/api/vaccination-requests?status=PENDING"),
        fetch("/api/notifications?unread=true"),
      ]);

      if (tRes.ok) {
        const data = await tRes.json();
        const tickets = data.tickets || [];
        setPendingTickets(tickets.slice(0, 5));
        setStats((s) => ({
          ...s,
          totalTickets: tickets.length,
          openTickets: tickets.filter((t: Ticket) => t.status === "OPEN").length,
          urgentTickets: tickets.filter((t: Ticket) => t.priority === "URGENT").length,
        }));
      }

      if (vRes.ok) {
        const data = await vRes.json();
        const requests = data.requests || [];
        setPendingVaccinations(requests.slice(0, 5));
        setStats((s) => ({ ...s, pendingVaccinations: requests.length }));
      }

      if (nRes.ok) {
        const data = await nRes.json();
        setNotifications(data.notifications || []);
      }
    } finally {
      setDataLoading(false);
    }
  };

  useEffect(() => {
    if (canManage) load();
  }, [canManage]);

  if (loading) {
    return (
      <Section bg="mist">
        <div className="text-center py-16">
          <Loader2 className="h-8 w-8 mx-auto text-brand-blue animate-spin" />
        </div>
      </Section>
    );
  }

  if (!canManage) {
    return (
      <Section bg="mist">
        <SectionHeading eyebrow="Access restricted" title="Management dashboard" />
        <div className="card-soft p-6 text-center">
          <AlertTriangle className="h-10 w-10 mx-auto text-amber-500 mb-3" />
          <p className="text-sm text-slate-600 mb-4">
            This dashboard is only available to veterinarians and administrators.
            {user ? ` Your current role is ${user.role.toLowerCase()}.` : " Please sign in with a vet or admin account."}
          </p>
          {!user && (
            <a href="/auth/signin" className="btn-primary inline-flex">Sign In as Vet/Admin</a>
          )}
        </div>
      </Section>
    );
  }

  return (
    <Section bg="mist">
      <SectionHeading
        eyebrow="Management Dashboard"
        title={`Welcome, ${user?.name.split(" ")[0]} — here's what needs attention`}
        subtitle="Review pending farmer problems, approve vaccination requests, and stay on top of your workload."
      />

      {dataLoading ? (
        <div className="text-center py-16">
          <Loader2 className="h-8 w-8 mx-auto text-brand-blue animate-spin" />
        </div>
      ) : (
        <>
          {/* Stats row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <StatCard
              icon={MessageSquare}
              label="Open Tickets"
              value={stats.openTickets}
              sub={`${stats.urgentTickets} urgent`}
              color="bg-brand-blue"
            />
            <StatCard
              icon={Syringe}
              label="Pending Vaccinations"
              value={stats.pendingVaccinations}
              sub="Awaiting review"
              color="bg-brand-amber"
            />
            <StatCard
              icon={Bell}
              label="Unread Alerts"
              value={notifications.length}
              sub="New this session"
              color="bg-brand-navy"
            />
            <StatCard
              icon={TrendingUp}
              label="Total Open"
              value={stats.openTickets + stats.pendingVaccinations}
              sub="Items in queue"
              color="bg-brand-green"
            />
          </div>

          <div className="grid lg:grid-cols-2 gap-6">
            {/* Pending tickets */}
            <div className="card-soft p-5">
              <div className="flex items-center justify-between mb-4">
                <h3
                  className="text-sm font-semibold text-brand-navy flex items-center gap-2"
                  style={{ fontFamily: "var(--font-montserrat)" }}
                >
                  <MessageSquare className="h-4 w-4 text-brand-blue" /> Pending Problems
                </h3>
                <button
                  onClick={() => onNavigate("tickets")}
                  className="text-xs text-brand-blue font-semibold hover:underline flex items-center gap-1"
                >
                  View all <ArrowRight className="h-3 w-3" />
                </button>
              </div>

              {pendingTickets.length === 0 ? (
                <div className="text-center py-8 text-sm text-slate-500">
                  <CheckCircle2 className="h-8 w-8 mx-auto text-brand-green mb-2" />
                  No open problems. You're all caught up!
                </div>
              ) : (
                <div className="space-y-2">
                  {pendingTickets.map((t) => (
                    <button
                      key={t.id}
                      onClick={() => onNavigate("tickets")}
                      className="w-full text-left p-3 rounded-md bg-brand-mist hover:bg-brand-blue-50 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span
                          className={`text-[10px] ${
                            t.priority === "URGENT"
                              ? "pill-clay"
                              : t.priority === "HIGH"
                              ? "pill-amber"
                              : "pill-outline"
                          }`}
                        >
                          {t.priority.toLowerCase()}
                        </span>
                        <span className="text-[10px] text-slate-400">
                          {new Date(t.createdAt).toLocaleDateString("en-IN", { day: "numeric", month: "short" })}
                        </span>
                      </div>
                      <div className="text-sm font-semibold text-brand-navy line-clamp-1">{t.title}</div>
                      <div className="text-[11px] text-slate-500 mt-0.5">
                        {t.author.name} · {t.author.location || "Unknown"}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Pending vaccination requests */}
            <div className="card-soft p-5">
              <div className="flex items-center justify-between mb-4">
                <h3
                  className="text-sm font-semibold text-brand-navy flex items-center gap-2"
                  style={{ fontFamily: "var(--font-montserrat)" }}
                >
                  <Syringe className="h-4 w-4 text-brand-amber" /> Vaccination Queue
                </h3>
                <button
                  onClick={() => onNavigate("vaccination")}
                  className="text-xs text-brand-blue font-semibold hover:underline flex items-center gap-1"
                >
                  Review all <ArrowRight className="h-3 w-3" />
                </button>
              </div>

              {pendingVaccinations.length === 0 ? (
                <div className="text-center py-8 text-sm text-slate-500">
                  <CheckCircle2 className="h-8 w-8 mx-auto text-brand-green mb-2" />
                  No pending vaccination requests.
                </div>
              ) : (
                <div className="space-y-2">
                  {pendingVaccinations.map((r) => (
                    <button
                      key={r.id}
                      onClick={() => onNavigate("vaccination")}
                      className="w-full text-left p-3 rounded-md bg-brand-mist hover:bg-brand-amber/10 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="pill-amber text-[10px]">Pending</span>
                        <span className="text-[10px] text-slate-400">
                          For {new Date(r.requestedDate).toLocaleDateString("en-IN", { day: "numeric", month: "short" })}
                        </span>
                      </div>
                      <div className="text-sm font-semibold text-brand-navy line-clamp-1">
                        {r.vaccineName}
                      </div>
                      <div className="text-[11px] text-slate-500 mt-0.5">
                        {r.cattle.name} ({r.cattle.tagNumber}) · {r.requester.name}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Recent notifications */}
          {notifications.length > 0 && (
            <div className="card-soft p-5 mt-6">
              <h3
                className="text-sm font-semibold text-brand-navy mb-3 flex items-center gap-2"
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                <Bell className="h-4 w-4 text-brand-amber" /> Recent Notifications
              </h3>
              <div className="space-y-2">
                {notifications.slice(0, 5).map((n) => (
                  <div key={n.id} className="flex items-start gap-3 p-2.5 rounded-md bg-brand-mist">
                    <Clock className="h-3.5 w-3.5 text-brand-blue mt-0.5 shrink-0" />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-semibold text-brand-navy">{n.title}</div>
                      <div className="text-xs text-slate-600">{n.message}</div>
                      <div className="text-[10px] text-slate-400 mt-0.5">
                        {new Date(n.createdAt).toLocaleString("en-IN", { dateStyle: "short", timeStyle: "short" })}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </Section>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  sub,
  color,
}: {
  icon: React.ElementType;
  label: string;
  value: number;
  sub: string;
  color: string;
}) {
  return (
    <div className="card-soft p-4 flex items-center gap-3">
      <div className={`h-10 w-10 rounded-md ${color} flex items-center justify-center shrink-0`}>
        <Icon className="h-5 w-5 text-white" />
      </div>
      <div>
        <div className="text-2xl font-bold text-brand-navy" style={{ fontFamily: "var(--font-montserrat)" }}>
          {value}
        </div>
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">{label}</div>
        <div className="text-[10px] text-slate-400">{sub}</div>
      </div>
    </div>
  );
}
