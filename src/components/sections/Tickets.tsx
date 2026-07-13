"use client";

import { useState, useEffect } from "react";
import {
  Plus,
  MessageSquare,
  Loader2,
  X,
  Send,
  ArrowLeft,
  AlertCircle,
  Clock,
  CheckCircle2,
  User as UserIcon,
  Stethoscope,
  Shield,
} from "lucide-react";
import { Section, SectionHeading } from "./Section";
import { useCurrentUser } from "@/hooks/use-current-user";

interface TicketAuthor {
  id: string;
  name: string;
  role: string;
  location: string | null;
}

interface Ticket {
  id: string;
  authorId: string;
  title: string;
  body: string;
  category: string;
  status: "OPEN" | "IN_PROGRESS" | "RESOLVED" | "CLOSED";
  priority: "LOW" | "MEDIUM" | "HIGH" | "URGENT";
  breedTag: string | null;
  cattleTag: string | null;
  createdAt: string;
  updatedAt: string;
  author: TicketAuthor;
  _count?: { replies: number };
}

interface TicketReply {
  id: string;
  ticketId: string;
  authorId: string;
  body: string;
  internal: boolean;
  createdAt: string;
  author: { id: string; name: string; role: string };
}

const statusConfig = {
  OPEN: { label: "Open", color: "pill-blue" },
  IN_PROGRESS: { label: "In Progress", color: "pill-amber" },
  RESOLVED: { label: "Resolved", color: "pill-green" },
  CLOSED: { label: "Closed", color: "pill-outline" },
};

const priorityConfig = {
  LOW: { label: "Low", color: "pill-outline" },
  MEDIUM: { label: "Medium", color: "pill-blue" },
  HIGH: { label: "High", color: "pill-amber" },
  URGENT: { label: "Urgent", color: "pill-clay" },
};

export function Tickets() {
  const { user, loading, canManage } = useCurrentUser();
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [listLoading, setListLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [selected, setSelected] = useState<Ticket | null>(null);
  const [filter, setFilter] = useState<"all" | "OPEN" | "IN_PROGRESS" | "RESOLVED" | "CLOSED">("all");

  const loadTickets = async () => {
    setListLoading(true);
    try {
      const res = await fetch(`/api/tickets${filter !== "all" ? `?status=${filter}` : ""}`);
      if (res.ok) {
        const data = await res.json();
        setTickets(data.tickets || []);
      }
    } finally {
      setListLoading(false);
    }
  };

  useEffect(() => {
    if (user) loadTickets();
  }, [user, filter]);

  if (loading) {
    return (
      <Section bg="white">
        <div className="text-center py-16">
          <Loader2 className="h-8 w-8 mx-auto text-brand-blue animate-spin" />
        </div>
      </Section>
    );
  }

  if (!user) {
    return (
      <Section bg="white">
        <SectionHeading eyebrow="Authentication required" title="Sign in to view problems" />
        <div className="card-soft p-6 text-center">
          <p className="text-sm text-slate-600 mb-4">
            You need to be signed in to post problems and view responses from veterinarians.
          </p>
          <a href="/auth/signin" className="btn-primary inline-flex">Sign In</a>
        </div>
      </Section>
    );
  }

  // Detail view
  if (selected) {
    return (
      <TicketDetail
        ticket={selected}
        canManage={!!canManage}
        currentUserId={user.id}
        onBack={() => {
          setSelected(null);
          loadTickets();
        }}
      />
    );
  }

  return (
    <Section bg="white">
      <SectionHeading
        eyebrow="Problem & Solution Hub"
        title="Post problems, get solutions"
        subtitle={
          canManage
            ? "Review farmer problems, respond with solutions, and manage ticket status. Internal notes are visible only to vets and admins."
            : "Post problems about your cattle — health, nutrition, breeding, or anything else. Veterinarians will respond with solutions."
        }
        action={
          <button onClick={() => setShowForm(true)} className="btn-primary">
            <Plus className="h-4 w-4" /> Post a Problem
          </button>
        }
      />

      {/* Filter tabs */}
      <div className="flex items-center gap-1 mb-6 border-b border-brand-line overflow-x-auto">
        {[
          { id: "all" as const, label: "All" },
          { id: "OPEN" as const, label: "Open" },
          { id: "IN_PROGRESS" as const, label: "In Progress" },
          { id: "RESOLVED" as const, label: "Resolved" },
          { id: "CLOSED" as const, label: "Closed" },
        ].map((t) => (
          <button
            key={t.id}
            onClick={() => setFilter(t.id)}
            className={`px-4 py-2.5 text-sm font-medium border-b-2 -mb-px whitespace-nowrap transition-colors ${
              filter === t.id
                ? "border-brand-navy text-brand-navy"
                : "border-transparent text-slate-500 hover:text-brand-navy"
            }`}
            style={{ fontFamily: "var(--font-montserrat)" }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {listLoading ? (
        <div className="text-center py-16">
          <Loader2 className="h-8 w-8 mx-auto text-brand-blue animate-spin" />
        </div>
      ) : tickets.length === 0 ? (
        <div className="card-soft p-10 text-center">
          <MessageSquare className="h-10 w-10 mx-auto text-slate-300 mb-2" />
          <p className="text-sm text-slate-500">No problems found. Click "Post a Problem" to create one.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {tickets.map((t) => {
            const s = statusConfig[t.status];
            const p = priorityConfig[t.priority];
            return (
              <button
                key={t.id}
                onClick={() => setSelected(t)}
                className="card-soft p-5 text-left hover:shadow-md transition-shadow w-full"
              >
                <div className="flex items-start justify-between gap-3 mb-2 flex-wrap">
                  <div className="flex items-center gap-2 flex-wrap">
                    {t.breedTag && <span className="pill-blue text-[10px]">{t.breedTag}</span>}
                    <span className={`${p.color} text-[10px]`}>{p.label}</span>
                    <span className={`${s.color} text-[10px]`}>{s.label}</span>
                  </div>
                  <span className="text-[11px] text-slate-400">
                    {new Date(t.createdAt).toLocaleDateString("en-IN", { day: "numeric", month: "short" })}
                  </span>
                </div>
                <h3
                  className="text-base font-bold text-brand-navy mb-1"
                  style={{ fontFamily: "var(--font-montserrat)" }}
                >
                  {t.title}
                </h3>
                <p className="text-sm text-slate-600 line-clamp-2 mb-3">{t.body}</p>
                <div className="flex items-center justify-between text-[11px] text-slate-500">
                  <div className="flex items-center gap-2">
                    <RoleIcon role={t.author.role} />
                    <span className="font-semibold text-brand-navy">{t.author.name}</span>
                    <span>·</span>
                    <span>{t.author.location || t.author.role.toLowerCase()}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    {t._count && t._count.replies > 0 && (
                      <span className="flex items-center gap-1">
                        <MessageSquare className="h-3 w-3" /> {t._count.replies}
                      </span>
                    )}
                    <span className="capitalize">{t.category}</span>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      )}

      {showForm && (
        <TicketForm
          onClose={() => setShowForm(false)}
          onCreated={() => {
            setShowForm(false);
            loadTickets();
          }}
        />
      )}
    </Section>
  );
}

function RoleIcon({ role }: { role: string }) {
  if (role === "VET")
    return <Stethoscope className="h-3 w-3 text-brand-green" />;
  if (role === "ADMIN")
    return <Shield className="h-3 w-3 text-brand-navy" />;
  return <UserIcon className="h-3 w-3 text-brand-blue" />;
}

function TicketForm({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const [form, setForm] = useState({
    title: "",
    body: "",
    category: "general",
    priority: "MEDIUM",
    breedTag: "",
    cattleTag: "",
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (!form.title || !form.body) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch("/api/tickets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to create ticket.");
      onCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create ticket.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-white rounded-lg shadow-2xl max-w-lg w-full p-6" onClick={(e) => e.stopPropagation()}>
        <h3 className="text-lg font-bold text-brand-navy mb-4" style={{ fontFamily: "var(--font-montserrat)" }}>
          Post a Problem
        </h3>
        <div className="space-y-3">
          <div>
            <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
              Title
            </label>
            <input
              type="text"
              value={form.title}
              onChange={(e) => setForm({ ...form, title: e.target.value })}
              className="input-soft"
              placeholder="My Gir cow's yield dropped suddenly..."
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                Category
              </label>
              <select
                value={form.category}
                onChange={(e) => setForm({ ...form, category: e.target.value })}
                className="input-soft"
              >
                <option value="general">General</option>
                <option value="health">Health</option>
                <option value="nutrition">Nutrition</option>
                <option value="breeding">Breeding</option>
                <option value="market">Market</option>
                <option value="other">Other</option>
              </select>
            </div>
            <div>
              <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                Priority
              </label>
              <select
                value={form.priority}
                onChange={(e) => setForm({ ...form, priority: e.target.value })}
                className="input-soft"
              >
                <option value="LOW">Low</option>
                <option value="MEDIUM">Medium</option>
                <option value="HIGH">High</option>
                <option value="URGENT">Urgent</option>
              </select>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                Breed (optional)
              </label>
              <input
                type="text"
                value={form.breedTag}
                onChange={(e) => setForm({ ...form, breedTag: e.target.value })}
                className="input-soft"
                placeholder="Gir, Murrah, etc."
              />
            </div>
            <div>
              <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                Cattle Tag (optional)
              </label>
              <input
                type="text"
                value={form.cattleTag}
                onChange={(e) => setForm({ ...form, cattleTag: e.target.value })}
                className="input-soft"
                placeholder="IND-GIR-001"
              />
            </div>
          </div>
          <div>
            <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
              Describe the problem
            </label>
            <textarea
              value={form.body}
              onChange={(e) => setForm({ ...form, body: e.target.value })}
              className="input-soft"
              rows={4}
              placeholder="Describe the issue in detail — symptoms, when it started, what you've tried..."
            />
          </div>
        </div>
        {error && (
          <div className="mt-3 flex items-start gap-2 p-2.5 rounded-md bg-red-50 border border-red-200 text-xs text-red-700">
            <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
            {error}
          </div>
        )}
        <div className="mt-5 flex items-center justify-end gap-2">
          <button onClick={onClose} className="btn-secondary">Cancel</button>
          <button
            onClick={submit}
            disabled={submitting || !form.title || !form.body}
            className="btn-primary"
          >
            {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            Post Problem
          </button>
        </div>
      </div>
    </div>
  );
}

function TicketDetail({
  ticket,
  canManage,
  currentUserId,
  onBack,
}: {
  ticket: Ticket;
  canManage: boolean;
  currentUserId: string;
  onBack: () => void;
}) {
  const [replies, setReplies] = useState<TicketReply[]>([]);
  const [replyBody, setReplyBody] = useState("");
  const [internal, setInternal] = useState(false);
  const [markResolved, setMarkResolved] = useState(false);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [status, setStatus] = useState(ticket.status);

  const load = async () => {
    setLoading(true);
    try {
      const res = await fetch(`/api/tickets/${ticket.id}`);
      if (res.ok) {
        const data = await res.json();
        setReplies(data.ticket.replies || []);
        setStatus(data.ticket.status);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [ticket.id]);

  const submitReply = async () => {
    if (!replyBody.trim()) return;
    setSubmitting(true);
    try {
      const res = await fetch(`/api/tickets/${ticket.id}/reply`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          body: replyBody,
          internal,
          markResolved: canManage && markResolved,
        }),
      });
      if (res.ok) {
        setReplyBody("");
        setInternal(false);
        setMarkResolved(false);
        await load();
      }
    } finally {
      setSubmitting(false);
    }
  };

  const updateStatus = async (newStatus: "OPEN" | "IN_PROGRESS" | "RESOLVED" | "CLOSED") => {
    const res = await fetch(`/api/tickets/${ticket.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: newStatus }),
    });
    if (res.ok) {
      setStatus(newStatus);
    }
  };

  const s = statusConfig[status];
  const p = priorityConfig[ticket.priority];

  return (
    <Section bg="white">
      <button
        onClick={onBack}
        className="text-xs text-slate-500 hover:text-brand-navy flex items-center gap-1 mb-4"
      >
        <ArrowLeft className="h-3.5 w-3.5" /> Back to all problems
      </button>

      {/* Ticket header */}
      <div className="card-soft p-6 mb-4">
        <div className="flex items-center gap-2 flex-wrap mb-3">
          {ticket.breedTag && <span className="pill-blue text-[10px]">{ticket.breedTag}</span>}
          <span className={`${p.color} text-[10px]`}>{p.label}</span>
          <span className={`${s.color} text-[10px]`}>{s.label}</span>
          <span className="pill-outline text-[10px] capitalize">{ticket.category}</span>
        </div>
        <h1
          className="text-2xl font-bold text-brand-navy mb-3"
          style={{ fontFamily: "var(--font-montserrat)" }}
        >
          {ticket.title}
        </h1>
        <div className="flex items-center gap-2 text-xs text-slate-500 mb-4">
          <RoleIcon role={ticket.author.role} />
          <span className="font-semibold text-brand-navy">{ticket.author.name}</span>
          <span>·</span>
          <span>{ticket.author.location || ticket.author.role.toLowerCase()}</span>
          <span>·</span>
          <Clock className="h-3 w-3" />
          <span>{new Date(ticket.createdAt).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" })}</span>
        </div>
        <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">{ticket.body}</p>

        {/* Status management */}
        {canManage && (
          <div className="mt-4 pt-4 border-t border-brand-line flex items-center gap-2 flex-wrap">
            <span className="text-xs text-slate-500 font-semibold">Update status:</span>
            {(["OPEN", "IN_PROGRESS", "RESOLVED", "CLOSED"] as const).map((st) => (
              <button
                key={st}
                onClick={() => updateStatus(st)}
                disabled={status === st}
                className={`text-xs px-2.5 py-1 rounded-md border transition-colors ${
                  status === st
                    ? "bg-brand-navy text-white border-brand-navy"
                    : "bg-white text-slate-600 border-brand-line hover:border-brand-navy"
                }`}
              >
                {statusConfig[st].label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Replies */}
      <div className="card-soft p-6">
        <h3
          className="text-sm font-semibold text-brand-navy mb-4"
          style={{ fontFamily: "var(--font-montserrat)" }}
        >
          Replies ({replies.length})
        </h3>

        {loading ? (
          <div className="text-center py-6">
            <Loader2 className="h-6 w-6 mx-auto text-brand-blue animate-spin" />
          </div>
        ) : replies.length === 0 ? (
          <p className="text-xs text-slate-500 text-center py-6">
            No replies yet. {canManage ? "Be the first to respond." : "A veterinarian will respond soon."}
          </p>
        ) : (
          <div className="space-y-3 mb-4">
            {replies.map((r) => (
              <div
                key={r.id}
                className={`p-3 rounded-md border ${
                  r.internal
                    ? "bg-amber-50 border-amber-200"
                    : r.authorId === currentUserId
                    ? "bg-brand-blue-50 border-brand-blue-100"
                    : "bg-brand-mist border-brand-line"
                }`}
              >
                <div className="flex items-center gap-2 mb-1.5">
                  <RoleIcon role={r.author.role} />
                  <span className="text-xs font-semibold text-brand-navy">{r.author.name}</span>
                  <span className="text-[10px] text-slate-500 capitalize">{r.author.role.toLowerCase()}</span>
                  {r.internal && (
                    <span className="pill-amber text-[9px]">Internal note</span>
                  )}
                  <span className="text-[10px] text-slate-400 ml-auto">
                    {new Date(r.createdAt).toLocaleString("en-IN", { dateStyle: "short", timeStyle: "short" })}
                  </span>
                </div>
                <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">{r.body}</p>
              </div>
            ))}
          </div>
        )}

        {/* Reply composer */}
        <div className="pt-4 border-t border-brand-line">
          <textarea
            value={replyBody}
            onChange={(e) => setReplyBody(e.target.value)}
            className="input-soft"
            rows={3}
            placeholder={canManage ? "Type your solution or advice..." : "Add more details or ask a follow-up..."}
          />
          <div className="mt-2 flex items-center justify-between gap-2 flex-wrap">
            <div className="flex items-center gap-3">
              {canManage && (
                <label className="flex items-center gap-1.5 text-xs text-slate-600 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={internal}
                    onChange={(e) => setInternal(e.target.checked)}
                    className="accent-brand-navy"
                  />
                  Internal note (vet/admin only)
                </label>
              )}
              {canManage && (
                <label className="flex items-center gap-1.5 text-xs text-slate-600 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={markResolved}
                    onChange={(e) => setMarkResolved(e.target.checked)}
                    className="accent-brand-navy"
                  />
                  Mark as resolved
                </label>
              )}
            </div>
            <button
              onClick={submitReply}
              disabled={submitting || !replyBody.trim()}
              className="btn-primary text-xs"
            >
              {submitting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Send className="h-3.5 w-3.5" />}
              Post Reply
            </button>
          </div>
        </div>
      </div>
    </Section>
  );
}
