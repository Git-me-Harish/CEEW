"use client";

import { useState, useEffect } from "react";
import {
  Plus,
  Syringe,
  Loader2,
  X,
  Send,
  AlertCircle,
  Clock,
  CheckCircle2,
  XCircle,
  Calendar,
  User as UserIcon,
  Stethoscope,
  Shield,
} from "lucide-react";
import { Section, SectionHeading } from "./Section";
import { useCurrentUser } from "@/hooks/use-current-user";

interface VaccinationRequest {
  id: string;
  cattleId: string;
  vaccineName: string;
  requestedDate: string;
  preferredTime: string | null;
  notes: string | null;
  status: "PENDING" | "APPROVED" | "REJECTED" | "SCHEDULED" | "COMPLETED" | "CANCELLED";
  reviewedAt: string | null;
  reviewNotes: string | null;
  scheduledDate: string | null;
  vetAssigned: string | null;
  createdAt: string;
  requester: { id: string; name: string; phone: string | null; location: string | null };
  cattle: { id: string; name: string; tagNumber: string; breed: string; species: string };
  reviewer: { id: string; name: string } | null;
}

interface Cattle {
  id: string;
  name: string;
  tagNumber: string;
  breed: string;
}

interface VaccineOption {
  name: string;
  disease: string;
  timing: string;
}

const statusConfig = {
  PENDING: { label: "Pending", color: "pill-amber", icon: Clock },
  APPROVED: { label: "Approved", color: "pill-blue", icon: CheckCircle2 },
  REJECTED: { label: "Rejected", color: "pill-clay", icon: XCircle },
  SCHEDULED: { label: "Scheduled", color: "pill-blue", icon: Calendar },
  COMPLETED: { label: "Completed", color: "pill-green", icon: CheckCircle2 },
  CANCELLED: { label: "Cancelled", color: "pill-outline", icon: XCircle },
};

export function VaccinationRequests() {
  const { user, loading, canManage } = useCurrentUser();
  const [requests, setRequests] = useState<VaccinationRequest[]>([]);
  const [vaccineOptions, setVaccineOptions] = useState<VaccineOption[]>([]);
  const [listLoading, setListLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [cattle, setCattle] = useState<Cattle[]>([]);
  const [filter, setFilter] = useState<"all" | "PENDING" | "APPROVED" | "SCHEDULED" | "COMPLETED" | "REJECTED">("all");
  const [reviewing, setReviewing] = useState<VaccinationRequest | null>(null);

  const load = async () => {
    setListLoading(true);
    try {
      const res = await fetch(`/api/vaccination-requests${filter !== "all" ? `?status=${filter}` : ""}`);
      if (res.ok) {
        const data = await res.json();
        setRequests(data.requests || []);
        setVaccineOptions(data.vaccineOptions || []);
      }
    } finally {
      setListLoading(false);
    }
  };

  useEffect(() => {
    if (user) load();
  }, [user, filter]);

  const loadCattle = async () => {
    const res = await fetch("/api/cattle");
    if (res.ok) {
      const data = await res.json();
      setCattle(data.cattle || []);
    }
  };

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
        <SectionHeading eyebrow="Authentication required" title="Sign in to manage vaccinations" />
        <div className="card-soft p-6 text-center">
          <p className="text-sm text-slate-600 mb-4">
            Sign in to request vaccinations for your cattle and track approval status.
          </p>
          <a href="/auth/signin" className="btn-primary inline-flex">Sign In</a>
        </div>
      </Section>
    );
  }

  if (reviewing && canManage) {
    return (
      <ReviewRequest
        request={reviewing}
        onClose={() => setReviewing(null)}
        onDone={() => {
          setReviewing(null);
          load();
        }}
      />
    );
  }

  return (
    <Section bg="white">
      <SectionHeading
        eyebrow="Vaccination Workflow"
        title={canManage ? "Review vaccination requests" : "Request a vaccination"}
        subtitle={
          canManage
            ? "Approve, reject, or schedule vaccination requests from farmers. Completed vaccinations automatically create a health record."
            : "Request a vaccination for your cattle. A veterinarian will review and approve it. You'll be notified at each step."
        }
        action={
          !canManage && (
            <button
              onClick={() => {
                loadCattle();
                setShowForm(true);
              }}
              className="btn-primary"
              disabled={cattle.length === 0 && showForm}
            >
              <Plus className="h-4 w-4" /> New Request
            </button>
          )
        }
      />

      {/* Filter tabs */}
      <div className="flex items-center gap-1 mb-6 border-b border-brand-line overflow-x-auto">
        {[
          { id: "all" as const, label: "All" },
          { id: "PENDING" as const, label: "Pending" },
          { id: "APPROVED" as const, label: "Approved" },
          { id: "SCHEDULED" as const, label: "Scheduled" },
          { id: "COMPLETED" as const, label: "Completed" },
          { id: "REJECTED" as const, label: "Rejected" },
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
      ) : requests.length === 0 ? (
        <div className="card-soft p-10 text-center">
          <Syringe className="h-10 w-10 mx-auto text-slate-300 mb-2" />
          <p className="text-sm text-slate-500">
            No vaccination requests {canManage ? "to review." : "yet. Click \"New Request\" to create one."}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {requests.map((r) => {
            const s = statusConfig[r.status];
            const StatusIcon = s.icon;
            return (
              <div key={r.id} className="card-soft p-5">
                <div className="flex items-start justify-between gap-3 mb-3 flex-wrap">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className={`${s.color} text-[10px] flex items-center gap-1`}>
                      <StatusIcon className="h-3 w-3" /> {s.label}
                    </span>
                    <span className="pill-blue text-[10px]">{r.cattle.breed}</span>
                    <span className="pill-outline text-[10px] capitalize">{r.cattle.species}</span>
                  </div>
                  <span className="text-[11px] text-slate-400">
                    Submitted {new Date(r.createdAt).toLocaleDateString("en-IN", { day: "numeric", month: "short" })}
                  </span>
                </div>

                <div className="grid md:grid-cols-2 gap-4 mb-3">
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-0.5">
                      Vaccine
                    </div>
                    <div className="text-sm font-bold text-brand-navy" style={{ fontFamily: "var(--font-montserrat)" }}>
                      {r.vaccineName}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-0.5">
                      Animal
                    </div>
                    <div className="text-sm text-brand-navy">
                      {r.cattle.name} <span className="text-slate-500">({r.cattle.tagNumber})</span>
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-0.5">
                      Requested Date
                    </div>
                    <div className="text-sm text-brand-navy">
                      {new Date(r.requestedDate).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" })}
                      {r.preferredTime && <span className="text-slate-500"> · {r.preferredTime}</span>}
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-0.5">
                      {canManage ? "Requested By" : "Reviewed By"}
                    </div>
                    <div className="text-sm text-brand-navy">
                      {canManage ? (
                        <span className="flex items-center gap-1.5">
                          <UserIcon className="h-3 w-3" />
                          {r.requester.name}
                          {r.requester.phone && <span className="text-slate-500">· {r.requester.phone}</span>}
                        </span>
                      ) : r.reviewer ? (
                        <span className="flex items-center gap-1.5">
                          <Stethoscope className="h-3 w-3" />
                          {r.reviewer.name}
                        </span>
                      ) : (
                        <span className="text-slate-400">Awaiting review</span>
                      )}
                    </div>
                  </div>
                </div>

                {r.notes && (
                  <div className="p-2.5 rounded-md bg-brand-mist text-xs text-slate-700 mb-2">
                    <strong className="text-brand-navy">Farmer note:</strong> {r.notes}
                  </div>
                )}

                {r.reviewNotes && (
                  <div className="p-2.5 rounded-md bg-blue-50 border border-blue-200 text-xs text-brand-navy mb-2">
                    <strong>Vet note:</strong> {r.reviewNotes}
                  </div>
                )}

                {r.scheduledDate && (
                  <div className="p-2.5 rounded-md bg-green-50 border border-green-200 text-xs text-brand-green-dark mb-2">
                    <Calendar className="h-3 w-3 inline mr-1" />
                    Scheduled for {new Date(r.scheduledDate).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" })}
                    {r.vetAssigned && <span> · Vet: {r.vetAssigned}</span>}
                  </div>
                )}

                {canManage && r.status === "PENDING" && (
                  <div className="mt-3 pt-3 border-t border-brand-line flex justify-end">
                    <button onClick={() => setReviewing(r)} className="btn-primary text-xs">
                      Review Request
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {showForm && (
        <NewRequestForm
          cattle={cattle}
          vaccineOptions={vaccineOptions}
          onClose={() => setShowForm(false)}
          onCreated={() => {
            setShowForm(false);
            load();
          }}
        />
      )}
    </Section>
  );
}

function NewRequestForm({
  cattle,
  vaccineOptions,
  onClose,
  onCreated,
}: {
  cattle: Cattle[];
  vaccineOptions: VaccineOption[];
  onClose: () => void;
  onCreated: () => void;
}) {
  const [form, setForm] = useState({
    cattleId: cattle[0]?.id || "",
    vaccineName: "",
    requestedDate: new Date(Date.now() + 7 * 86400000).toISOString().split("T")[0],
    preferredTime: "",
    notes: "",
  });
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (!form.cattleId || !form.vaccineName || !form.requestedDate) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch("/api/vaccination-requests", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to create request.");
      onCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create request.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-white rounded-lg shadow-2xl max-w-lg w-full p-6" onClick={(e) => e.stopPropagation()}>
        <h3 className="text-lg font-bold text-brand-navy mb-4" style={{ fontFamily: "var(--font-montserrat)" }}>
          Request a Vaccination
        </h3>

        {cattle.length === 0 ? (
          <div className="p-4 rounded-md bg-amber-50 border border-amber-200 text-sm text-amber-800">
            You need to register at least one cattle first. Go to the Milk Tracker tab to add cattle.
          </div>
        ) : (
          <div className="space-y-3">
            <div>
              <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                Select Animal
              </label>
              <select
                value={form.cattleId}
                onChange={(e) => setForm({ ...form, cattleId: e.target.value })}
                className="input-soft"
              >
                {cattle.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name} — {c.breed} ({c.tagNumber})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                Vaccine
              </label>
              <select
                value={form.vaccineName}
                onChange={(e) => setForm({ ...form, vaccineName: e.target.value })}
                className="input-soft"
              >
                <option value="">Select a vaccine...</option>
                {vaccineOptions.map((v) => (
                  <option key={v.name} value={v.name}>
                    {v.name} (for {v.disease})
                  </option>
                ))}
              </select>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                  Preferred Date
                </label>
                <input
                  type="date"
                  value={form.requestedDate}
                  onChange={(e) => setForm({ ...form, requestedDate: e.target.value })}
                  className="input-soft"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                  Preferred Time
                </label>
                <input
                  type="text"
                  value={form.preferredTime}
                  onChange={(e) => setForm({ ...form, preferredTime: e.target.value })}
                  className="input-soft"
                  placeholder="Morning / Evening"
                />
              </div>
            </div>
            <div>
              <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                Notes (optional)
              </label>
              <textarea
                value={form.notes}
                onChange={(e) => setForm({ ...form, notes: e.target.value })}
                className="input-soft"
                rows={2}
                placeholder="Any specific concerns or context for the vet..."
              />
            </div>
          </div>
        )}

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
            disabled={submitting || !form.cattleId || !form.vaccineName || !form.requestedDate}
            className="btn-primary"
          >
            {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            Submit Request
          </button>
        </div>
      </div>
    </div>
  );
}

function ReviewRequest({
  request,
  onClose,
  onDone,
}: {
  request: VaccinationRequest;
  onClose: () => void;
  onDone: () => void;
}) {
  const [status, setStatus] = useState<"APPROVED" | "REJECTED" | "SCHEDULED" | "COMPLETED">("APPROVED");
  const [reviewNotes, setReviewNotes] = useState("");
  const [scheduledDate, setScheduledDate] = useState("");
  const [vetAssigned, setVetAssigned] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const body: { status: string; reviewNotes: string; scheduledDate?: string; vetAssigned?: string } = {
        status,
        reviewNotes,
      };
      if (status === "SCHEDULED" || status === "COMPLETED") {
        if (scheduledDate) body.scheduledDate = scheduledDate;
        if (vetAssigned) body.vetAssigned = vetAssigned;
      }
      const res = await fetch(`/api/vaccination-requests/${request.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to update request.");
      onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to update request.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Section bg="white">
      <button onClick={onClose} className="text-xs text-slate-500 hover:text-brand-navy flex items-center gap-1 mb-4">
        <X className="h-3.5 w-3.5" /> Close review
      </button>

      <div className="card-soft p-6">
        <h2 className="text-lg font-bold text-brand-navy mb-1" style={{ fontFamily: "var(--font-montserrat)" }}>
          Review Vaccination Request
        </h2>
        <p className="text-xs text-slate-500 mb-5">Request #{request.id.slice(-6)}</p>

        <div className="grid md:grid-cols-2 gap-4 mb-5 p-4 rounded-md bg-brand-mist">
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-0.5">Farmer</div>
            <div className="text-sm text-brand-navy font-semibold">{request.requester.name}</div>
            <div className="text-xs text-slate-500">
              {request.requester.phone && <span>{request.requester.phone} · </span>}
              {request.requester.location || "No location"}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-0.5">Animal</div>
            <div className="text-sm text-brand-navy font-semibold">
              {request.cattle.name} ({request.cattle.tagNumber})
            </div>
            <div className="text-xs text-slate-500">{request.cattle.breed} · {request.cattle.species}</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-0.5">Vaccine</div>
            <div className="text-sm text-brand-navy font-semibold">{request.vaccineName}</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-0.5">Requested Date</div>
            <div className="text-sm text-brand-navy">
              {new Date(request.requestedDate).toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" })}
              {request.preferredTime && <span className="text-slate-500"> · {request.preferredTime}</span>}
            </div>
          </div>
        </div>

        {request.notes && (
          <div className="mb-5 p-3 rounded-md bg-amber-50 border border-amber-200 text-sm text-amber-900">
            <strong>Farmer note:</strong> {request.notes}
          </div>
        )}

        <div className="space-y-3">
          <div>
            <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2 block">
              Decision
            </label>
            <div className="grid grid-cols-4 gap-2">
              {([
                { id: "APPROVED" as const, label: "Approve", icon: CheckCircle2, color: "border-brand-green text-brand-green-dark" },
                { id: "SCHEDULED" as const, label: "Schedule", icon: Calendar, color: "border-brand-blue text-brand-blue" },
                { id: "COMPLETED" as const, label: "Complete", icon: CheckCircle2, color: "border-brand-green text-brand-green-dark" },
                { id: "REJECTED" as const, label: "Reject", icon: XCircle, color: "border-red-400 text-red-600" },
              ]).map((opt) => (
                <button
                  key={opt.id}
                  onClick={() => setStatus(opt.id)}
                  className={`p-2.5 rounded-md border-2 text-center transition-all ${
                    status === opt.id ? `${opt.color} bg-current/5` : "border-brand-line text-slate-500 hover:border-brand-navy"
                  }`}
                >
                  <opt.icon className="h-4 w-4 mx-auto mb-1" />
                  <div className="text-xs font-semibold" style={{ fontFamily: "var(--font-montserrat)" }}>
                    {opt.label}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {(status === "SCHEDULED" || status === "COMPLETED") && (
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                  {status === "COMPLETED" ? "Completed On" : "Scheduled For"}
                </label>
                <input
                  type="datetime-local"
                  value={scheduledDate}
                  onChange={(e) => setScheduledDate(e.target.value)}
                  className="input-soft"
                />
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                  Vet Assigned
                </label>
                <input
                  type="text"
                  value={vetAssigned}
                  onChange={(e) => setVetAssigned(e.target.value)}
                  className="input-soft"
                  placeholder="Dr. Name"
                />
              </div>
            </div>
          )}

          <div>
            <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
              Review Notes (sent to farmer)
            </label>
            <textarea
              value={reviewNotes}
              onChange={(e) => setReviewNotes(e.target.value)}
              className="input-soft"
              rows={3}
              placeholder="Message for the farmer — e.g., 'Approved. Please bring the animal to the clinic on the scheduled date.'"
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
          <button onClick={submit} disabled={submitting} className="btn-primary">
            {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Shield className="h-4 w-4" />}
            Submit Decision
          </button>
        </div>
      </div>
    </Section>
  );
}
