"use client";

import { useState, useEffect } from "react";
import { Plus, TrendingUp, TrendingDown, BarChart3, GlassWater, Droplet, Trash2, Loader2, Milk } from "lucide-react";
import { Section, SectionHeading } from "./Section";

interface Cattle {
  id: string;
  tagNumber: string;
  name: string;
  breed: string;
  species: string;
  birthDate: string;
  weightKg: number;
}

interface MilkLog {
  id: string;
  cattleId: string;
  date: string;
  morningKg: number;
  eveningKg: number;
  fatPct: number | null;
  cattle?: { name: string; tagNumber: string; breed: string };
}

export function MilkTracker() {
  const [cattle, setCattle] = useState<Cattle[]>([]);
  const [logs, setLogs] = useState<MilkLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({
    cattleId: "",
    date: new Date().toISOString().split("T")[0],
    morningKg: "",
    eveningKg: "",
    fatPct: "",
  });
  const [submitting, setSubmitting] = useState(false);

  const loadData = async () => {
    setLoading(true);
    try {
      const [cattleRes, logsRes] = await Promise.all([
        fetch("/api/cattle"),
        fetch("/api/milk-log?days=14"),
      ]);
      const cattleData = await cattleRes.json();
      const logsData = await logsRes.json();
      setCattle(cattleData.cattle || []);
      setLogs(logsData.logs || []);
      if (cattleData.cattle?.length > 0 && !form.cattleId) {
        setForm((f) => ({ ...f, cattleId: cattleData.cattle[0].id }));
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const submit = async () => {
    if (!form.cattleId || !form.morningKg || !form.eveningKg) return;
    setSubmitting(true);
    try {
      const res = await fetch("/api/milk-log", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (res.ok) {
        setForm((f) => ({ ...f, morningKg: "", eveningKg: "", fatPct: "" }));
        setShowForm(false);
        await loadData();
      }
    } finally {
      setSubmitting(false);
    }
  };

  // Compute stats
  const totalToday = logs
    .filter((l) => l.date.startsWith(new Date().toISOString().split("T")[0]))
    .reduce((s, l) => s + l.morningKg + l.eveningKg, 0);

  const total14 = logs.reduce((s, l) => s + l.morningKg + l.eveningKg, 0);
  const avgDaily = logs.length > 0 ? total14 / 14 : 0;

  // Build 14-day series for chart
  const series: { date: string; total: number }[] = [];
  for (let i = 13; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    const dateStr = d.toISOString().split("T")[0];
    const dayTotal = logs
      .filter((l) => l.date.startsWith(dateStr))
      .reduce((s, l) => s + l.morningKg + l.eveningKg, 0);
    series.push({ date: dateStr, total: dayTotal });
  }
  const maxVal = Math.max(...series.map((s) => s.total), 1);

  // Per-cattle breakdown
  const perCattle = cattle.map((c) => {
    const cLogs = logs.filter((l) => l.cattleId === c.id);
    const total = cLogs.reduce((s, l) => s + l.morningKg + l.eveningKg, 0);
    const avg = cLogs.length > 0 ? total / cLogs.length : 0;
    return { cattle: c, total, avg, logCount: cLogs.length };
  });

  return (
    <Section bg="mist">
      <SectionHeading
        eyebrow="Production analytics"
        title="Milk Production Tracker"
        subtitle="Log daily morning and evening yields per animal. Track 14-day trends, average production, and per-animal performance to make informed breeding and feeding decisions."
        action={
          <button onClick={() => setShowForm(true)} className="btn-primary">
            <Plus className="h-4 w-4" /> Log Milk
          </button>
        }
      />

      {loading ? (
        <div className="text-center py-16">
          <Loader2 className="h-8 w-8 mx-auto text-brand-blue animate-spin" />
        </div>
      ) : (
        <>
          {/* Stats cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <StatCard
              icon={Milk}
              label="Today's Total"
              value={`${totalToday.toFixed(1)} L`}
              accent="bg-brand-navy"
            />
            <StatCard
              icon={BarChart3}
              label="14-Day Average"
              value={`${avgDaily.toFixed(1)} L/day`}
              accent="bg-brand-blue"
            />
            <StatCard
              icon={Droplet}
              label="14-Day Total"
              value={`${total14.toFixed(1)} L`}
              accent="bg-brand-green"
            />
            <StatCard
              icon={GlassWater}
              label="Active Animals"
              value={`${cattle.length}`}
              accent="bg-brand-amber"
            />
          </div>

          <div className="grid lg:grid-cols-3 gap-6">
            {/* Trend chart */}
            <div className="lg:col-span-2 card-soft p-5">
              <h3
                className="text-sm font-semibold text-brand-navy mb-4"
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                14-Day Production Trend
              </h3>
              <div className="h-64 flex items-end gap-1.5">
                {series.map((s, i) => {
                  const height = (s.total / maxVal) * 100;
                  return (
                    <div key={i} className="flex-1 flex flex-col items-center gap-1.5 group">
                      <div className="text-[10px] text-slate-500 opacity-0 group-hover:opacity-100 transition-opacity font-medium">
                        {s.total.toFixed(1)}
                      </div>
                      <div className="w-full bg-brand-mist rounded-t-sm overflow-hidden flex-1 flex items-end">
                        <div
                          className="w-full bg-gradient-to-t from-brand-blue to-brand-amber rounded-t-sm transition-all hover:from-brand-navy hover:to-brand-blue"
                          style={{ height: `${Math.max(2, height)}%` }}
                        />
                      </div>
                      <div className="text-[9px] text-slate-400">
                        {new Date(s.date).getDate()}
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="mt-3 text-[11px] text-slate-500 flex items-center justify-between">
                <span>Last 14 days · Total litres per day</span>
                <span className="flex items-center gap-1.5">
                  <span className="h-2 w-2 bg-brand-blue rounded-sm" /> Daily yield
                </span>
              </div>
            </div>

            {/* Per-animal breakdown */}
            <div className="card-soft p-5">
              <h3
                className="text-sm font-semibold text-brand-navy mb-4"
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                Per-Animal Performance (14d)
              </h3>
              <div className="space-y-3">
                {perCattle.length === 0 ? (
                  <div className="text-xs text-slate-500 text-center py-6">
                    No cattle registered yet.
                  </div>
                ) : (
                  perCattle.map((p) => (
                    <div key={p.cattle.id} className="p-3 rounded-md bg-brand-mist border border-brand-line">
                      <div className="flex items-start justify-between mb-1.5">
                        <div>
                          <div className="text-sm font-semibold text-brand-navy" style={{ fontFamily: "var(--font-montserrat)" }}>
                            {p.cattle.name}
                          </div>
                          <div className="text-[10px] text-slate-500">
                            {p.cattle.breed} · {p.cattle.tagNumber}
                          </div>
                        </div>
                        <span className="text-xs font-bold text-brand-blue" style={{ fontFamily: "var(--font-montserrat)" }}>
                          {p.avg.toFixed(1)} L/day
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-[10px] text-slate-500">
                        <span>{p.logCount} logs · {p.total.toFixed(1)} L total</span>
                        {p.avg > 10 ? (
                          <TrendingUp className="h-3 w-3 text-brand-green" />
                        ) : (
                          <TrendingDown className="h-3 w-3 text-brand-amber" />
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Recent logs table */}
          <div className="mt-6 card-soft p-5">
            <h3
              className="text-sm font-semibold text-brand-navy mb-4"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              Recent Milk Logs
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-brand-line text-slate-500 uppercase text-[10px] tracking-wider">
                    <th className="text-left py-2 px-2 font-semibold">Date</th>
                    <th className="text-left py-2 px-2 font-semibold">Animal</th>
                    <th className="text-left py-2 px-2 font-semibold">Breed</th>
                    <th className="text-right py-2 px-2 font-semibold">Morning (L)</th>
                    <th className="text-right py-2 px-2 font-semibold">Evening (L)</th>
                    <th className="text-right py-2 px-2 font-semibold">Total</th>
                    <th className="text-right py-2 px-2 font-semibold">Fat %</th>
                  </tr>
                </thead>
                <tbody>
                  {logs.slice(0, 14).map((l) => (
                    <tr key={l.id} className="border-b border-brand-line/50">
                      <td className="py-2 px-2 text-slate-600">
                        {new Date(l.date).toLocaleDateString("en-IN", { day: "2-digit", month: "short" })}
                      </td>
                      <td className="py-2 px-2 text-brand-navy font-medium">
                        {l.cattle?.name || "—"}
                      </td>
                      <td className="py-2 px-2 text-slate-600">{l.cattle?.breed}</td>
                      <td className="py-2 px-2 text-right text-slate-600">{l.morningKg.toFixed(1)}</td>
                      <td className="py-2 px-2 text-right text-slate-600">{l.eveningKg.toFixed(1)}</td>
                      <td className="py-2 px-2 text-right font-semibold text-brand-navy">
                        {(l.morningKg + l.eveningKg).toFixed(1)}
                      </td>
                      <td className="py-2 px-2 text-right text-slate-600">
                        {l.fatPct ? l.fatPct.toFixed(1) : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {logs.length === 0 && (
              <div className="text-center py-8 text-sm text-slate-500">
                No milk logs yet. Click "Log Milk" to add your first entry.
              </div>
            )}
          </div>
        </>
      )}

      {/* Form modal */}
      {showForm && (
        <div
          className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
          onClick={() => setShowForm(false)}
        >
          <div
            className="bg-white rounded-lg shadow-2xl max-w-md w-full p-6"
            onClick={(e) => e.stopPropagation()}
          >
            <h3
              className="text-lg font-bold text-brand-navy mb-4"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              Log Daily Milk Yield
            </h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                  Animal
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
                  Date
                </label>
                <input
                  type="date"
                  value={form.date}
                  onChange={(e) => setForm({ ...form, date: e.target.value })}
                  className="input-soft"
                />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                    Morning (L)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={form.morningKg}
                    onChange={(e) => setForm({ ...form, morningKg: e.target.value })}
                    className="input-soft"
                    placeholder="5.5"
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                    Evening (L)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={form.eveningKg}
                    onChange={(e) => setForm({ ...form, eveningKg: e.target.value })}
                    className="input-soft"
                    placeholder="4.5"
                  />
                </div>
              </div>
              <div>
                <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-1.5 block">
                  Fat % (optional)
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={form.fatPct}
                  onChange={(e) => setForm({ ...form, fatPct: e.target.value })}
                  className="input-soft"
                  placeholder="4.5"
                />
              </div>
            </div>
            <div className="mt-5 flex items-center justify-end gap-2">
              <button onClick={() => setShowForm(false)} className="btn-secondary">
                Cancel
              </button>
              <button
                onClick={submit}
                disabled={submitting || !form.cattleId || !form.morningKg || !form.eveningKg}
                className="btn-primary"
              >
                {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
                Save Entry
              </button>
            </div>
          </div>
        </div>
      )}
    </Section>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  accent,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  accent: string;
}) {
  return (
    <div className="card-soft p-4 flex items-center gap-3">
      <div className={`h-10 w-10 rounded-md ${accent} flex items-center justify-center shrink-0`}>
        <Icon className="h-5 w-5 text-white" />
      </div>
      <div>
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
          {label}
        </div>
        <div className="text-base font-bold text-brand-navy" style={{ fontFamily: "var(--font-montserrat)" }}>
          {value}
        </div>
      </div>
    </div>
  );
}
