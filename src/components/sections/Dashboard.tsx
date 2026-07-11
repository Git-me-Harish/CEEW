"use client";

import { useEffect, useState } from "react";
import {
  Sparkles,
  BookOpen,
  HeartPulse,
  Calculator,
  Milk,
  TrendingUp,
  Users,
  ArrowRight,
  Sun,
  Droplet,
  Wind,
  Thermometer,
  Calendar,
} from "lucide-react";
import { Section, SectionHeading } from "./Section";
import { breedStats } from "@/data/breeds";
import { diseases } from "@/data/health";
import { govtSchemes, marketPrices } from "@/data/schemes";

interface DashboardProps {
  onNavigate: (tab: "classifier" | "encyclopedia" | "health" | "nutrition" | "milk" | "market" | "forum") => void;
}

export function Dashboard({ onNavigate }: DashboardProps) {
  const [stats, setStats] = useState<{ cattle: number; milkToday: number; logs14: number }>({
    cattle: 0,
    milkToday: 0,
    logs14: 0,
  });

  useEffect(() => {
    (async () => {
      try {
        const [cattleRes, logsRes] = await Promise.all([
          fetch("/api/cattle"),
          fetch("/api/milk-log?days=14"),
        ]);
        const cData = await cattleRes.json();
        const lData = await logsRes.json();
        const todayStr = new Date().toISOString().split("T")[0];
        const milkToday = (lData.logs || [])
          .filter((l: { date: string }) => l.date.startsWith(todayStr))
          .reduce((s: number, l: { morningKg: number; eveningKg: number }) => s + l.morningKg + l.eveningKg, 0);
        setStats({
          cattle: cData.cattle?.length || 0,
          milkToday,
          logs14: lData.logs?.length || 0,
        });
      } catch {
        // ignore
      }
    })();
  }, []);

  const quickActions = [
    {
      icon: Sparkles,
      title: "Identify a Breed",
      description: "Upload a photo — AI identifies the bovine breed in seconds.",
      tab: "classifier" as const,
      color: "bg-brand-amber",
    },
    {
      icon: BookOpen,
      title: "Browse Breed Library",
      description: `Explore ${breedStats.totalBreeds} Indian bovine breeds with full profiles.`,
      tab: "encyclopedia" as const,
      color: "bg-brand-blue",
    },
    {
      icon: HeartPulse,
      title: "Health & Vaccination",
      description: `${diseases.length} diseases, ${6} vaccine schedules, vet directory.`,
      tab: "health" as const,
      color: "bg-brand-green",
    },
    {
      icon: Calculator,
      title: "Nutrition Calculator",
      description: "Get a daily ration plan based on your animal's weight and yield.",
      tab: "nutrition" as const,
      color: "bg-brand-navy",
    },
    {
      icon: Milk,
      title: "Log Today's Milk",
      description: "Record morning and evening yields for each animal.",
      tab: "milk" as const,
      color: "bg-brand-clay",
    },
    {
      icon: TrendingUp,
      title: "Market & Schemes",
      description: `${marketPrices.length} mandi prices + ${govtSchemes.length} government schemes.`,
      tab: "market" as const,
      color: "bg-brand-blue",
    },
  ];

  return (
    <Section bg="mist">
      <SectionHeading
        eyebrow="Farmer Dashboard"
        title="Your operation at a glance"
        subtitle="Quick access to everything you need to manage your cattle or buffalo scientifically — herd status, quick actions, today's advisory, and market snapshot."
      />

      {/* Stats row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard
          icon={Users}
          label="Registered Animals"
          value={stats.cattle.toString()}
          sub="Cattle & buffalo"
          color="bg-brand-navy"
        />
        <StatCard
          icon={Milk}
          label="Today's Milk"
          value={`${stats.milkToday.toFixed(1)} L`}
          sub="All animals combined"
          color="bg-brand-blue"
        />
        <StatCard
          icon={Calendar}
          label="Logs (14 days)"
          value={stats.logs14.toString()}
          sub="Milk production entries"
          color="bg-brand-green"
        />
        <StatCard
          icon={BookOpen}
          label="Breed Library"
          value={breedStats.totalBreeds.toString()}
          sub={`${breedStats.endangered} endangered`}
          color="bg-brand-amber"
        />
      </div>

      {/* Quick actions */}
      <div className="mb-8">
        <h3
          className="text-sm font-semibold text-brand-navy mb-4"
          style={{ fontFamily: "var(--font-montserrat)" }}
        >
          Quick Actions
        </h3>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {quickActions.map((a, i) => (
            <button
              key={i}
              onClick={() => onNavigate(a.tab)}
              className="card-soft p-5 text-left hover:shadow-lg hover:-translate-y-0.5 transition-all group"
            >
              <div className="flex items-start justify-between mb-3">
                <div className={`h-10 w-10 rounded-md ${a.color} flex items-center justify-center`}>
                  <a.icon className="h-5 w-5 text-white" />
                </div>
                <ArrowRight className="h-4 w-4 text-slate-300 group-hover:text-brand-blue group-hover:translate-x-1 transition-all" />
              </div>
              <h4
                className="text-sm font-bold text-brand-navy mb-1"
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                {a.title}
              </h4>
              <p className="text-xs text-slate-600 leading-relaxed">{a.description}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Today's advisory + market snapshot */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Weather advisory */}
        <div className="card-soft p-5">
          <h3
            className="text-sm font-semibold text-brand-navy mb-4 flex items-center gap-2"
            style={{ fontFamily: "var(--font-montserrat)" }}
          >
            <Sun className="h-4 w-4 text-brand-amber" /> Today's Bovine Advisory
          </h3>
          <div className="grid grid-cols-3 gap-2 mb-4 text-xs">
            <div className="p-2.5 rounded-md bg-brand-mist text-center">
              <Thermometer className="h-3.5 w-3.5 mx-auto text-brand-blue mb-1" />
              <div className="font-bold text-brand-navy">28°C</div>
              <div className="text-[10px] text-slate-500">Avg temp</div>
            </div>
            <div className="p-2.5 rounded-md bg-brand-mist text-center">
              <Droplet className="h-3.5 w-3.5 mx-auto text-brand-blue mb-1" />
              <div className="font-bold text-brand-navy">68%</div>
              <div className="text-[10px] text-slate-500">Humidity</div>
            </div>
            <div className="p-2.5 rounded-md bg-brand-mist text-center">
              <Wind className="h-3.5 w-3.5 mx-auto text-brand-blue mb-1" />
              <div className="font-bold text-brand-navy">12 km/h</div>
              <div className="text-[10px] text-slate-500">Wind</div>
            </div>
          </div>
          <div className="space-y-2 text-xs">
            <div className="p-2.5 rounded-md bg-amber-50 border border-amber-200">
              <div className="font-semibold text-amber-800 mb-0.5">Heat stress watch</div>
              <p className="text-amber-700 leading-relaxed">
                Provide cool drinking water every 2 hours. Use fans and sprinklers in sheds. Avoid grazing between 11 AM – 4 PM.
              </p>
            </div>
            <div className="p-2.5 rounded-md bg-blue-50 border border-blue-200">
              <div className="font-semibold text-brand-blue mb-0.5">Vaccination reminder</div>
              <p className="text-brand-blue leading-relaxed">
                Next FMD booster due in 18 days. Schedule vet visit for whole herd.
              </p>
            </div>
            <div className="p-2.5 rounded-md bg-green-50 border border-green-200">
              <div className="font-semibold text-brand-green-dark mb-0.5">Fodder planning</div>
              <p className="text-brand-green-dark leading-relaxed">
                Berseem cutting ready in 5 days. Plan silage making for surplus green fodder.
              </p>
            </div>
          </div>
        </div>

        {/* Market snapshot */}
        <div className="card-soft p-5 lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h3
              className="text-sm font-semibold text-brand-navy flex items-center gap-2"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              <TrendingUp className="h-4 w-4 text-brand-green" /> Market Snapshot
            </h3>
            <button
              onClick={() => onNavigate("market")}
              className="text-xs text-brand-blue font-semibold hover:underline flex items-center gap-1"
            >
              View all <ArrowRight className="h-3 w-3" />
            </button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {marketPrices.slice(0, 6).map((p, i) => (
              <div key={i} className="p-3 rounded-md bg-brand-mist border border-brand-line">
                <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold mb-1 truncate">
                  {p.commodity}
                </div>
                <div
                  className="text-base font-bold text-brand-navy"
                  style={{ fontFamily: "var(--font-montserrat)" }}
                >
                  {p.modalPrice >= 1000
                    ? `₹${(p.modalPrice / 1000).toFixed(1)}k`
                    : `₹${p.modalPrice}`}
                  <span className="text-[10px] text-slate-500 font-normal">/{p.unit.replace("per ", "")}</span>
                </div>
                <div
                  className={`text-[10px] mt-0.5 ${
                    p.trend === "up"
                      ? "text-brand-green"
                      : p.trend === "down"
                      ? "text-red-500"
                      : "text-slate-500"
                  }`}
                >
                  {p.trend === "up" ? "▲" : p.trend === "down" ? "▼" : "→"} {Math.abs(p.changePct)}%
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
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
  value: string;
  sub: string;
  color: string;
}) {
  return (
    <div className="card-soft p-4">
      <div className="flex items-center justify-between mb-3">
        <div className={`h-9 w-9 rounded-md ${color} flex items-center justify-center`}>
          <Icon className="h-4 w-4 text-white" />
        </div>
      </div>
      <div className="text-2xl font-bold text-brand-navy" style={{ fontFamily: "var(--font-montserrat)" }}>
        {value}
      </div>
      <div className="text-xs font-semibold text-brand-navy mt-0.5">{label}</div>
      <div className="text-[10px] text-slate-500">{sub}</div>
    </div>
  );
}
