"use client";

import { useState, useMemo } from "react";
import { Search, AlertTriangle, Shield, Syringe, Phone, MapPin, Clock, Activity, Stethoscope, ChevronRight, X } from "lucide-react";
import { Section, SectionHeading } from "./Section";
import { diseases, vaccineSchedule, type DiseaseCategory, type Disease } from "@/data/health";
import { vetDirectory } from "@/data/schemes";

const categoryColors: Record<DiseaseCategory, string> = {
  bacterial: "pill-clay",
  viral: "pill-amber",
  parasitic: "pill-blue",
  fungal: "pill-green",
  metabolic: "pill-navy",
  nutritional: "pill-outline",
  reproductive: "pill-outline",
};

const severityColors: Record<string, string> = {
  low: "pill-green",
  moderate: "pill-blue",
  high: "pill-amber",
  critical: "pill-clay",
};

export function HealthHub() {
  const [tab, setTab] = useState<"diseases" | "vaccines" | "vets">("diseases");
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<DiseaseCategory | "all">("all");
  const [selected, setSelected] = useState<Disease | null>(null);

  const filteredDiseases = useMemo(() => {
    return diseases.filter((d) => {
      if (query) {
        const q = query.toLowerCase();
        if (
          !d.name.toLowerCase().includes(q) &&
          !(d.localName?.toLowerCase().includes(q) ?? false) &&
          !d.symptoms.some((s) => s.toLowerCase().includes(q))
        )
          return false;
      }
      if (category !== "all" && d.category !== category) return false;
      return true;
    });
  }, [query, category]);

  return (
    <Section bg="mist">
      <SectionHeading
        eyebrow="Veterinary knowledge"
        title="Health, Disease & Vaccination Hub"
        subtitle="A farmer's reference to common bovine diseases — symptoms, causes, prevention, and treatment — plus the NDBB vaccination schedule and a directory of veterinary resources across India."
      />

      {/* Tabs */}
      <div className="flex items-center gap-1 mb-6 border-b border-brand-line overflow-x-auto">
        {[
          { id: "diseases" as const, label: `Disease Library (${diseases.length})`, icon: AlertTriangle },
          { id: "vaccines" as const, label: `Vaccination Schedule (${vaccineSchedule.length})`, icon: Syringe },
          { id: "vets" as const, label: `Veterinary Directory (${vetDirectory.length})`, icon: Stethoscope },
        ].map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`inline-flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 -mb-px whitespace-nowrap transition-colors ${
              tab === t.id
                ? "border-brand-navy text-brand-navy"
                : "border-transparent text-slate-500 hover:text-brand-navy"
            }`}
            style={{ fontFamily: "var(--font-montserrat)" }}
          >
            <t.icon className="h-4 w-4" />
            {t.label}
          </button>
        ))}
      </div>

      {/* Diseases tab */}
      {tab === "diseases" && (
        <div>
          {/* Filters */}
          <div className="card-soft p-4 mb-5">
            <div className="grid md:grid-cols-12 gap-3">
              <div className="md:col-span-7">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Search disease, symptom, or local name..."
                    className="input-soft pl-9"
                  />
                </div>
              </div>
              <div className="md:col-span-5">
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value as DiseaseCategory | "all")}
                  className="input-soft"
                >
                  <option value="all">All categories</option>
                  <option value="viral">Viral</option>
                  <option value="bacterial">Bacterial</option>
                  <option value="parasitic">Parasitic</option>
                  <option value="metabolic">Metabolic</option>
                  <option value="reproductive">Reproductive</option>
                </select>
              </div>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            {filteredDiseases.map((d) => (
              <DiseaseCard key={d.id} disease={d} onClick={() => setSelected(d)} />
            ))}
          </div>
          {filteredDiseases.length === 0 && (
            <div className="text-center py-12 card-soft">
              <Search className="h-10 w-10 mx-auto text-slate-300 mb-2" />
              <p className="text-sm text-slate-500">No diseases match your search.</p>
            </div>
          )}
        </div>
      )}

      {/* Vaccines tab */}
      {tab === "vaccines" && (
        <div className="grid gap-4">
          <div className="p-4 rounded-md bg-brand-blue-50 border border-brand-blue-100 flex items-start gap-3">
            <Shield className="h-5 w-5 text-brand-blue shrink-0 mt-0.5" />
            <div className="text-sm text-brand-navy">
              <strong>Prevention is cheaper than cure.</strong> Vaccinate your herd on schedule —
              FMD every 6 months, HS &amp; BQ annually before monsoon, Brucellosis once in female
              calves 4–8 months. Contact your local vet or call 1962 for free FMD vaccination under
              the National FMD Control Programme.
            </div>
          </div>
          {vaccineSchedule.map((v) => (
            <div key={v.id} className="card-soft p-5">
              <div className="flex items-start justify-between gap-3 mb-3 flex-wrap">
                <div>
                  <h3
                    className="text-base font-bold text-brand-navy"
                    style={{ fontFamily: "var(--font-montserrat)" }}
                  >
                    {v.vaccineName}
                  </h3>
                  <div className="text-sm text-slate-600">For: {v.disease}</div>
                </div>
                <span className="pill-green text-[10px]">Annual</span>
              </div>
              <div className="grid md:grid-cols-4 gap-3 mb-3">
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-0.5">
                    First dose
                  </div>
                  <div className="text-sm font-semibold text-brand-navy">{v.ageAtFirstDose}</div>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-0.5">
                    Booster
                  </div>
                  <div className="text-sm font-semibold text-brand-navy">{v.boosterInterval}</div>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-0.5">
                    Best timing
                  </div>
                  <div className="text-sm font-semibold text-brand-navy">{v.timing}</div>
                </div>
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-0.5">
                    Dose
                  </div>
                  <div className="text-sm font-semibold text-brand-navy">{v.dose}</div>
                </div>
              </div>
              <div className="pt-3 border-t border-brand-line">
                <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1">
                  Notes
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">{v.notes}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Vets tab */}
      {tab === "vets" && (
        <div className="grid md:grid-cols-2 gap-4">
          {vetDirectory.map((v) => (
            <div key={v.id} className="card-soft p-5">
              <div className="flex items-start justify-between gap-2 mb-2">
                <div>
                  <h3
                    className="text-base font-bold text-brand-navy"
                    style={{ fontFamily: "var(--font-montserrat)" }}
                  >
                    {v.name}
                  </h3>
                  <div className="text-xs text-slate-500 mt-0.5">{v.organisation}</div>
                </div>
                <span className="pill-blue text-[10px] capitalize">{v.type.replace("-", " ")}</span>
              </div>
              <div className="grid grid-cols-1 gap-2 text-xs mb-3">
                <div className="flex items-start gap-2 text-slate-600">
                  <MapPin className="h-3.5 w-3.5 mt-0.5 shrink-0 text-brand-blue" />
                  <span>
                    <strong className="text-brand-navy">{v.district}</strong> · {v.state}
                  </span>
                </div>
                <div className="flex items-start gap-2 text-slate-600">
                  <Phone className="h-3.5 w-3.5 mt-0.5 shrink-0 text-brand-blue" />
                  <span>{v.contact}</span>
                </div>
                <div className="flex items-start gap-2 text-slate-600">
                  <Clock className="h-3.5 w-3.5 mt-0.5 shrink-0 text-brand-blue" />
                  <span>{v.availability}</span>
                </div>
              </div>
              <div className="pt-3 border-t border-brand-line">
                <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1.5">
                  Services
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {v.services.map((s, i) => (
                    <span key={i} className="pill-outline text-[10px]">
                      {s}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {selected && <DiseaseDetailModal disease={selected} onClose={() => setSelected(null)} />}
    </Section>
  );
}

function DiseaseCard({ disease, onClick }: { disease: Disease; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="card-soft p-5 text-left hover:shadow-lg transition-shadow"
    >
      <div className="flex items-start justify-between gap-3 mb-2 flex-wrap">
        <div>
          <h3
            className="text-base font-bold text-brand-navy"
            style={{ fontFamily: "var(--font-montserrat)" }}
          >
            {disease.name}
          </h3>
          {disease.localName && (
            <div className="text-xs text-slate-500 mt-0.5">Local: {disease.localName}</div>
          )}
        </div>
        <div className="flex flex-wrap gap-1.5">
          <span className={`${categoryColors[disease.category]} text-[10px] capitalize`}>
            {disease.category}
          </span>
          <span className={`${severityColors[disease.severity]} text-[10px] capitalize`}>
            {disease.severity}
          </span>
          {disease.zoonotic && <span className="pill-clay text-[10px]">Zoonotic</span>}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs mb-3">
        <div>
          <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Incubation</div>
          <div className="font-semibold text-brand-navy">{disease.incubationDays}</div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Mortality</div>
          <div className="font-semibold text-brand-navy">{disease.mortalityRate}</div>
        </div>
      </div>
      <div className="text-xs text-slate-600 mb-3">
        <span className="font-semibold text-brand-navy">Key symptoms:</span>{" "}
        {disease.symptoms.slice(0, 3).join(", ")}...
      </div>
      <div className="flex items-center justify-between pt-3 border-t border-brand-line">
        <div className="flex items-center gap-1.5 text-[11px]">
          {disease.vaccineAvailable ? (
            <>
              <Shield className="h-3.5 w-3.5 text-brand-green" />
              <span className="text-brand-green-dark font-medium">Vaccine available</span>
            </>
          ) : (
            <>
              <AlertTriangle className="h-3.5 w-3.5 text-brand-amber" />
              <span className="text-brand-amber font-medium">No vaccine</span>
            </>
          )}
        </div>
        <div className="flex items-center gap-1 text-xs text-brand-blue font-medium">
          View details <ChevronRight className="h-3.5 w-3.5" />
        </div>
      </div>
    </button>
  );
}

function DiseaseDetailModal({ disease, onClose }: { disease: Disease; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="bg-brand-navy p-5 text-white">
          <button
            onClick={onClose}
            className="absolute top-3 right-3 h-9 w-9 rounded-full bg-white/15 text-white flex items-center justify-center hover:bg-white/25"
          >
            <X className="h-4 w-4" />
          </button>
          <div className="flex flex-wrap gap-1.5 mb-2">
            <span className="pill-amber text-[10px] capitalize">{disease.category}</span>
            <span
              className={`${
                severityColors[disease.severity]
              } text-[10px] capitalize`}
            >
              {disease.severity} severity
            </span>
            {disease.zoonotic && (
              <span className="pill-clay text-[10px]">Zoonotic risk</span>
            )}
          </div>
          <h2
            className="text-2xl font-bold"
            style={{ fontFamily: "var(--font-montserrat)" }}
          >
            {disease.name}
          </h2>
          {disease.localName && (
            <div className="text-sm text-white/70 mt-1">Commonly known as: {disease.localName}</div>
          )}
        </div>

        <div className="p-6 space-y-5">
          {/* Quick facts */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="p-3 rounded-md bg-brand-mist">
              <Clock className="h-4 w-4 text-brand-blue mb-1" />
              <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Incubation</div>
              <div className="text-sm font-bold text-brand-navy">{disease.incubationDays}</div>
            </div>
            <div className="p-3 rounded-md bg-brand-mist">
              <Activity className="h-4 w-4 text-brand-blue mb-1" />
              <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Mortality</div>
              <div className="text-sm font-bold text-brand-navy">{disease.mortalityRate}</div>
            </div>
            <div className="p-3 rounded-md bg-brand-mist">
              <Shield className="h-4 w-4 text-brand-blue mb-1" />
              <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Vaccine</div>
              <div className="text-sm font-bold text-brand-navy">
                {disease.vaccineAvailable ? "Available" : "Not available"}
              </div>
            </div>
            <div className="p-3 rounded-md bg-brand-mist">
              <Stethoscope className="h-4 w-4 text-brand-blue mb-1" />
              <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Affected</div>
              <div className="text-sm font-bold text-brand-navy capitalize">
                {disease.affectedSpecies.join(", ")}
              </div>
            </div>
          </div>

          {/* Symptoms */}
          <div>
            <h3
              className="text-sm font-semibold text-brand-navy mb-2"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              Symptoms to watch for
            </h3>
            <ul className="space-y-1.5">
              {disease.symptoms.map((s, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                  <span className="numbered-badge !h-5 !w-5 !text-[10px] shrink-0 mt-0.5">
                    {i + 1}
                  </span>
                  {s}
                </li>
              ))}
            </ul>
          </div>

          {/* Causes & transmission */}
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-3 rounded-md bg-brand-mist border border-brand-line">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1">
                Cause
              </div>
              <p className="text-xs text-slate-700 leading-relaxed">{disease.causes}</p>
            </div>
            <div className="p-3 rounded-md bg-brand-mist border border-brand-line">
              <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-1">
                Transmission
              </div>
              <p className="text-xs text-slate-700 leading-relaxed">{disease.transmission}</p>
            </div>
          </div>

          {/* Prevention */}
          <div>
            <h3
              className="text-sm font-semibold text-brand-navy mb-2 flex items-center gap-2"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              <Shield className="h-4 w-4 text-brand-green" /> Prevention
            </h3>
            <ul className="space-y-1.5">
              {disease.prevention.map((p, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                  <span className="text-brand-green shrink-0">●</span>
                  {p}
                </li>
              ))}
            </ul>
          </div>

          {/* Treatment */}
          <div>
            <h3
              className="text-sm font-semibold text-brand-navy mb-2 flex items-center gap-2"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              <Syringe className="h-4 w-4 text-brand-blue" /> Treatment
            </h3>
            <ul className="space-y-1.5">
              {disease.treatment.map((t, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                  <span className="text-brand-blue shrink-0">●</span>
                  {t}
                </li>
              ))}
            </ul>
          </div>

          {/* Disclaimer */}
          <div className="p-3 rounded-md bg-amber-50 border border-amber-200 text-xs text-amber-800 flex items-start gap-2">
            <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
            <span>
              <strong>Important:</strong> This is general advisory information. Always consult a
              registered veterinarian for diagnosis and treatment. Self-medication can harm your
              animals and create drug-resistant pathogens. Call 1962 for the veterinary helpline.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
