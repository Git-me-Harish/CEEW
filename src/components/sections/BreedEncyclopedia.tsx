"use client";

import { useState, useMemo } from "react";
import { Search, Filter, X, MapPin, Droplet, Scale, Thermometer, Shield, ChevronRight, ArrowLeft } from "lucide-react";
import { Section, SectionHeading } from "./Section";
import { breeds, breedStats, type Breed, type BreedCategory, type BovineType, type BreedUse } from "@/data/breeds";

export function BreedEncyclopedia() {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<BreedCategory | "all">("all");
  const [type, setType] = useState<BovineType | "all">("all");
  const [use, setUse] = useState<BreedUse | "all">("all");
  const [selected, setSelected] = useState<Breed | null>(null);

  const filtered = useMemo(() => {
    return breeds.filter((b) => {
      if (query) {
        const q = query.toLowerCase();
        if (
          !b.name.toLowerCase().includes(q) &&
          !b.origin.toLowerCase().includes(q) &&
          !b.region.toLowerCase().includes(q) &&
          !b.description.toLowerCase().includes(q)
        )
          return false;
      }
      if (category !== "all" && b.category !== category) return false;
      if (type !== "all" && b.type !== type) return false;
      if (use !== "all" && b.primaryUse !== use) return false;
      return true;
    });
  }, [query, category, type, use]);

  const clearFilters = () => {
    setQuery("");
    setCategory("all");
    setType("all");
    setUse("all");
  };

  return (
    <Section bg="white">
      <SectionHeading
        eyebrow="Knowledge base"
        title="Indian Bovine Breed Library"
        subtitle={`Explore ${breedStats.totalBreeds} bovine breeds — ${breedStats.indigenousCattle} indigenous cattle, ${breedStats.buffaloes} buffalo breeds, and ${breedStats.exotic} exotic breeds used in Indian crossbreeding programs.`}
        action={
          <div className="hidden sm:flex items-center gap-2 text-xs">
            <span className="pill-green">{breedStats.indigenousCattle} Indigenous</span>
            <span className="pill-blue">{breedStats.buffaloes} Buffalo</span>
            <span className="pill-amber">{breedStats.exotic} Exotic</span>
            <span className="pill-outline">{breedStats.endangered} Endangered</span>
          </div>
        }
      />

      {/* Filter bar */}
      <div className="card-soft p-4 mb-6">
        <div className="grid md:grid-cols-12 gap-3">
          <div className="md:col-span-5">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search by name, region, or trait..."
                className="input-soft pl-9"
              />
            </div>
          </div>
          <div className="md:col-span-2">
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value as BreedCategory | "all")}
              className="input-soft"
            >
              <option value="all">All categories</option>
              <option value="indigenous">Indigenous</option>
              <option value="exotic">Exotic</option>
            </select>
          </div>
          <div className="md:col-span-2">
            <select
              value={type}
              onChange={(e) => setType(e.target.value as BovineType | "all")}
              className="input-soft"
            >
              <option value="all">All species</option>
              <option value="cattle">Cattle</option>
              <option value="buffalo">Buffalo</option>
            </select>
          </div>
          <div className="md:col-span-2">
            <select
              value={use}
              onChange={(e) => setUse(e.target.value as BreedUse | "all")}
              className="input-soft"
            >
              <option value="all">All uses</option>
              <option value="dairy">Dairy</option>
              <option value="dual">Dual-purpose</option>
              <option value="draught">Draught</option>
            </select>
          </div>
          <div className="md:col-span-1">
            <button
              onClick={clearFilters}
              className="btn-secondary w-full text-xs"
              disabled={!query && category === "all" && type === "all" && use === "all"}
            >
              <X className="h-3.5 w-3.5" /> Clear
            </button>
          </div>
        </div>
        <div className="mt-3 text-xs text-slate-500 flex items-center gap-2">
          <Filter className="h-3.5 w-3.5" />
          Showing <strong className="text-brand-navy">{filtered.length}</strong> of {breeds.length} breeds
        </div>
      </div>

      {/* Breed grid */}
      {filtered.length === 0 ? (
        <div className="text-center py-16 card-soft">
          <Search className="h-10 w-10 mx-auto text-slate-300 mb-2" />
          <p className="text-slate-500 text-sm">No breeds match your filters. Try clearing them.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((breed) => (
            <BreedCard key={breed.id} breed={breed} onClick={() => setSelected(breed)} />
          ))}
        </div>
      )}

      {/* Detail modal */}
      {selected && <BreedDetailModal breed={selected} onClose={() => setSelected(null)} />}
    </Section>
  );
}

function BreedCard({ breed, onClick }: { breed: Breed; onClick: () => void }) {
  const statusColor =
    breed.conservationStatus === "endangered"
      ? "pill-clay"
      : breed.conservationStatus === "threatened"
      ? "pill-amber"
      : "pill-green";

  return (
    <button
      onClick={onClick}
      className="card-soft overflow-hidden text-left hover:shadow-lg hover:-translate-y-0.5 transition-all group"
    >
      {/* Image placeholder area */}
      <div
        className="h-44 relative image-placeholder"
        style={{
          background:
            breed.type === "buffalo"
              ? "linear-gradient(135deg, #1A2332 0%, #2D5A87 100%)"
              : breed.category === "exotic"
              ? "linear-gradient(135deg, #4CAF50 0%, #2F7E33 100%)"
              : "linear-gradient(135deg, #F2A93B 0%, #B5651D 100%)",
        }}
      >
        <div className="absolute inset-0 flex flex-col items-center justify-center text-white/80">
          <div
            className="text-base font-bold tracking-tight"
            style={{ fontFamily: "var(--font-montserrat)" }}
          >
            {breed.name}
          </div>
          <div className="text-[10px] uppercase tracking-widest text-white/60 mt-1">
            /breeds/{breed.id}.jpg
          </div>
        </div>
        <div className="absolute top-2 left-2 flex gap-1.5">
          <span className="pill-navy text-[10px]">{breed.type}</span>
        </div>
        <div className="absolute top-2 right-2">
          <span className={`${statusColor} text-[10px] capitalize`}>{breed.conservationStatus}</span>
        </div>
      </div>

      {/* Card body */}
      <div className="p-4">
        <div className="flex items-start justify-between gap-2 mb-2">
          <div>
            <h3
              className="text-base font-bold text-brand-navy"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              {breed.name}
            </h3>
            <div className="flex items-center gap-1 text-xs text-slate-500">
              <MapPin className="h-3 w-3" />
              {breed.region.split(",")[0]}
            </div>
          </div>
          <span className="pill-blue text-[10px] capitalize">{breed.primaryUse}</span>
        </div>
        <p className="text-xs text-slate-600 line-clamp-2 leading-relaxed mb-3">
          {breed.description}
        </p>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="p-2 rounded bg-brand-mist">
            <div className="text-slate-500 text-[10px] uppercase tracking-wider">Milk yield</div>
            <div className="font-semibold text-brand-navy">{breed.milkYieldKgPerLactation} kg</div>
          </div>
          <div className="p-2 rounded bg-brand-mist">
            <div className="text-slate-500 text-[10px] uppercase tracking-wider">Fat content</div>
            <div className="font-semibold text-brand-navy">{breed.fatContent}%</div>
          </div>
        </div>
        <div className="mt-3 flex items-center justify-end text-xs text-brand-blue font-medium group-hover:gap-2 gap-1 transition-all">
          View full profile <ChevronRight className="h-3.5 w-3.5" />
        </div>
      </div>
    </button>
  );
}

function BreedDetailModal({ breed, onClose }: { breed: Breed; onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Hero image placeholder */}
        <div
          className="h-56 md:h-64 relative"
          style={{
            background:
              breed.type === "buffalo"
                ? "linear-gradient(135deg, #1A2332 0%, #2D5A87 100%)"
                : breed.category === "exotic"
                ? "linear-gradient(135deg, #4CAF50 0%, #2F7E33 100%)"
                : "linear-gradient(135deg, #F2A93B 0%, #B5651D 100%)",
          }}
        >
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white/80">
            <div
              className="text-2xl font-bold"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              {breed.name}
            </div>
            <div className="text-[11px] uppercase tracking-widest text-white/60 mt-2">
              Drop image at /public/breeds/{breed.id}.jpg
            </div>
          </div>
          <button
            onClick={onClose}
            className="absolute top-3 right-3 h-9 w-9 rounded-full bg-black/50 text-white flex items-center justify-center hover:bg-black/70"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6">
          <button
            onClick={onClose}
            className="text-xs text-slate-500 hover:text-brand-navy flex items-center gap-1 mb-4"
          >
            <ArrowLeft className="h-3.5 w-3.5" /> Back to library
          </button>

          <div className="flex items-start justify-between gap-3 mb-2 flex-wrap">
            <h2
              className="text-2xl font-bold text-brand-navy"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              {breed.name}
            </h2>
            <div className="flex flex-wrap gap-1.5">
              <span className="pill-blue capitalize">{breed.type}</span>
              <span className="pill-green capitalize">{breed.category}</span>
              <span className="pill-amber capitalize">{breed.primaryUse}</span>
              <span
                className={`${
                  breed.conservationStatus === "endangered"
                    ? "pill-clay"
                    : breed.conservationStatus === "threatened"
                    ? "pill-amber"
                    : "pill-green"
                } capitalize`}
              >
                {breed.conservationStatus}
              </span>
            </div>
          </div>

          <p className="text-sm text-slate-600 mb-5 leading-relaxed">{breed.description}</p>

          {/* Quick stats grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
            <StatCard
              icon={Droplet}
              label="Milk / lactation"
              value={`${breed.milkYieldKgPerLactation} kg`}
            />
            <StatCard icon={Scale} label="Fat content" value={`${breed.fatContent}%`} />
            <StatCard
              icon={Thermometer}
              label="Heat tolerance"
              value={breed.heatTolerance}
              capitalize
            />
            <StatCard
              icon={Shield}
              label="Disease resistance"
              value={breed.diseaseResistance}
              capitalize
            />
          </div>

          {/* Detail rows */}
          <div className="grid md:grid-cols-2 gap-x-6 gap-y-3 mb-5">
            <DetailRow label="Origin" value={breed.origin} />
            <DetailRow label="Native region" value={breed.region} />
            <DetailRow label="Body weight" value={breed.avgBodyWeightKg} />
            <DetailRow label="Coat colour" value={breed.color} />
            <DetailRow label="Horn type" value={breed.hornType} />
            <DetailRow label="Temperament" value={breed.temperament} />
            <DetailRow label="Lactation length" value={`${breed.lactationDays} days`} />
            <DetailRow label="Primary use" value={breed.primaryUse} capitalize />
          </div>

          {/* Distinguishing features */}
          <div className="mb-5">
            <h3
              className="text-sm font-semibold text-brand-navy mb-2"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              Distinguishing features
            </h3>
            <ul className="space-y-1.5">
              {breed.distinguishingFeatures.map((f, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                  <span className="numbered-badge !h-5 !w-5 !text-[10px] shrink-0 mt-0.5">
                    {i + 1}
                  </span>
                  {f}
                </li>
              ))}
            </ul>
          </div>

          {/* Footer note */}
          <div className="p-3 rounded-md bg-brand-blue-50 border border-brand-blue-100 text-xs text-brand-navy">
            <strong>Conservation note:</strong> This breed is currently classified as{" "}
            <span className="font-semibold capitalize">{breed.conservationStatus}</span>. Support
            indigenous breed conservation through Rashtriya Gokul Mission and state breeding programs.
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  capitalize,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  capitalize?: boolean;
}) {
  return (
    <div className="p-3 rounded-md bg-brand-mist border border-brand-line">
      <Icon className="h-4 w-4 text-brand-blue mb-1.5" />
      <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
        {label}
      </div>
      <div
        className={`text-sm font-bold text-brand-navy ${capitalize ? "capitalize" : ""}`}
        style={{ fontFamily: "var(--font-montserrat)" }}
      >
        {value}
      </div>
    </div>
  );
}

function DetailRow({
  label,
  value,
  capitalize,
}: {
  label: string;
  value: string;
  capitalize?: boolean;
}) {
  return (
    <div className="flex flex-col py-2 border-b border-brand-line">
      <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold mb-0.5">
        {label}
      </div>
      <div className={`text-sm text-brand-navy ${capitalize ? "capitalize" : ""}`}>{value}</div>
    </div>
  );
}
