"use client";

import { useState } from "react";
import { Calculator, RotateCcw, TrendingUp, Droplet, Wheat, Beef, GlassWater, Snowflake, Leaf, Baby } from "lucide-react";
import { Section, SectionHeading } from "./Section";
import {
  feedingStandards,
  feedstuffs,
  calculateFeed,
  advisoryThresholds,
  type AnimalClass,
} from "@/data/nutrition";

export function NutritionCalculator() {
  const [bodyWeight, setBodyWeight] = useState(400);
  const [milkYield, setMilkYield] = useState(10);
  const [fatPct, setFatPct] = useState(4.5);
  const [animalClass, setAnimalClass] = useState<AnimalClass>("lactating-cow");
  const [monthsPregnant, setMonthsPregnant] = useState(0);

  const result = calculateFeed({
    bodyWeightKg: bodyWeight,
    milkYieldKgDay: milkYield,
    fatPct,
    animalClass,
    monthsPregnant,
  });

  const reset = () => {
    setBodyWeight(400);
    setMilkYield(10);
    setFatPct(4.5);
    setAnimalClass("lactating-cow");
    setMonthsPregnant(0);
  };

  return (
    <Section bg="white">
      <SectionHeading
        eyebrow="ICAR-NIANP based"
        title="Nutrition & Feed Calculator"
        subtitle="Get a daily ration plan for your cattle or buffalo based on body weight, milk yield, fat %, and physiological stage. Aligned with Indian Council of Agricultural Research feeding standards."
      />

      <div className="grid lg:grid-cols-5 gap-6">
        {/* Inputs */}
        <div className="lg:col-span-2">
          <div className="card-soft p-5">
            <div className="flex items-center justify-between mb-4">
              <h3
                className="text-sm font-semibold text-brand-navy"
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                Animal Details
              </h3>
              <button onClick={reset} className="text-xs text-slate-500 hover:text-brand-navy flex items-center gap-1">
                <RotateCcw className="h-3.5 w-3.5" /> Reset
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2 block">
                  Animal Class
                </label>
                <select
                  value={animalClass}
                  onChange={(e) => setAnimalClass(e.target.value as AnimalClass)}
                  className="input-soft"
                >
                  {feedingStandards.map((s) => (
                    <option key={s.class} value={s.class}>
                      {s.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Body Weight
                  </label>
                  <span className="text-sm font-bold text-brand-navy" style={{ fontFamily: "var(--font-montserrat)" }}>
                    {bodyWeight} kg
                  </span>
                </div>
                <input
                  type="range"
                  min={50}
                  max={800}
                  step={10}
                  value={bodyWeight}
                  onChange={(e) => setBodyWeight(parseInt(e.target.value))}
                  className="w-full accent-brand-blue"
                />
                <div className="flex justify-between text-[10px] text-slate-400 mt-1">
                  <span>50 kg (calf)</span>
                  <span>800 kg (Jaffarabadi bull)</span>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Daily Milk Yield
                  </label>
                  <span className="text-sm font-bold text-brand-navy" style={{ fontFamily: "var(--font-montserrat)" }}>
                    {milkYield} L/day
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={40}
                  step={0.5}
                  value={milkYield}
                  onChange={(e) => setMilkYield(parseFloat(e.target.value))}
                  className="w-full accent-brand-blue"
                />
                <div className="flex justify-between text-[10px] text-slate-400 mt-1">
                  <span>Dry / 0 L</span>
                  <span>High yielder / 40 L</span>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Milk Fat %
                  </label>
                  <span className="text-sm font-bold text-brand-navy" style={{ fontFamily: "var(--font-montserrat)" }}>
                    {fatPct}%
                  </span>
                </div>
                <input
                  type="range"
                  min={3}
                  max={9}
                  step={0.1}
                  value={fatPct}
                  onChange={(e) => setFatPct(parseFloat(e.target.value))}
                  className="w-full accent-brand-blue"
                />
                <div className="flex justify-between text-[10px] text-slate-400 mt-1">
                  <span>Cow (3.5%)</span>
                  <span>Buffalo (8.5%)</span>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                    Months Pregnant
                  </label>
                  <span className="text-sm font-bold text-brand-navy" style={{ fontFamily: "var(--font-montserrat)" }}>
                    {monthsPregnant} mo
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={9}
                  step={1}
                  value={monthsPregnant}
                  onChange={(e) => setMonthsPregnant(parseInt(e.target.value))}
                  className="w-full accent-brand-blue"
                />
                <div className="flex justify-between text-[10px] text-slate-400 mt-1">
                  <span>Open</span>
                  <span>9 mo (due)</span>
                </div>
              </div>
            </div>

            <div className="mt-5 pt-4 border-t border-brand-line p-3 rounded-md bg-brand-blue-50 text-xs text-brand-navy">
              <strong>Note:</strong> Calculator provides maintenance + production + pregnancy
              requirements. Adjust ration based on actual animal condition, climate, and feed
              quality. Always provide mineral mixture and salt free choice.
            </div>
          </div>
        </div>

        {/* Results */}
        <div className="lg:col-span-3 space-y-4">
          {/* Daily ration */}
          <div className="card-soft p-5">
            <div className="flex items-center justify-between mb-4">
              <h3
                className="text-sm font-semibold text-brand-navy flex items-center gap-2"
                style={{ fontFamily: "var(--font-montserrat)" }}
              >
                <Calculator className="h-4 w-4 text-brand-blue" />
                Recommended Daily Ration
              </h3>
              <div className="text-xs text-slate-500">
                Est. cost:{" "}
                <span className="font-bold text-brand-navy" style={{ fontFamily: "var(--font-montserrat)" }}>
                  ₹{result.estimatedDailyCost}/day
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
              <RationCard
                icon={Leaf}
                label="Green Fodder"
                value={`${result.greenFodderKg} kg`}
                color="bg-brand-green"
              />
              <RationCard
                icon={Wheat}
                label="Dry Roughage"
                value={`${result.dryRoughageKg} kg`}
                color="bg-brand-amber"
              />
              <RationCard
                icon={Beef}
                label="Concentrate Mix"
                value={`${result.concentrateKg} kg`}
                color="bg-brand-blue"
              />
              <RationCard
                icon={GlassWater}
                label="Drinking Water"
                value={`${result.waterL} L`}
                color="bg-brand-navy"
              />
            </div>

            <div className="space-y-2">
              {result.breakdown.map((item, i) => (
                <div
                  key={i}
                  className="flex items-start gap-3 p-2.5 rounded-md bg-brand-mist border border-brand-line"
                >
                  <div className="text-sm font-bold text-brand-navy w-32 shrink-0" style={{ fontFamily: "var(--font-montserrat)" }}>
                    {item.quantity}
                  </div>
                  <div className="text-xs">
                    <div className="font-medium text-brand-navy">{item.item}</div>
                    <div className="text-slate-500 leading-relaxed">{item.reason}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Nutrient requirements */}
          <div className="card-soft p-5">
            <h3
              className="text-sm font-semibold text-brand-navy mb-3"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              Daily Nutrient Requirements
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <NutrientCard label="Dry Matter" value={`${result.dmRequiredKg} kg`} sub="Total feed intake" />
              <NutrientCard label="DCP" value={`${result.dcpRequiredKg} kg`} sub="Digestible crude protein" />
              <NutrientCard label="TDN" value={`${result.tdnRequiredKg} kg`} sub="Total digestible nutrients" />
              <NutrientCard label="Calcium" value={`${result.calciumRequiredG} g`} sub="Bone & milk" />
              <NutrientCard label="Phosphorus" value={`${result.phosphorusRequiredG} g`} sub="Metabolic functions" />
              <NutrientCard label="Mineral Mix" value={`${result.mineralMixtureG} g`} sub="Trace minerals" />
              <NutrientCard label="Salt" value={`${result.saltG} g`} sub="Sodium source" />
              <NutrientCard label="Vit A" value="10000 IU" sub="For vision & immunity" />
            </div>
          </div>

          {/* Feedstuffs reference */}
          <div className="card-soft p-5">
            <h3
              className="text-sm font-semibold text-brand-navy mb-3"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              Common Indian Feedstuffs — Composition Reference
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-brand-line text-slate-500 uppercase text-[10px] tracking-wider">
                    <th className="text-left py-2 px-2 font-semibold">Feedstuff</th>
                    <th className="text-right py-2 px-2 font-semibold">DM %</th>
                    <th className="text-right py-2 px-2 font-semibold">CP %</th>
                    <th className="text-right py-2 px-2 font-semibold">TDN %</th>
                    <th className="text-right py-2 px-2 font-semibold">₹/kg</th>
                  </tr>
                </thead>
                <tbody>
                  {feedstuffs.slice(0, 10).map((f) => (
                    <tr key={f.id} className="border-b border-brand-line/50">
                      <td className="py-2 px-2 text-brand-navy font-medium">{f.name}</td>
                      <td className="py-2 px-2 text-right text-slate-600">{f.dmPct}</td>
                      <td className="py-2 px-2 text-right text-slate-600">{f.cpPct}</td>
                      <td className="py-2 px-2 text-right text-slate-600">{f.tdnPct}</td>
                      <td className="py-2 px-2 text-right text-slate-600">₹{f.costPerKg}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-2 text-[11px] text-slate-500">
              DM = Dry Matter · CP = Crude Protein · TDN = Total Digestible Nutrients
            </div>
          </div>

          {/* Advisory */}
          <div className="card-soft p-5">
            <h3
              className="text-sm font-semibold text-brand-navy mb-3 flex items-center gap-2"
              style={{ fontFamily: "var(--font-montserrat)" }}
            >
              <Snowflake className="h-4 w-4 text-brand-blue" /> Seasonal &amp; Condition-Based Advisory
            </h3>
            <div className="space-y-2">
              {advisoryThresholds.map((a, i) => (
                <div key={i} className="p-3 rounded-md bg-brand-mist border border-brand-line">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-semibold text-brand-navy" style={{ fontFamily: "var(--font-montserrat)" }}>
                      {a.condition}
                    </span>
                    <span className="text-[10px] text-slate-500 font-mono">{a.threshold}</span>
                  </div>
                  <p className="text-xs text-slate-600 leading-relaxed">{a.advisory}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </Section>
  );
}

function RationCard({
  icon: Icon,
  label,
  value,
  color,
}: {
  icon: React.ElementType;
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className={`p-3 rounded-md ${color} text-white`}>
      <Icon className="h-4 w-4 mb-1.5 opacity-80" />
      <div className="text-[10px] uppercase tracking-wider opacity-90">{label}</div>
      <div className="text-base font-bold" style={{ fontFamily: "var(--font-montserrat)" }}>
        {value}
      </div>
    </div>
  );
}

function NutrientCard({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="p-2.5 rounded-md border border-brand-line bg-white">
      <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">{label}</div>
      <div className="text-sm font-bold text-brand-navy" style={{ fontFamily: "var(--font-montserrat)" }}>
        {value}
      </div>
      <div className="text-[10px] text-slate-400 mt-0.5">{sub}</div>
    </div>
  );
}
