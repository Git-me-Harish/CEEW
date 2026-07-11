// Bovine nutrition data — feed calculator & feeding standards
// Based on ICAR-NIANP feeding standards for Indian bovines

export type AnimalClass =
  | "calf"
  | "heifer"
  | "dry-cow"
  | "pregnant-cow"
  | "lactating-cow"
  | "lactating-buffalo"
  | "bull"
  | "working-bullock";

export interface FeedingStandard {
  class: AnimalClass;
  label: string;
  dmKgPer100KgBW: string; // dry matter intake per 100 kg body weight
  dcpKgPerDay: string; // digestible crude protein
  tdnKgPerDay: string; // total digestible nutrients
  calciumGPerDay: string;
  phosphorusGPerDay: string;
  vitAIU: string;
  vitDIU: string;
  notes: string;
}

export const feedingStandards: FeedingStandard[] = [
  {
    class: "calf",
    label: "Calf (0–6 months)",
    dmKgPer100KgBW: "2.5–3.0",
    dcpKgPerDay: "0.10–0.20",
    tdnKgPerDay: "0.40–0.70",
    calciumGPerDay: "12–18",
    phosphorusGPerDay: "8–12",
    vitAIU: "1500–3000",
    vitDIU: "300–500",
    notes:
      "Colostrum in first 2 hours (1/10th of body weight). Calf starter from week 2. Good quality hay from week 3. Wean at 90 days.",
  },
  {
    class: "heifer",
    label: "Heifer (6–24 months)",
    dmKgPer100KgBW: "2.5–3.0",
    dcpKgPerDay: "0.25–0.40",
    tdnKgPerDay: "1.20–2.50",
    calciumGPerDay: "25–40",
    phosphorusGPerDay: "15–25",
    vitAIU: "4000–6000",
    vitDIU: "500–800",
    notes:
      "Aim for 350–400 g/day average daily gain. Provide green fodder ad lib. Mineral mixture 30–50 g/day. Breed at 18–24 months / 250–280 kg BW.",
  },
  {
    class: "dry-cow",
    label: "Dry Cow (last 2 months of pregnancy)",
    dmKgPer100KgBW: "2.0–2.5",
    dcpKgPerDay: "0.40–0.50",
    tdnKgPerDay: "3.0–4.0",
    calciumGPerDay: "40–60",
    phosphorusGPerDay: "25–35",
    vitAIU: "6000–8000",
    vitDIU: "800–1200",
    notes:
      "Steaming up: increase grain 1 kg/day in last 2 weeks. Avoid excess calcium to prevent milk fever. Provide good quality hay.",
  },
  {
    class: "pregnant-cow",
    label: "Pregnant Cow (early/mid gestation)",
    dmKgPer100KgBW: "2.0–2.5",
    dcpKgPerDay: "0.35–0.45",
    tdnKgPerDay: "2.5–3.5",
    calciumGPerDay: "35–50",
    phosphorusGPerDay: "20–30",
    vitAIU: "5000–7000",
    vitDIU: "600–900",
    notes:
      "Maintain BCS 3.0–3.5. Avoid sudden ration changes. Provide clean water at all times. Free choice mineral mixture.",
  },
  {
    class: "lactating-cow",
    label: "Lactating Cow (crossbred)",
    dmKgPer100KgBW: "3.0–3.5",
    dcpKgPerDay: "0.60–1.20 (0.06 kg per kg milk)",
    tdnKgPerDay: "4.0–7.0 (0.35 kg per kg milk)",
    calciumGPerDay: "60–100 (2.5 g per kg milk)",
    phosphorusGPerDay: "35–60 (1.5 g per kg milk)",
    vitAIU: "8000–12000",
    vitDIU: "1000–1500",
    notes:
      "Add 1 kg concentrate per 2.5–3.0 kg milk above maintenance. Provide ad lib green fodder. Mineral mixture 50–80 g/day. Salt 30–50 g/day.",
  },
  {
    class: "lactating-buffalo",
    label: "Lactating Buffalo",
    dmKgPer100KgBW: "2.5–3.0",
    dcpKgPerDay: "0.70–1.30 (0.08 kg per kg milk)",
    tdnKgPerDay: "4.0–7.0 (0.40 kg per kg milk)",
    calciumGPerDay: "50–90 (2.5 g per kg milk)",
    phosphorusGPerDay: "30–55 (1.5 g per kg milk)",
    vitAIU: "10000–15000",
    vitDIU: "1500–2000",
    notes:
      "Buffalo has higher DCP requirement than cow per kg milk due to higher protein milk. Add 1 kg concentrate per 2.0–2.5 kg milk. Ensure drinking water below 25°C in summer.",
  },
  {
    class: "bull",
    label: "Breeding Bull",
    dmKgPer100KgBW: "2.0–2.5",
    dcpKgPerDay: "0.50–0.70",
    tdnKgPerDay: "3.0–4.5",
    calciumGPerDay: "30–50",
    phosphorusGPerDay: "20–35",
    vitAIU: "6000–8000",
    vitDIU: "800–1000",
    notes:
      "Maintain BCS 3.0–3.5 — not over-fat. Reduce grain if bull is not in active service. Provide regular exercise. Vitamin A critical for semen quality.",
  },
  {
    class: "working-bullock",
    label: "Working Bullock",
    dmKgPer100KgBW: "2.5–3.0",
    dcpKgPerDay: "0.45–0.60",
    tdnKgPerDay: "3.5–5.0 (extra 0.5 kg TDN per 2 hours work)",
    calciumGPerDay: "30–40",
    phosphorusGPerDay: "20–30",
    vitAIU: "5000–7000",
    vitDIU: "600–800",
    notes:
      "Increase TDN by 25–50% on working days. Provide extra grain at start of work day. Rest animals for 1–2 hours at midday. Salt 30–50 g/day.",
  },
];

// Common Indian feedstuffs with composition per kg dry matter
export interface Feedstuff {
  id: string;
  name: string;
  category: "roughage" | "concentrate" | "mineral" | "supplement";
  dmPct: number; // dry matter %
  cpPct: number; // crude protein %
  tdnPct: number; // total digestible nutrients %
  calciumPct: number;
  phosphorusPct: number;
  costPerKg: number;
  availability: string;
  notes: string;
}

export const feedstuffs: Feedstuff[] = [
  {
    id: "maize-fodder",
    name: "Maize Green Fodder",
    category: "roughage",
    dmPct: 25,
    cpPct: 8,
    tdnPct: 65,
    calciumPct: 0.4,
    phosphorusPct: 0.25,
    costPerKg: 4,
    availability: "Year-round with irrigation",
    notes: "Premier green fodder. Harvest at milk-dough stage for best nutritive value.",
  },
  {
    id: "sorghum-fodder",
    name: "Sorghum (Jowar) Fodder",
    category: "roughage",
    dmPct: 28,
    cpPct: 7,
    tdnPct: 60,
    calciumPct: 0.4,
    phosphorusPct: 0.2,
    costPerKg: 3.5,
    availability: "Kharif & Rabi season",
    notes: "Caution: prussic acid poisoning in young stunted growth. Harvest at 75–90 days.",
  },
  {
    id: "berseem",
    name: "Berseem (Egyptian Clover)",
    category: "roughage",
    dmPct: 18,
    cpPct: 18,
    tdnPct: 65,
    calciumPct: 1.5,
    phosphorusPct: 0.4,
    costPerKg: 5,
    availability: "November–April (Rabi)",
    notes: "Rich protein source. 6–7 cuts per season. Excellent for lactating animals.",
  },
  {
    id: "lucerne",
    name: "Lucerne (Alfalfa)",
    category: "roughage",
    dmPct: 22,
    cpPct: 20,
    tdnPct: 65,
    calciumPct: 1.8,
    phosphorusPct: 0.3,
    costPerKg: 6,
    availability: "Year-round (perennial)",
    notes: "King of fodders — high protein and calcium. 10–12 cuts/year. Needs good drainage.",
  },
  {
    id: "wheat-straw",
    name: "Wheat Straw (Tudi)",
    category: "roughage",
    dmPct: 90,
    cpPct: 3.5,
    tdnPct: 45,
    calciumPct: 0.2,
    phosphorusPct: 0.1,
    costPerKg: 9,
    availability: "Year-round (stored)",
    notes: "Common dry roughage in North India. Treat with urea to improve protein content.",
  },
  {
    id: "paddy-straw",
    name: "Paddy Straw",
    category: "roughage",
    dmPct: 90,
    cpPct: 4,
    tdnPct: 40,
    calciumPct: 0.2,
    phosphorusPct: 0.1,
    costPerKg: 6,
    availability: "Year-round (stored)",
    notes: "Lower quality than wheat straw. High silica content. Best for bedding + roughage.",
  },
  {
    id: "maize-grain",
    name: "Maize Grain",
    category: "concentrate",
    dmPct: 89,
    cpPct: 9,
    tdnPct: 85,
    calciumPct: 0.02,
    phosphorusPct: 0.28,
    costPerKg: 22,
    availability: "Year-round",
    notes: "High energy, low fibre. Premier concentrate for lactating animals. Coarsely ground before feeding.",
  },
  {
    id: "mustard-cake",
    name: "Mustard Oil Cake",
    category: "concentrate",
    dmPct: 90,
    cpPct: 32,
    tdnPct: 75,
    calciumPct: 0.6,
    phosphorusPct: 1.0,
    costPerKg: 32,
    availability: "Year-round",
    notes: "Common protein supplement in North India. Has glucosinolates — limit to 30% of concentrate mix.",
  },
  {
    id: "groundnut-cake",
    name: "Groundnut Oil Cake (Decorticated)",
    category: "concentrate",
    dmPct: 90,
    cpPct: 42,
    tdnPct: 78,
    calciumPct: 0.2,
    phosphorusPct: 0.6,
    costPerKg: 42,
    availability: "Year-round",
    notes: "Premium protein supplement. Aflatoxin risk if stored improperly — always check quality.",
  },
  {
    id: "soybean-meal",
    name: "Soybean Meal",
    category: "concentrate",
    dmPct: 90,
    cpPct: 45,
    tdnPct: 80,
    calciumPct: 0.3,
    phosphorusPct: 0.7,
    costPerKg: 52,
    availability: "Year-round",
    notes: "Excellent amino acid profile. Premium protein source for high-yielding animals and calves.",
  },
  {
    id: "rice-bran",
    name: "Deoiled Rice Bran",
    category: "concentrate",
    dmPct: 90,
    cpPct: 14,
    tdnPct: 65,
    calciumPct: 0.1,
    phosphorusPct: 1.5,
    costPerKg: 14,
    availability: "Year-round",
    notes: "Cost-effective energy + phosphorus source. Limit to 30% of concentrate due to fibre and phytates.",
  },
  {
    id: "molasses",
    name: "Molasses",
    category: "supplement",
    dmPct: 75,
    cpPct: 4,
    tdnPct: 75,
    calciumPct: 0.8,
    phosphorusPct: 0.1,
    costPerKg: 14,
    availability: "Year-round",
    notes: "Energy supplement, dust suppressant, and palatability enhancer. Limit to 10–15% of ration.",
  },
  {
    id: "mineral-mixture",
    name: "Mineral Mixture (BIS Type II)",
    category: "mineral",
    dmPct: 99,
    cpPct: 0,
    tdnPct: 0,
    calciumPct: 32,
    phosphorusPct: 12,
    costPerKg: 60,
    availability: "Year-round",
    notes: "Contains Ca, P, trace minerals (Cu, Zn, Mn, Co, I, Se). 50–80 g/day for lactating animals.",
  },
  {
    id: "common-salt",
    name: "Common Salt",
    category: "mineral",
    dmPct: 99,
    cpPct: 0,
    tdnPct: 0,
    calciumPct: 0,
    phosphorusPct: 0,
    costPerKg: 20,
    availability: "Year-round",
    notes: "Sodium and chloride source. 30–50 g/day for adult cattle. Essential for electrolyte balance.",
  },
];

// Feed calculator function — computes daily ration
export interface FeedCalcInput {
  bodyWeightKg: number;
  milkYieldKgDay: number;
  fatPct: number;
  animalClass: AnimalClass;
  monthsPregnant: number;
}

export interface FeedCalcResult {
  dmRequiredKg: number;
  dcpRequiredKg: number;
  tdnRequiredKg: number;
  calciumRequiredG: number;
  phosphorusRequiredG: number;
  greenFodderKg: number;
  dryRoughageKg: number;
  concentrateKg: number;
  mineralMixtureG: number;
  saltG: number;
  waterL: number;
  estimatedDailyCost: number;
  breakdown: { item: string; quantity: string; reason: string }[];
}

export function calculateFeed(input: FeedCalcInput): FeedCalcResult {
  const { bodyWeightKg, milkYieldKgDay, fatPct, animalClass, monthsPregnant } = input;

  // Maintenance requirements (ICAR-NIANP standards)
  const maintDCP = 0.03 * bodyWeightKg ** 0.75; // kg DCP for maintenance
  const maintTDN = 0.155 * bodyWeightKg ** 0.75; // kg TDN for maintenance

  // Production requirements
  const milkFatYield = milkYieldKgDay * fatPct / 100;
  const prodDCP = milkYieldKgDay * 0.06 + milkFatYield * 0.4;
  const prodTDN = milkYieldKgDay * 0.35 + milkFatYield * 0.95;

  // Pregnancy allowance (last 2 months)
  const pregnancyDCP = monthsPregnant >= 7 ? 0.15 : 0;
  const pregnancyTDN = monthsPregnant >= 7 ? 1.0 : 0;

  const totalDCP = maintDCP + prodDCP + pregnancyDCP;
  const totalTDN = maintTDN + prodTDN + pregnancyTDN;
  const totalDM = bodyWeightKg * 0.03 + milkYieldKgDay * 0.4; // kg DM
  const calciumG = 16 + milkYieldKgDay * 2.5;
  const phosphorusG = 12 + milkYieldKgDay * 1.5;

  // Ration formulation: 60% roughage (green + dry), 40% concentrate (high producers)
  const roughageKg = (totalDM * 0.6) / 0.9; // as-fed for dry roughage basis
  const greenFodderKg = (roughageKg * 0.6) / 0.25; // 25% DM
  const dryRoughageKg = (roughageKg * 0.4) / 0.9; // 90% DM
  const concentrateKg = (totalDM * 0.4) / 0.9; // 90% DM

  // Minerals & supplements
  const mineralMixtureG = Math.min(80, Math.max(30, milkYieldKgDay * 8 + 20));
  const saltG = Math.min(50, Math.max(20, milkYieldKgDay * 5 + 15));
  const waterL = bodyWeightKg * 0.07 + milkYieldKgDay * 4 + (bodyWeightKg * 0.05 * (input.animalClass === "lactating-buffalo" ? 1.5 : 1));

  // Cost estimation
  const cost =
    greenFodderKg * 4 +
    dryRoughageKg * 9 +
    concentrateKg * 28 +
    (mineralMixtureG / 1000) * 60 +
    (saltG / 1000) * 20;

  return {
    dmRequiredKg: parseFloat(totalDM.toFixed(1)),
    dcpRequiredKg: parseFloat(totalDCP.toFixed(2)),
    tdnRequiredKg: parseFloat(totalTDN.toFixed(2)),
    calciumRequiredG: Math.round(calciumG),
    phosphorusRequiredG: Math.round(phosphorusG),
    greenFodderKg: Math.round(greenFodderKg),
    dryRoughageKg: Math.round(dryRoughageKg),
    concentrateKg: parseFloat(concentrateKg.toFixed(1)),
    mineralMixtureG: Math.round(mineralMixtureG),
    saltG: Math.round(saltG),
    waterL: Math.round(waterL),
    estimatedDailyCost: Math.round(cost),
    breakdown: [
      {
        item: "Green Fodder (Maize/Berseem/Lucerne)",
        quantity: `${Math.round(greenFodderKg)} kg`,
        reason: "Provides protein, calcium, vitamins, and bulk. Should be 35–45% of ration DM.",
      },
      {
        item: "Dry Roughage (Wheat/Paddy Straw)",
        quantity: `${Math.round(dryRoughageKg)} kg`,
        reason: "Provides fibre for rumen health. 20–25% of ration DM. Urea-treated if poor quality.",
      },
      {
        item: "Concentrate Mix (Grain + Cake + Bran)",
        quantity: `${concentrateKg.toFixed(1)} kg`,
        reason: "Provides energy and protein for production. Add 1 kg per 2.5 kg milk above maintenance.",
      },
      {
        item: "Mineral Mixture (BIS Type II)",
        quantity: `${Math.round(mineralMixtureG)} g`,
        reason: "Provides Ca, P, trace minerals. Critical for high-yielding animals.",
      },
      {
        item: "Common Salt",
        quantity: `${Math.round(saltG)} g`,
        reason: "Sodium and chloride source. Essential for electrolyte balance and palatability.",
      },
      {
        item: "Clean Drinking Water",
        quantity: `${Math.round(waterL)} L`,
        reason: "Always available ad lib. Buffalo needs cooler water in summer. 1 kg milk = 4 L water.",
      },
    ],
  };
}

// Heat stress & weather advisory thresholds
export interface AdvisoryThreshold {
  condition: string;
  threshold: string;
  advisory: string;
  icon: string;
}

export const advisoryThresholds: AdvisoryThreshold[] = [
  {
    condition: "Heat stress",
    threshold: "THI > 80 (Temperature-Humidity Index)",
    advisory:
      "Provide cool drinking water at 2-hour intervals. Use fans and sprinklers. Graze early morning and late evening only. Increase energy density of ration by 10%. Add vitamin C (5 g/day) and electrolytes.",
    icon: "thermometer",
  },
  {
    condition: "Cold stress",
    threshold: "Temperature < 7°C (especially for crossbreds)",
    advisory:
      "Close shed curtains at night. Provide dry bedding (straw). Increase ration energy by 10–15%. Use warm water (15–20°C) for drinking. Calves need jackets or heat lamps.",
    icon: "snowflake",
  },
  {
    condition: "Monsoon humidity",
    threshold: "Relative humidity > 80% with 25–32°C temperature",
    advisory:
      "Increase deworming frequency. Watch for mastitis, FMD, HS outbreaks. Provide dry bedding. Avoid grazing in waterlogged pastures (liver fluke risk). Vaccinate HS before monsoon.",
    icon: "rain",
  },
  {
    condition: "Fodder scarcity",
    threshold: "Green fodder availability < 5 kg/animal/day",
    advisory:
      "Make silage from surplus green fodder. Use urea-treated straw. Concentrate mix becomes 50% of ration. Add vitamin A (10,000 IU/day). Grow fodder trees (Subabul, Gliricidia) as buffer.",
    icon: "leaf",
  },
  {
    condition: "Calving time",
    threshold: "Last 2 weeks of pregnancy",
    advisory:
      "Provide clean calving area. Reduce calcium in last week. Keep calcium borogluconate ready. Calf should get colostrum within 2 hours. Watch for retention of placenta.",
    icon: "baby",
  },
  {
    condition: "Lactation peak",
    threshold: "30–60 days post-calving",
    advisory:
      "Maximise energy intake. Increase concentrate gradually (0.5 kg/day). Watch for ketosis. Ensure 60% good quality roughage. Add niacin (6 g/day) and protected fat for high yielders.",
    icon: "trending-up",
  },
];
