// Comprehensive bovine disease database & vaccination schedule
// Aligned with Indian dairy farming conditions and NDBB guidelines

export type DiseaseCategory =
  | "bacterial"
  | "viral"
  | "parasitic"
  | "fungal"
  | "metabolic"
  | "nutritional"
  | "reproductive";

export type Severity = "low" | "moderate" | "high" | "critical";

export interface Disease {
  id: string;
  name: string;
  localName?: string;
  category: DiseaseCategory;
  severity: Severity;
  affectedSpecies: ("cattle" | "buffalo")[];
  zoonotic: boolean; // can it spread to humans
  symptoms: string[];
  causes: string;
  transmission: string;
  prevention: string[];
  treatment: string[];
  vaccineAvailable: boolean;
  incubationDays: string;
  mortalityRate: string;
}

export const diseases: Disease[] = [
  {
    id: "fmd",
    name: "Foot and Mouth Disease (FMD)",
    localName: "Muhkhiya / Khurkiya",
    category: "viral",
    severity: "critical",
    affectedSpecies: ["cattle", "buffalo"],
    zoonotic: false,
    symptoms: [
      "High fever (104–106°F)",
      "Excessive salivation and drooling",
      "Blisters in mouth, tongue, and lips",
      "Blisters between hooves and on udder",
      "Lameness and reluctance to move",
      "Sudden drop in milk yield",
      "Loss of appetite",
    ],
    causes:
      "Aphthovirus (FMDV) with seven serotypes (O, A, C, SAT1, SAT2, SAT3, Asia1). Serotype O is most common in India.",
    transmission:
      "Direct contact with infected animals, contaminated feed/water, aerosol droplets, animal products, and human carriers via clothing/vehicles.",
    prevention: [
      "Vaccinate every 6 months with trivalent vaccine (O, A, Asia1)",
      "Quarantine new animals for 21 days before introducing to herd",
      "Disinfect premises with 4% sodium carbonate",
      "Restrict vehicle and visitor movement during outbreaks",
      "Report suspected cases to local veterinary officer immediately",
    ],
    treatment: [
      "No specific antiviral treatment — supportive care only",
      "Apply alum or boric acid mouth wash to blisters",
      "Wash hooves with 1% copper sulfate solution",
      "Provide soft palatable feed (kanji, gruel, mashed fodder)",
      "Administer antibiotics to prevent secondary bacterial infections",
      "Isolate infected animals and provide clean drinking water",
    ],
    vaccineAvailable: true,
    incubationDays: "2–14 days",
    mortalityRate: "1–5% (adults), up to 50% (young calves)",
  },
  {
    id: "hs",
    name: "Haemorrhagic Septicaemia (HS)",
    localName: "Galghotu / Pashu khatua",
    category: "bacterial",
    severity: "critical",
    affectedSpecies: ["cattle", "buffalo"],
    zoonotic: false,
    symptoms: [
      "Sudden high fever (106–107°F)",
      "Difficulty breathing and grunting",
      "Swelling of throat, dewlap, and brisket",
      "Excessive salivation",
      "Nasal discharge turning bloody",
      "Recumbency and death within 24–48 hours",
    ],
    causes: "Pasteurella multocida serotype B:2 (Carters type 6)",
    transmission:
      "Through contaminated water, feed, and close contact. Outbreaks peak during monsoon and humid weather.",
    prevention: [
      "Vaccinate annually before monsoon (May–June)",
      "Avoid grazing in swampy waterlogged areas",
      "Provide clean drinking water",
      "Maintain dry, well-ventilated sheds",
    ],
    treatment: [
      "Emergency veterinary intervention required — death can occur in hours",
      "Intravenous sulfonamides and antibiotics (oxytetracycline, enrofloxacin)",
      "Anti-inflammatory drugs (meloxicam, flunixin)",
      "IV fluids to counter shock",
      "Early treatment can save 50–60% of cases",
    ],
    vaccineAvailable: true,
    incubationDays: "24–72 hours",
    mortalityRate: "80–100% without treatment",
  },
  {
    id: "bq",
    name: "Black Quarter (BQ)",
    localName: "Sui Kattam / Karia Pada",
    category: "bacterial",
    severity: "critical",
    affectedSpecies: ["cattle", "buffalo"],
    zoonotic: false,
    symptoms: [
      "Sudden high fever (106–107°F)",
      "Lameness in one limb",
      "Swelling of affected muscle (crepitating on touch)",
      "Skin discoloration to dark red/purple",
      "Gas bubbles under skin",
      "Death within 24–48 hours of symptom onset",
    ],
    causes: "Clostridium chauvoei spore-forming bacterium",
    transmission:
      "Spores persist in soil for years; infection enters through wounds or ingestion of contaminated feed.",
    prevention: [
      "Annual vaccination before monsoon (April–May)",
      "Avoid grazing animals on infected pastures",
      "Proper wound management and aseptic castration",
      "Do not open carcasses on grazing land — burn or bury deep",
    ],
    treatment: [
      "Emergency veterinary call required",
      "Massive doses of penicillin intravenously",
      "Surgical incision of affected muscle to expose to oxygen",
      "Anti-inflammatory and supportive therapy",
      "Often fatal even with treatment once symptoms appear",
    ],
    vaccineAvailable: true,
    incubationDays: "1–5 days",
    mortalityRate: "90–100% without early treatment",
  },
  {
    id: "brucellosis",
    name: "Brucellosis",
    localName: "Contagious abortion",
    category: "bacterial",
    severity: "high",
    affectedSpecies: ["cattle", "buffalo"],
    zoonotic: true,
    symptoms: [
      "Late-term abortion (7–9 months gestation)",
      "Retention of placenta after calving",
      "Reduced milk yield",
      "Swollen testicles in bulls (orchitis)",
      "Lameness due to joint involvement",
      "Repeated breeding failures",
    ],
    causes: "Brucella abortus bacterium (B. melitensis in some regions)",
    transmission:
      "Spread through contact with aborted foetus, placenta, uterine discharges, and contaminated feed/water. Highly zoonotic — humans get undulant fever.",
    prevention: [
      "Vaccinate female calves 4–8 months with Brucella abortus strain 19 or RB51",
      "Test and cull seropositive animals",
      "Burn or bury aborted material deep",
      "Wear gloves when handling abortions",
      "Quarantine new animals and test before introduction",
      "Do not consume raw milk from infected herds",
    ],
    treatment: [
      "No curative treatment — infected animals should be culled",
      "Antibiotic therapy (streptomycin + tetracycline) is rarely effective in cattle",
      "Humans treated with doxycycline + rifampicin for 6 weeks",
      "Focus on prevention through vaccination and biosecurity",
    ],
    vaccineAvailable: true,
    incubationDays: "14–120 days",
    mortalityRate: "Low mortality but high economic loss",
  },
  {
    id: "mastitis",
    name: "Mastitis",
    localName: "Stan shoth",
    category: "bacterial",
    severity: "high",
    affectedSpecies: ["cattle", "buffalo"],
    zoonotic: false,
    symptoms: [
      "Inflammation and swelling of udder quarter",
      "Milk becomes watery, clotted, or blood-tinged",
      "Pain and heat in affected quarter",
      "Reduced milk yield",
      "Fever and loss of appetite in severe cases",
      "Hardening of udder tissue in chronic cases",
    ],
    causes:
      "Bacterial infection — Staphylococcus aureus, Streptococcus agalactiae, E. coli, Klebsiella, and Pseudomonas species.",
    transmission:
      "Bacteria enter through teat canal via contaminated milking hands, equipment, or environment. Poor hygiene and improper milking technique increase risk.",
    prevention: [
      "Practice clean milking: wash udder, dry with individual cloth",
      "Dip teats in 0.5% iodine or 4% sodium hypochlorite after milking",
      "Treat dry cows with dry cow therapy at end of lactation",
      "Maintain clean, dry bedding",
      "Use properly maintained milking machines with correct vacuum",
      "Cull chronic carrier cows",
    ],
    treatment: [
      "Veterinarian to perform antibiotic sensitivity test on milk sample",
      "Intramammary antibiotic tubes (amoxicillin, cloxacillin) for 3–5 days",
      "Systemic antibiotics in severe cases",
      "Frequent stripping of affected quarter to remove bacteria",
      "Anti-inflammatory drugs (meloxicam) for pain and swelling",
      "Supportive therapy: fluids, vitamins",
    ],
    vaccineAvailable: false,
    incubationDays: "Variable (days to weeks)",
    mortalityRate: "Low mortality, high production loss",
  },
  {
    id: "theileriosis",
    name: "Bovine Theileriosis",
    localName: "Tick fever",
    category: "parasitic",
    severity: "high",
    affectedSpecies: ["cattle"],
    zoonotic: false,
    symptoms: [
      "High fever (104–107°F) lasting 1–2 weeks",
      "Swollen lymph nodes (prescapular, prefemoral)",
      "Pale mucous membranes (anaemia)",
      "Loss of appetite and weakness",
      "Drop in milk yield",
      "Laboured breathing in late stages",
    ],
    causes: "Theileria annulata protozoan (transmitted by Hyalomma ticks)",
    transmission: "Biological vector — Hyalomma anatolicum ticks. Disease common in crossbred cattle.",
    prevention: [
      "Regular acaricidal dipping/spraying (deltamethrin, amitraz)",
      "Vaccinate calves with Theileria annulata (Rakshar) vaccine",
      "Keep indigenous cattle in tick-endemic areas — they have natural resistance",
      "Pasture rotation to break tick life cycle",
    ],
    treatment: [
      "Buparvaquone (Butalex) injection — drug of choice",
      "Supportive therapy: haematinics, IV fluids, blood transfusion in severe anaemia",
      "Oxytetracycline to prevent secondary infections",
      "Treat concurrent tick infestation",
      "Crossbred cattle have higher mortality — start treatment early",
    ],
    vaccineAvailable: true,
    incubationDays: "10–25 days",
    mortalityRate: "10–90% (crossbreds more susceptible)",
  },
  {
    id: "fascioliasis",
    name: "Fascioliasis (Liver Fluke)",
    localName: "Phepar keeda",
    category: "parasitic",
    severity: "moderate",
    affectedSpecies: ["cattle", "buffalo"],
    zoonotic: true,
    symptoms: [
      "Chronic weight loss despite good appetite",
      "Bottle jaw (submandibular oedema)",
      "Pale mucous membranes",
      "Diarrhoea or constipation alternating",
      "Reduced milk yield",
      "Rough hair coat",
    ],
    causes: "Fasciola gigantica (tropical liver fluke) or Fasciola hepatica",
    transmission:
      "Snail is intermediate host. Cattle ingest metacercariae on grass in waterlogged pastures. Common in irrigated and flood-prone areas.",
    prevention: [
      "Deworm with flukicide (triclabendazole, oxyclozanide) every 6 months",
      "Drain stagnant water in pastures",
      "Avoid grazing in marshy areas during monsoon",
      "Control snail populations with copper sulfate",
    ],
    treatment: [
      "Triclabendazole 12 mg/kg oral — drug of choice, effective against all stages",
      "Oxyclozanide 15 mg/kg oral — effective against adult flukes",
      "Albendazole 15 mg/kg oral — broad-spectrum option",
      "Supportive therapy: haematinics, vitamin B12, mineral mixture",
      "Repeat treatment in 6–8 weeks to kill immature flukes",
    ],
    vaccineAvailable: false,
    incubationDays: "10–14 weeks (chronic form)",
    mortalityRate: "Low in adults, high in young calves",
  },
  {
    id: "hsd",
    name: "Hypocalcaemia (Milk Fever)",
    localName: "Doodh bugar",
    category: "metabolic",
    severity: "critical",
    affectedSpecies: ["cattle", "buffalo"],
    zoonotic: false,
    symptoms: [
      "Occurs within 72 hours of calving",
      "Loss of appetite and dullness",
      "Sternal recumbency with head turned to flank (S-shaped neck)",
      "Cold ears and horns",
      "Dry muzzle and constipation",
      "Coma and death within 12–24 hours if untreated",
    ],
    causes: "Sudden drop in blood calcium levels due to onset of lactation, especially in high-yielding older cows.",
    transmission: "Not contagious — metabolic disorder",
    prevention: [
      "Feed low-calcium diet in last 2 weeks of pregnancy",
      "Provide anionic salts pre-partum to acidify diet",
      "Ensure adequate vitamin D3 supplementation",
      "Avoid over-conditioning dry cows",
      "Inject calcium borogluconate at calving for high-risk cows",
    ],
    treatment: [
      "Emergency IV calcium borogluconate (slow, 10–15 min) — veterinary supervision",
      "Keep cow standing after infusion to avoid relapse",
      "Subcutaneous calcium for less acute cases",
      "Oral calcium gels for follow-up",
      "Monitor for relapse — may need second infusion in 12 hours",
    ],
    vaccineAvailable: false,
    incubationDays: "Sudden onset at calving",
    mortalityRate: "60–80% without treatment, <5% with prompt treatment",
  },
  {
    id: "ketosis",
    name: "Ketosis",
    localName: "Acetonaemia",
    category: "metabolic",
    severity: "moderate",
    affectedSpecies: ["cattle", "buffalo"],
    zoonotic: false,
    symptoms: [
      "Onset 2–6 weeks post-calving in high-yielding cows",
      "Loss of appetite, especially for grain",
      "Sweet acetone breath (fruity smell)",
      "Sudden drop in milk yield",
      "Weight loss despite good appetite for roughage",
      "Nervous form: licking, chewing inanimate objects, incoordination",
    ],
    causes: "Negative energy balance in early lactation — body fat mobilised produces ketone bodies.",
    transmission: "Not contagious — metabolic disorder",
    prevention: [
      "Ensure proper body condition at calving (BCS 3.0–3.5)",
      "Feed high-quality ration in early lactation",
      "Provide propylene glycol (250 ml/day) for high-risk cows",
      "Avoid stress and sudden ration changes",
      "Provide palatable energy-dense feed",
    ],
    treatment: [
      "IV dextrose 50% (250–500 ml) for rapid response",
      "Oral propylene glycol 250 ml twice daily for 5 days",
      "Corticosteroids (dexamethasone) to stimulate glucose production",
      "Vitamin B12 (cyanocobalamin) to aid metabolism",
      "Adjust ration to increase energy density",
    ],
    vaccineAvailable: false,
    incubationDays: "Develops over days",
    mortalityRate: "Low with treatment, can become chronic",
  },
  {
    id: "rinderpest",
    name: "Rinderpest (Eradicated)",
    localName: "Mata roga",
    category: "viral",
    severity: "critical",
    affectedSpecies: ["cattle", "buffalo"],
    zoonotic: false,
    symptoms: [
      "High fever (104–107°F)",
      "Necrotic lesions in mouth and tongue",
      "Profuse bloody diarrhoea",
      "Dehydration and rapid weight loss",
      "Death within 6–12 days",
    ],
    causes: "Morbillivirus — declared eradicated globally in 2011",
    transmission:
      "Historical disease — eradicated through mass vaccination. Maintain surveillance against accidental re-emergence.",
    prevention: [
      "Maintain surveillance and rapid reporting",
      "Strict import controls on live animals and animal products",
      "Veterinary preparedness for rapid response",
    ],
    treatment: [
      "No treatment available — historical disease",
      "Eradicated globally by FAO/OIE in 2011",
      "India declared free in 2006",
    ],
    vaccineAvailable: true,
    incubationDays: "3–15 days",
    mortalityRate: "Historical: 90–100%",
  },
  {
    id: "thed",
    name: "Theileriosis (Tropical)",
    localName: "Tick fever / Hema",
    category: "parasitic",
    severity: "moderate",
    affectedSpecies: ["cattle"],
    zoonotic: false,
    symptoms: [
      "Fever with swelling of superficial lymph nodes",
      "Pale mucous membranes indicating anaemia",
      "Reduced milk yield in dairy animals",
      "Anorexia and weight loss",
      "Difficult breathing",
    ],
    causes: "Theileria orientalis (Ikeda genotype) — emerging in some parts of India",
    transmission: "Haemaphysalis longicornis ticks and other tick vectors",
    prevention: [
      "Tick control with acaricides",
      "Vaccination where available",
      "Indigenous breed selection for endemic areas",
    ],
    treatment: [
      "Buparvaquone therapy",
      "Oxytetracycline for secondary infections",
      "Blood transfusion for severe anaemia",
      "Supportive care and rest",
    ],
    vaccineAvailable: false,
    incubationDays: "7–21 days",
    mortalityRate: "5–30% in crossbreds",
  },
  {
    id: "rp",
    name: "Reproductive Disorders Complex",
    localName: "Prajanan sankat",
    category: "reproductive",
    severity: "moderate",
    affectedSpecies: ["cattle", "buffalo"],
    zoonotic: false,
    symptoms: [
      "Repeat breeding (3+ services without conception)",
      "Anoestrus (failure to come into heat)",
      "Sub-oestrus (silent heat not detected)",
      "Retained placenta after calving",
      "Uterine prolapse or torsion",
      "Metritis and endometritis",
    ],
    causes:
      "Multi-factorial: nutritional deficiencies (minerals, vitamins), hormonal imbalance, uterine infections, heat stress, poor breeding management.",
    transmission: "Not contagious in most cases — management and nutrition-related",
    prevention: [
      "Provide balanced ration with mineral mixture and vitamins",
      "Detect heat accurately using pedometry or observation 3x daily",
      "Maintain body condition score 3.0–3.5 at breeding",
      "Use fertile, disease-tested semen from reputable AI centres",
      "Practice clean calving and post-partum care",
      "Regular veterinary gynaecology camps",
    ],
    treatment: [
      "Veterinary diagnosis of specific cause (rectal palpation, ultrasound)",
      "Hormonal therapy: GnRH, PGF2α, progesterone as indicated",
      "Intra-uterine antibiotics for endometritis",
      "Mineral and vitamin supplementation",
      "Cull animals with permanent reproductive damage",
    ],
    vaccineAvailable: false,
    incubationDays: "Variable",
    mortalityRate: "Low mortality, high economic loss",
  },
];

// Vaccination schedule aligned with NDBB guidelines
export interface VaccineScheduleItem {
  id: string;
  disease: string;
  vaccineName: string;
  ageAtFirstDose: string;
  boosterInterval: string;
  timing: string;
  route: string;
  dose: string;
  notes: string;
}

export const vaccineSchedule: VaccineScheduleItem[] = [
  {
    id: "fmd-vac",
    disease: "Foot and Mouth Disease",
    vaccineName: "FMD trivalent (O, A, Asia1) — Rakshafmd",
    ageAtFirstDose: "4 months",
    boosterInterval: "Every 6 months",
    timing: "March & September (National FMD Control Programme)",
    route: "Subcutaneous, mid-neck",
    dose: "2 ml (cattle/buffalo)",
    notes:
      "Cover entire herd simultaneously. Avoid vaccinating pregnant animals in last month. Mild swelling at injection site is normal.",
  },
  {
    id: "hs-vac",
    disease: "Haemorrhagic Septicaemia",
    vaccineName: "HS oil-adjuvant or alum-precipitated vaccine",
    ageAtFirstDose: "6 months",
    boosterInterval: "Annually (before monsoon)",
    timing: "May–June (pre-monsoon)",
    route: "Subcutaneous, mid-neck",
    dose: "2 ml (cattle), 3 ml (buffalo)",
    notes:
      "Oil-adjuvant gives longer immunity (1 year) but causes more swelling. Alum vaccine gives 4–6 month protection.",
  },
  {
    id: "bq-vac",
    disease: "Black Quarter",
    vaccineName: "BQ vaccine (formalin-killed)",
    ageAtFirstDose: "6 months",
    boosterInterval: "Annually (before monsoon)",
    timing: "April–May",
    route: "Subcutaneous, mid-neck",
    dose: "1 ml (cattle/buffalo)",
    notes: "Do not vaccinate animals under treatment with antibiotics. Avoid in last month of pregnancy.",
  },
  {
    id: "brucellosis-vac",
    disease: "Brucellosis",
    vaccineName: "Brucella abortus strain 19 (or RB51)",
    ageAtFirstDose: "4–8 months (female calves only)",
    boosterInterval: "Single dose — lifelong immunity",
    timing: "Any time before first breeding",
    route: "Subcutaneous, behind shoulder",
    dose: "Standard dose per manufacturer",
    notes:
      "Vaccinate ONLY female calves. Do not vaccinate adult pregnant animals. Avoid handling by pregnant women.",
  },
  {
    id: "theileriosis-vac",
    disease: "Bovine Theileriosis",
    vaccineName: "Theileria annulata vaccine (Rakshar)",
    ageAtFirstDose: "2–4 weeks (calves only)",
    boosterInterval: "Single dose — lifelong immunity",
    timing: "Best in calves under 6 months",
    route: "Subcutaneous, mid-neck",
    dose: "1 ml (reconstituted live vaccine)",
    notes:
      "Only effective in young calves. Adult vaccination causes severe reaction. Use within 4 hours of reconstitution.",
  },
  {
    id: "etv-vac",
    disease: "Enterotoxaemia (Pulpy Kidney)",
    vaccineName: "ETV combined with HS/BQ (combined vaccines available)",
    ageAtFirstDose: "6 months",
    boosterInterval: "Annually",
    timing: "Spring season",
    route: "Subcutaneous, mid-neck",
    dose: "2 ml",
    notes: "More important in sheep and goats but useful in cattle on high-concentrate rations.",
  },
];

export function filterDiseases(opts: { query?: string; category?: DiseaseCategory | "all" }) {
  return diseases.filter((d) => {
    if (opts.query) {
      const q = opts.query.toLowerCase();
      const match =
        d.name.toLowerCase().includes(q) ||
        (d.localName?.toLowerCase().includes(q) ?? false) ||
        d.symptoms.some((s) => s.toLowerCase().includes(q));
      if (!match) return false;
    }
    if (opts.category && opts.category !== "all" && d.category !== opts.category) return false;
    return true;
  });
}
