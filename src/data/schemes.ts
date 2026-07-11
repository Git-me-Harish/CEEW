// Government schemes & policies for Indian bovine sector
// Aligned with central and state government programs

export interface GovScheme {
  id: string;
  name: string;
  ministry: string;
  category: "subsidy" | "insurance" | "infrastructure" | "breeding" | "health" | "market";
  eligibility: string;
  benefit: string;
  subsidyPct: string;
  applicationProcess: string;
  documents: string[];
  website: string;
  summary: string;
}

export const govtSchemes: GovScheme[] = [
  {
    id: "rashtriya-gokul-mission",
    name: "Rashtriya Gokul Mission",
    ministry: "Department of Animal Husbandry & Dairying, GoI",
    category: "breeding",
    eligibility:
      "State governments, Goseva Aayogs, registered gaushalas, breeders' associations, FPOs engaged in indigenous bovine development.",
    benefit:
      "Financial assistance for establishment of Gokul Grams (cattle development centres), indigenous breed conservation, bull mother farm modernisation, and breed improvement through pedigree selection and progeny testing.",
    subsidyPct: "100% central funding (90:10 for NE & Himalayan states)",
    applicationProcess:
      "Apply through State Animal Husbandry Department as project proposals. State Government forwards to DAHD for sanction. Implementing agencies selected at state level.",
    documents: [
      "Project proposal with detailed project report",
      "Registration certificate of organisation",
      "Land documents for proposed Gokul Gram",
      "Bank account details and audit reports",
      "Endorsement from State AH Department",
    ],
    website: "https://dahd.nic.in/schemes/rashtriya-gokul-mission",
    summary:
      "Flagship central scheme for development and conservation of indigenous bovine breeds. Supports Gokul Grams, breed conservation, and genetic upgradation of native cattle through scientific breeding.",
  },
  {
    id: "npmdd",
    name: "National Programme for Dairy Development (NPDD)",
    ministry: "Department of Animal Husbandry & Dairying, GoI",
    category: "infrastructure",
    eligibility:
      "Milk unions, federations, dairy cooperatives, registered milk producer companies. State Dairy Development Departments can also apply.",
    benefit:
      "Infrastructure creation for milk procurement, chilling, processing, and training. Support for village-level dairy infrastructure, AI centres, and quality milk production.",
    subsidyPct: "50% of project cost (60% for NE & Himalayan states, 90% for FPOs up to Rs 50 lakh)",
    applicationProcess:
      "Apply through State AH Department or directly on DAHD portal. Project appraisal by NDDB/SIDBI. Sanction by DAHD Inter-Ministerial Committee.",
    documents: [
      "Detailed Project Report (DPR)",
      "Constitutional documents of the cooperative/company",
      "Audited financial statements for last 3 years",
      "Land ownership / lease documents",
      "Bank sanction letter for loan component",
    ],
    website: "https://dahd.nic.in/schemes/national-programme-dairy-development",
    summary:
      "Created by merging earlier schemes, NPDD supports dairy infrastructure including milk chilling centres, bulk coolers, AI facility establishment, and milk testing laboratories for unorganised dairy sectors.",
  },
  {
    id: "pashu-kisan-credit-card",
    name: "Pashu Kisan Credit Card (KCC for Animal Husbandry)",
    ministry: "Ministry of Finance, GoI (NABARD guidelines)",
    category: "subsidy",
    eligibility:
      "Individual farmers, joint liability groups, and SHGs engaged in dairying, poultry, fisheries, or beekeeping. No land ownership requirement.",
    benefit:
      "Working capital loan up to Rs 1,60,000 per farmer for animal husbandry activities (dairy, piggery, poultry). Interest subvention of 2% + additional 3% for prompt payment (effective rate ~4%).",
    subsidyPct: "Interest subvention up to 5% (effective interest rate 4%)",
    applicationProcess:
      "Apply at any commercial bank, RRB, or cooperative bank with KCC application form and livestock details. Bank verifies and issues KCC card within 30 days.",
    documents: [
      "Aadhaar card",
      "Land records (if owned) or NOC from landlord",
      "Quotation for animal purchase (if new purchase)",
      "Photographs and bank account details",
      "Self-declaration of livestock owned",
    ],
    website: "https://www.nabard.org/kcc.asp",
    summary:
      "Extension of KCC to animal husbandry farmers. Provides low-interest working capital for feed, fodder, healthcare, and purchase of animals. Coverage includes dairy, piggery, poultry, fisheries, and beekeeping.",
  },
  {
    id: "pashu-bima",
    name: "Pradhan Mantri Pashu Bima Yojana (Livestock Insurance)",
    ministry: "Department of Animal Husbandry & Dairying, GoI",
    category: "insurance",
    eligibility:
      "All indigenous, crossbred cattle, buffaloes, and other livestock owners. Maximum 10 cattle per beneficiary under subsidy.",
    benefit:
      "Insurance coverage for cattle/buffalo at agreed value (max Rs 60,000 for crossbred cow, Rs 40,000 for indigenous, Rs 50,000 for buffalo). Premium subsidy makes cover affordable for smallholders.",
    subsidyPct:
      "Premium subsidy: 50% for general farmers, 70% for SC/ST/small farmers, balance borne by farmer",
    applicationProcess:
      "Approach empanelled insurance company (LIC, New India, Oriental, etc.) or Common Service Centre. Tag animal with unique ID. Pay premium. Policy issued within 7 days.",
    documents: [
      "Aadhaar card and bank account details",
      "Photograph of animal",
      "Animal identification (ear tag number)",
      "Vaccination certificate",
      "Purchase proof or valuation certificate",
    ],
    website: "https://dahd.nic.in/schemes",
    summary:
      "Subsidised livestock insurance covering death due to disease, accident, surgical risks, and natural calamities. Premium subsidised for 5 cattle per farmer (10 for SC/ST). Reduces economic risk for dairy farmers.",
  },
  {
    id: "npddt",
    name: "Dairy Processing & Infrastructure Development Fund (DIDF)",
    ministry: "Department of Animal Husbandry & Dairying, NABARD",
    category: "infrastructure",
    eligibility:
      "Milk unions, state dairy federations, multi-state milk cooperatives, NDDB subsidiaries, and milk producer companies.",
    benefit:
      "Loan of Rs 11,084 crore for dairy processing, chilling, value addition, and human capacity building. Interest subvention makes effective interest rate 6–7%.",
    subsidyPct: "Interest subvention of 2–3% on NABARD loan",
    applicationProcess:
      "Submit DPR to NABARD through milk union/federation. NABARD appraises and sanctions. State government provides guarantee. Project implementation by milk union.",
    documents: [
      "Detailed Project Report (DPR)",
      "Board resolution authorising MD to borrow",
      "Audited financial statements for 5 years",
      "State government guarantee",
      "Loan agreement with NABARD",
    ],
    website: "https://www.nabard.org/didf.asp",
    summary:
      "Rs 11,084 crore fund for modernisation of dairy infrastructure — processing plants, chilling centres, packaging, and human resource development. Targets doubling farmer milk procurement income.",
  },
  {
    id: "kcc-fodder",
    name: "Fodder & Feed Development Sub-Mission (SMPP)",
    ministry: "Department of Animal Husbandry & Dairying, GoI",
    category: "infrastructure",
    eligibility:
      "Individual farmers, FPOs, cooperatives, NGOs, and state governments engaged in fodder production, seed multiplication, and silage making.",
    benefit:
      "Subsidy for establishment of fodder seed production farms, silage making units, hay baling units, total mixed ration (TMR) plants, and fodder banks. Up to 50% of unit cost.",
    subsidyPct: "25–50% (up to 90% for NE/Himalayan states)",
    applicationProcess:
      "Apply through State AH Department or directly on DAHD online portal. Project sanctioned by State Sanctioning Committee. Funds released directly to beneficiary account.",
    documents: [
      "Detailed Project Report",
      "Land ownership proof",
      "Bank account details",
      "Quotations for equipment",
      "Registration certificate (for organisations)",
    ],
    website: "https://dahd.nic.in/schemes",
    summary:
      "Sub-mission under National Mission on Bovine Breeding & Dairy Development for fodder and feed development. Addresses critical fodder shortage (35% deficit in green fodder) through area expansion and productivity enhancement.",
  },
  {
    id: "erdp",
    name: "e-Raksha Disease Control Programme (Pest & Disease Control)",
    ministry: "Department of Animal Husbandry & Dairying, GoI",
    category: "health",
    eligibility: "All cattle and buffalo farmers — vaccination free of cost under national programme.",
    benefit:
      "Free vaccination against FMD, Brucellosis, PPR (sheep/goat), classical swine fever, and dog rabies. Free deworming in some states. Compensation for culled animals in some schemes.",
    subsidyPct: "100% free for farmers under FMD-Mukta campaign",
    applicationProcess:
      "Contact local veterinary officer or attend village-level vaccination camps. Animals identified by ear-tagging under INAPH (Information Network for Animal Productivity & Health).",
    documents: [
      "Aadhaar card",
      "Bank account details (for compensation, if any)",
      "Animal details (count, age, breed)",
      "Vaccination records",
    ],
    website: "https://dahd.nic.in/schemes",
    summary:
      "Sub-Mission on Livestock Health & Disease Control. Provides free mass vaccination against major diseases including FMD (FMD-CP), Brucellosis (Brucellosis-CP), and PPR. Coverage expanded to all districts under FMD Mukt Bharat campaign.",
  },
  {
    id: "kisan-suvidha",
    name: "Animal Husbandry Infrastructure Development Fund (AHIDF)",
    ministry: "Department of Animal Husbandry & Dairying, GoI",
    category: "infrastructure",
    eligibility:
      "Private companies, FPOs, individual entrepreneurs, cooperatives investing in dairy, meat processing, animal feed plants.",
    benefit:
      "Rs 15,000 crore fund with 3% interest subvention + credit guarantee up to 25% of project cost for MSMEs. Loan up to 90% of project cost.",
    subsidyPct: "3% interest subvention on loans + 25% credit guarantee coverage",
    applicationProcess:
      "Apply online on AHIDF portal. Eligibility checked by NABARD/SIDBI. Project appraisal and sanction within 30 days. Disbursement in tranches.",
    documents: [
      "Project report with detailed financials",
      "Entity registration (Pvt Ltd, FPO, cooperative)",
      "PAN and GST registration",
      "Bank account with main bank",
      "Quotations for plant & machinery",
      "Land documents / lease agreement",
    ],
    website: "https://ahidf.veda.gov.in/",
    summary:
      "Rs 15,000 crore fund to encourage private investment in dairy, meat, and animal feed infrastructure. Targets 7,500+ new projects creating 35 lakh jobs. Includes credit guarantee and interest subvention.",
  },
  {
    id: "soi",
    name: "Semen Station Modernisation Scheme",
    ministry: "Department of Animal Husbandry & Dairying, GoI",
    category: "breeding",
    eligibility:
      "State livestock development boards, BAIF, NDDB, established semen stations seeking modernisation.",
    benefit:
      "Financial assistance for upgrading semen stations with state-of-art equipment, sex-sorted semen technology, and quality control laboratories. Up to 100% central assistance.",
    subsidyPct: "100% central funding for government stations",
    applicationProcess:
      "State AH Department submits proposal to DAHD. Central Sanctioning Committee approves. Implementation by state LD boards.",
    documents: [
      "Project proposal with cost estimates",
      "Existing infrastructure details",
      "Audited financial statements",
      "Strategic plan for 5 years",
      "State government endorsement",
    ],
    website: "https://dahd.nic.in/schemes",
    summary:
      "Modernisation of semen stations for production of high-genetic-merit semen doses including sex-sorted semen for elite cattle. Aims to double AI coverage from present 30% to 65% by 2030.",
  },
  {
    id: "kisan-credit-mushroom",
    name: "Dairy Entrepreneurship Development Scheme (DEDS)",
    ministry: "Department of Animal Husbandry & Dairying, NABARD",
    category: "subsidy",
    eligibility:
      "Individual entrepreneurs, SHGs, FPOs, cooperatives starting dairy, poultry, piggery, sheep, goat, or rabbit enterprises.",
    benefit:
      "Entrepreneurship development support with back-ended capital subsidy of 25–33.33% of project cost (up to Rs 20 lakh for dairy farms).",
    subsidyPct: "25% (33.33% for SC/ST/women/N-E states)",
    applicationProcess:
      "Submit project report to bank. Bank appraises and sanctions. After implementation, NABARD releases subsidy to bank, adjusted against loan.",
    documents: [
      "Detailed project report",
      "Aadhaar and PAN card",
      "Bank account details",
      "Land records (if applicable)",
      "Quotations for animals and equipment",
    ],
    website: "https://www.nabard.org/deds.asp",
    summary:
      "NABARD-operated scheme promoting dairy entrepreneurship. Supports establishment of small dairy farms (2–10 animals), milk product units, dairies with chilling facilities, and calf-rearing stations. Creates rural livelihoods.",
  },
];

// Quick stats
export const schemeStats = {
  total: govtSchemes.length,
  subsidies: govtSchemes.filter((s) => s.category === "subsidy").length,
  insurance: govtSchemes.filter((s) => s.category === "insurance").length,
  infrastructure: govtSchemes.filter((s) => s.category === "infrastructure").length,
};

// Market data — daily prices snapshot (mock realistic data)
export interface MarketPrice {
  commodity: string;
  category: "milk" | "cattle" | "fodder" | "ghee";
  unit: string;
  modalPrice: number;
  minPrice: number;
  maxPrice: number;
  trend: "up" | "down" | "stable";
  changePct: number;
  market: string;
  date: string;
}

export const marketPrices: MarketPrice[] = [
  {
    commodity: "Cow Milk (Crossbred)",
    category: "milk",
    unit: "per litre",
    modalPrice: 32,
    minPrice: 28,
    maxPrice: 38,
    trend: "up",
    changePct: 2.4,
    market: "Cooperative avg (Amul/Mother Dairy)",
    date: "Daily updated",
  },
  {
    commodity: "Buffalo Milk (Murrah)",
    category: "milk",
    unit: "per litre",
    modalPrice: 58,
    minPrice: 52,
    maxPrice: 65,
    trend: "up",
    changePct: 3.1,
    market: "Haryana/Punjab dairy",
    date: "Daily updated",
  },
  {
    commodity: "Indigenous Cow Milk (Gir)",
    category: "milk",
    unit: "per litre",
    modalPrice: 75,
    minPrice: 60,
    maxPrice: 120,
    trend: "up",
    changePct: 5.2,
    market: "A2 milk premium market",
    date: "Daily updated",
  },
  {
    commodity: "Ghee (Cow A2)",
    category: "ghee",
    unit: "per kg",
    modalPrice: 1200,
    minPrice: 900,
    maxPrice: 1800,
    trend: "up",
    changePct: 4.5,
    market: "Retail premium",
    date: "Daily updated",
  },
  {
    commodity: "Gir Heifer (in-milk)",
    category: "cattle",
    unit: "per animal",
    modalPrice: 65000,
    minPrice: 50000,
    maxPrice: 95000,
    trend: "stable",
    changePct: 0.8,
    market: "Bhavnagar cattle fair",
    date: "Weekly",
  },
  {
    commodity: "Murrah Buffalo (in-milk)",
    category: "cattle",
    unit: "per animal",
    modalPrice: 85000,
    minPrice: 60000,
    maxPrice: 250000,
    trend: "up",
    changePct: 2.1,
    market: "Rohtak/Hisar mandi",
    date: "Weekly",
  },
  {
    commodity: "Sahiwal Cow (in-milk)",
    category: "cattle",
    unit: "per animal",
    modalPrice: 70000,
    minPrice: 50000,
    maxPrice: 125000,
    trend: "stable",
    changePct: 0.4,
    market: "Punjab Livestock Market",
    date: "Weekly",
  },
  {
    commodity: "Crossbred Heifer",
    category: "cattle",
    unit: "per animal",
    modalPrice: 35000,
    minPrice: 28000,
    maxPrice: 48000,
    trend: "down",
    changePct: -1.5,
    market: "Maharashtra mandi",
    date: "Weekly",
  },
  {
    commodity: "Green Maize Fodder",
    category: "fodder",
    unit: "per quintal",
    modalPrice: 350,
    minPrice: 280,
    maxPrice: 450,
    trend: "up",
    changePct: 3.8,
    market: "Pune/Ahmednagar mandi",
    date: "Daily",
  },
  {
    commodity: "Wheat Straw (Tudi)",
    category: "fodder",
    unit: "per quintal",
    modalPrice: 850,
    minPrice: 700,
    maxPrice: 1100,
    trend: "up",
    changePct: 2.2,
    market: "Punjab/Haryana",
    date: "Daily",
  },
  {
    commodity: "Sorghum Hay (Jowar)",
    category: "fodder",
    unit: "per quintal",
    modalPrice: 1100,
    minPrice: 900,
    maxPrice: 1400,
    trend: "stable",
    changePct: 0.5,
    market: "Maharashtra/Karnataka",
    date: "Daily",
  },
  {
    commodity: "Concentrate Mix (Cattle Feed)",
    category: "fodder",
    unit: "per quintal",
    modalPrice: 2100,
    minPrice: 1850,
    maxPrice: 2450,
    trend: "up",
    changePct: 1.8,
    market: "Brand feeds (Amul, Chokha)",
    date: "Weekly",
  },
];

// Veterinary resource directory (illustrative data)
export interface VetDirectory {
  id: string;
  name: string;
  type: "hospital" | "dispensary" | "mobile" | "polyclinic";
  organisation: string;
  district: string;
  state: string;
  services: string[];
  contact: string;
  availability: string;
}

export const vetDirectory: VetDirectory[] = [
  {
    id: "vet-1",
    name: "District Veterinary Hospital",
    type: "hospital",
    organisation: "State Animal Husbandry Department",
    district: "Every district HQ",
    state: "Pan-India",
    services: [
      "AI (Artificial Insemination)",
      "Disease diagnosis & treatment",
      "Surgery & obstetrics",
      "Vaccination",
      "Post-mortem",
    ],
    contact: "Reach via local AH office",
    availability: "Mon–Sat, 9 AM – 5 PM",
  },
  {
    id: "vet-2",
    name: "Primary Veterinary Centre",
    type: "dispensary",
    organisation: "State AH Department",
    district: "Block level (one per 10–15 villages)",
    state: "Pan-India",
    services: ["First aid", "Deworming", "Vaccination", "AI", "Minor surgery"],
    contact: "Block Development Office",
    availability: "Daily except Sundays",
  },
  {
    id: "vet-3",
    name: "Mobile Veterinary Unit (MVU)",
    type: "mobile",
    organisation: "State AH Department (under AHIDF)",
    district: "Operates in clusters of 20 villages",
    state: "20+ states covered",
    services: [
      "Doorstep AI",
      "Pregnancy diagnosis",
      "Treatment of sick animals",
      "Deworming & vaccination",
      "Minor surgical interventions",
    ],
    contact: "Toll-free 1962 / State AH helpline",
    availability: "Scheduled visits per village, 5 days/week",
  },
  {
    id: "vet-4",
    name: "Veterinary University Hospital",
    type: "polyclinic",
    organisation: "State Veterinary Universities (TANUVAS, DUVASU, RAJUVAS, MAFSU, etc.)",
    district: "University campus cities",
    state: "Multiple states",
    services: [
      "Advanced diagnostic & imaging",
      "Specialised surgery (orthopaedic, ophthalmic)",
      "Reproductive medicine",
      "Referral hospital for complex cases",
      "Clinical training & research",
    ],
    contact: "Direct appointment at hospital reception",
    availability: "24x7 emergency for critical cases",
  },
  {
    id: "vet-5",
    name: "Krishi Vigyan Kendra (KVK) Veterinary Cell",
    type: "dispensary",
    organisation: "ICAR (Indian Council of Agricultural Research)",
    district: "One per district (725 KVKs total)",
    state: "Pan-India",
    services: ["Health camps", "Training for farmers", "Diagnostic support", "Awareness programs"],
    contact: "Local KVK office",
    availability: "Working hours Mon–Fri",
  },
  {
    id: "vet-6",
    name: "Milk Cooperative Veterinary Service",
    type: "dispensary",
    organisation: "Amul, Mother Dairy, Nandini, Verka, Sudha, OMFED etc.",
    district: "Cooperative operational areas",
    state: "Dairy-majority states",
    services: [
      "Free/subsidised AI for member farmers",
      "Pregnancy diagnosis",
      "Vaccination camps",
      "Mastitis treatment",
      "Feed & fodder advisory",
    ],
    contact: "Local milk collection centre / dairy development officer",
    availability: "Daily at village collection centres",
  },
  {
    id: "vet-7",
    name: "Private Veterinary Practitioner (PVP)",
    type: "dispensary",
    organisation: "Self-employed / chain clinics",
    district: "All major towns & cities",
    state: "Pan-India",
    services: [
      "On-call diagnosis & treatment",
      "Surgery and obstetrics",
      "Deworming & vaccination",
      "Emergency services",
      "Farm consultancy",
    ],
    contact: "Local directories / WhatsApp groups",
    availability: "24x7 emergency call-out (chargeable)",
  },
  {
    id: "vet-8",
    name: "Goshala Veterinary Service",
    type: "dispensary",
    organisation: "Registered goshalas (Go-Seva Aayog empanelled)",
    district: "All districts with significant goshala presence",
    state: "Pan-India",
    services: ["Rescue & rehabilitation", "Long-term care", "Free treatment for unproductive cattle"],
    contact: "Local goshala office",
    availability: "Daily",
  },
];
