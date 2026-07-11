// Seed script — populates the database with demo data for first-time visitors
import { db } from "../src/lib/db";

async function main() {
  console.log("Seeding database...");

  // Demo cattle
  const cattle1 = await db.cattle.create({
    data: {
      tagNumber: "IND-GIR-001",
      name: "Gauri",
      breed: "Gir",
      species: "cattle",
      sex: "female",
      birthDate: new Date("2020-04-15"),
      weightKg: 410,
      source: "Purchased from Junagadh cattle fair",
      notes: "High-yielding Gir cow, second lactation",
    },
  });

  const cattle2 = await db.cattle.create({
    data: {
      tagNumber: "IND-MUR-002",
      name: "Nandini",
      breed: "Murrah Buffalo",
      species: "buffalo",
      sex: "female",
      birthDate: new Date("2019-08-22"),
      weightKg: 580,
      source: "Home-bred",
      notes: "Peak yielder, 18 L/day at peak",
    },
  });

  const cattle3 = await db.cattle.create({
    data: {
      tagNumber: "IND-SAH-003",
      name: "Sundari",
      breed: "Sahiwal",
      species: "cattle",
      sex: "female",
      birthDate: new Date("2021-11-03"),
      weightKg: 360,
      source: "AI from NDRI semen",
      notes: "First calver, growing well",
    },
  });

  // Seed milk logs for the last 14 days
  const today = new Date();
  for (let i = 13; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);

    const baseYield1 = 10 + Math.random() * 3;
    const baseYield2 = 16 + Math.random() * 4;
    const baseYield3 = 6 + Math.random() * 2;

    await db.milkLog.create({
      data: {
        cattleId: cattle1.id,
        date,
        morningKg: parseFloat((baseYield1 * 0.55).toFixed(1)),
        eveningKg: parseFloat((baseYield1 * 0.45).toFixed(1)),
        fatPct: 4.7 + Math.random() * 0.3,
      },
    });

    await db.milkLog.create({
      data: {
        cattleId: cattle2.id,
        date,
        morningKg: parseFloat((baseYield2 * 0.5).toFixed(1)),
        eveningKg: parseFloat((baseYield2 * 0.5).toFixed(1)),
        fatPct: 7.5 + Math.random() * 0.5,
      },
    });

    await db.milkLog.create({
      data: {
        cattleId: cattle3.id,
        date,
        morningKg: parseFloat((baseYield3 * 0.55).toFixed(1)),
        eveningKg: parseFloat((baseYield3 * 0.45).toFixed(1)),
        fatPct: 5.0 + Math.random() * 0.3,
      },
    });
  }

  // Health records
  await db.healthRecord.create({
    data: {
      cattleId: cattle1.id,
      date: new Date("2025-09-15"),
      type: "vaccination",
      event: "FMD Trivalent Vaccine",
      description: "Six-monthly booster. No adverse reaction.",
      cost: 50,
    },
  });
  await db.healthRecord.create({
    data: {
      cattleId: cattle2.id,
      date: new Date("2025-09-15"),
      type: "vaccination",
      event: "FMD Trivalent Vaccine",
      description: "Six-monthly booster.",
      cost: 50,
    },
  });
  await db.healthRecord.create({
    data: {
      cattleId: cattle1.id,
      date: new Date("2025-08-10"),
      type: "deworming",
      event: "Fenbendazole 7.5 g oral",
      description: "Routine quarterly deworming.",
      cost: 80,
    },
  });
  await db.healthRecord.create({
    data: {
      cattleId: cattle2.id,
      date: new Date("2025-05-20"),
      type: "calving",
      event: "Normal calving — male calf",
      description: "Calving uneventful. Calf healthy, birth weight 35 kg.",
    },
  });

  // Forum posts
  const p1 = await db.forumPost.create({
    data: {
      authorName: "Rajesh Patel",
      authorRole: "Dairy Farmer, Anand",
      topic: "Best fodder mix for Gir cows in summer?",
      breedTag: "Gir",
      body: "My Gir cow's yield drops by 30% in peak summer (May-June). Currently feeding maize fodder + wheat straw + 4 kg concentrate. Looking for advice on summer ration adjustments.",
    },
  });
  await db.forumReply.create({
    data: {
      postId: p1.id,
      authorName: "Dr. Anjali Sharma",
      authorRole: "Veterinary Officer",
      body: "Add 1 kg molasses to improve palatability and energy. Provide chilled drinking water (below 25°C) at least 4 times a day. Increase vitamin C to 5 g/day and add electrolytes. Consider subabul tree leaves as a drought-resistant protein source. Graze only between 5-9 AM and 5-7 PM.",
    },
  });
  await db.forumReply.create({
    data: {
      postId: p1.id,
      authorName: "Mahendra Singh",
      authorRole: "Farmer, Mehsana",
      body: "I plant cluster bean (guar) as summer fodder — it's drought-tolerant and high protein. Also use hydroponic barley sprouts (6-day growth) for 2 kg/day. Works well for my 8 Gir cows.",
    },
  });

  const p2 = await db.forumPost.create({
    data: {
      authorName: "Lakshmi Devi",
      authorRole: "Dairy Farmer, Chittoor",
      topic: "Punganur calf rearing — slow growth",
      breedTag: "Punganur",
      body: "My Punganur heifer is 8 months old and weighs only 45 kg. Is this normal for the breed? What should I feed for better growth?",
    },
  });
  await db.forumReply.create({
    data: {
      postId: p2.id,
      authorName: "Dr. KV Raghavendra",
      authorRole: "Veterinary University, Tirupati",
      body: "Punganur is a dwarf breed so 45 kg at 8 months is within normal range (adult female 110-180 kg). Focus on balanced protein (16-18% CP) calf starter, good quality berseem/lucerne hay, and 30 g/day mineral mixture. Target average daily gain of 200-250 g. Don't compare with crossbred growth rates.",
    },
  });

  const p3 = await db.forumPost.create({
    data: {
      authorName: "Suresh Kumar",
      authorRole: "Progressive Farmer, Rohtak",
      topic: "Murrah buffalo sold for Rs 1.2 crore at Hisar mela",
      breedTag: "Murrah Buffalo",
      body: "Sharing that a Murrah buffalo from my village (Kheri Sadh, Rohtak) was sold for Rs 1.2 crore at the Hisar animal fair. Key traits: 28 L/day peak yield, 8.2% fat, udder conformation score 9/10. This shows the potential of purebred Murrah breeding when done scientifically.",
    },
  });

  console.log("Seed completed!");
  console.log(`Created cattle: ${cattle1.name}, ${cattle2.name}, ${cattle3.name}`);
  console.log(`Forum posts: 3 with replies`);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await db.$disconnect();
  });
