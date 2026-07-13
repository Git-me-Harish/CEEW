// Seed script — populates the database with demo users, cattle, tickets, and vaccination requests
import bcrypt from "bcryptjs";
import { db } from "../src/lib/db";

async function main() {
  console.log("Seeding database...");

  // ─── Users ─────────────────────────────────────────────
  const passwordHash = await bcrypt.hash("demo123", 10);

  const admin = await db.user.create({
    data: {
      email: "admin@demo.in",
      name: "Platform Admin",
      passwordHash,
      role: "ADMIN",
      phone: "+91 98765 43210",
      location: "New Delhi, Delhi",
    },
  });

  const vet = await db.user.create({
    data: {
      email: "vet@demo.in",
      name: "Dr. Anjali Sharma",
      passwordHash,
      role: "VET",
      phone: "+91 98765 12345",
      location: "Anand, Gujarat",
    },
  });

  const farmer1 = await db.user.create({
    data: {
      email: "farmer@demo.in",
      name: "Rajesh Patel",
      passwordHash,
      role: "FARMER",
      phone: "+91 98765 67890",
      location: "Junagadh, Gujarat",
    },
  });

  const farmer2 = await db.user.create({
    data: {
      email: "suresh@demo.in",
      name: "Suresh Kumar",
      passwordHash,
      role: "FARMER",
      phone: "+91 98765 11111",
      location: "Rohtak, Haryana",
    },
  });

  console.log(`Created 4 users: admin, vet, farmer1, farmer2`);

  // ─── Cattle ────────────────────────────────────────────
  const cattle1 = await db.cattle.create({
    data: {
      ownerId: farmer1.id,
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
      ownerId: farmer1.id,
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
      ownerId: farmer2.id,
      tagNumber: "HR-MUR-001",
      name: "Rohtak Rani",
      breed: "Murrah Buffalo",
      species: "buffalo",
      sex: "female",
      birthDate: new Date("2018-06-10"),
      weightKg: 620,
      source: "Hisar cattle fair",
      notes: "Champion bloodline — 22 L/day peak",
    },
  });

  console.log(`Created 3 cattle`);

  // ─── Milk logs (14 days × 2 cattle for farmer1) ────────
  const today = new Date();
  for (let i = 13; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);

    const baseYield1 = 10 + Math.random() * 3;
    const baseYield2 = 16 + Math.random() * 4;

    await db.milkLog.create({
      data: {
        cattleId: cattle1.id,
        recordedById: farmer1.id,
        date,
        morningKg: parseFloat((baseYield1 * 0.55).toFixed(1)),
        eveningKg: parseFloat((baseYield1 * 0.45).toFixed(1)),
        fatPct: 4.7 + Math.random() * 0.3,
      },
    });

    await db.milkLog.create({
      data: {
        cattleId: cattle2.id,
        recordedById: farmer1.id,
        date,
        morningKg: parseFloat((baseYield2 * 0.5).toFixed(1)),
        eveningKg: parseFloat((baseYield2 * 0.5).toFixed(1)),
        fatPct: 7.5 + Math.random() * 0.5,
      },
    });
  }
  console.log(`Created 28 milk logs (14 days × 2 cattle)`);

  // ─── Health records ────────────────────────────────────
  await db.healthRecord.create({
    data: {
      cattleId: cattle1.id,
      recordedById: vet.id,
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
      recordedById: vet.id,
      date: new Date("2025-09-15"),
      type: "vaccination",
      event: "FMD Trivalent Vaccine",
      description: "Six-monthly booster.",
      cost: 50,
    },
  });

  // ─── Tickets (problems) ────────────────────────────────
  const ticket1 = await db.ticket.create({
    data: {
      authorId: farmer1.id,
      title: "Gir cow's milk yield dropped 30% in summer — please advise",
      body:
        "My Gir cow Gauri (IND-GIR-001) was giving 11 L/day until March. Since mid-April she's down to 7-8 L/day. She's eating normally and shows no other symptoms. I'm in Junagadh where temperatures hit 42°C. Currently feeding maize fodder + wheat straw + 4 kg concentrate. Is this heat stress? Should I change the ration? Any supplements I should add?",
      category: "nutrition",
      priority: "HIGH",
      breedTag: "Gir",
      cattleTag: "IND-GIR-001",
      status: "IN_PROGRESS",
    },
  });

  await db.ticketReply.create({
    data: {
      ticketId: ticket1.id,
      authorId: vet.id,
      body:
        "Rajesh ji, this is classic heat stress. Gir is heat-tolerant but 42°C with humidity will still drop yields by 25-30%. Here's what I recommend:\n\n1. Add 1 kg molasses to improve energy intake and palatability\n2. Provide chilled drinking water (below 25°C) at least 4 times a day — keep the tank in shade\n3. Increase vitamin C to 5 g/day and add electrolytes to water\n4. Graze only between 5-9 AM and 5-7 PM\n5. Consider Subabul tree leaves as a drought-resistant protein source\n\nYour ration is otherwise fine. Try these for 2 weeks and let me know the change. I'm marking this in-progress.",
      internal: false,
    },
  });

  const ticket2 = await db.ticket.create({
    data: {
      authorId: farmer2.id,
      title: "Murrah buffalo — repeating breeding, not conceiving",
      body:
        "My Murrah buffalo Rohtak Rani has been bred 3 times via AI in the last 4 months but isn't conceiving. Semen quality from the centre is supposed to be good. She had a normal calving 8 months ago. Please advise on what to check.",
      category: "breeding",
      priority: "HIGH",
      breedTag: "Murrah Buffalo",
      cattleTag: "HR-MUR-001",
      status: "OPEN",
    },
  });

  const ticket3 = await db.ticket.create({
    data: {
      authorId: farmer1.id,
      title: "Which government scheme covers Murrah buffalo purchase subsidy?",
      body:
        "I want to buy 2 more Murrah buffaloes to expand my herd. Are there any central or Gujarat state schemes that provide subsidy for buffalo purchase by small farmers?",
      category: "market",
      priority: "MEDIUM",
      status: "RESOLVED",
    },
  });

  await db.ticketReply.create({
    data: {
      ticketId: ticket3.id,
      authorId: vet.id,
      body:
        "Yes — look into the Dairy Entrepreneurship Development Scheme (DEDS) operated by NABARD. It provides 25% back-ended subsidy (33.33% for SC/ST/women) on projects up to Rs 20 lakh for dairy farms. Also check the Pashu Kisan Credit Card for working capital at 4% effective interest. Visit your local NABARD office or apply through a commercial bank.",
      internal: false,
    },
  });

  console.log(`Created 3 tickets with replies`);

  // ─── Vaccination requests ──────────────────────────────
  await db.vaccinationRequest.create({
    data: {
      requesterId: farmer1.id,
      cattleId: cattle1.id,
      vaccineName: "FMD trivalent (O, A, Asia1) — Rakshafmd",
      requestedDate: new Date(Date.now() + 7 * 86400000),
      preferredTime: "Morning",
      notes: "Six-monthly booster due. Cow is healthy, no current medications.",
      status: "PENDING",
    },
  });

  await db.vaccinationRequest.create({
    data: {
      requesterId: farmer1.id,
      cattleId: cattle2.id,
      vaccineName: "HS oil-adjuvant vaccine",
      requestedDate: new Date(Date.now() + 14 * 86400000),
      preferredTime: "Evening",
      notes: "Pre-monsoon vaccination. Buffalo had mild reaction last time — please observe for 30 min after.",
      status: "PENDING",
    },
  });

  await db.vaccinationRequest.create({
    data: {
      requesterId: farmer2.id,
      cattleId: cattle3.id,
      vaccineName: "Brucella abortus strain 19",
      requestedDate: new Date(Date.now() + 5 * 86400000),
      notes: "Female calf 6 months old — first Brucellosis vaccination.",
      status: "PENDING",
    },
  });

  console.log(`Created 3 vaccination requests (all pending)`);

  // ─── Notifications for vet ─────────────────────────────
  await db.notification.createMany({
    data: [
      {
        userId: vet.id,
        type: "ticket_new",
        title: "New ticket: Gir cow's milk yield dropped",
        message: "Rajesh Patel posted a HIGH priority problem.",
        link: "/tickets",
      },
      {
        userId: vet.id,
        type: "ticket_new",
        title: "New ticket: Murrah buffalo repeating breeding",
        message: "Suresh Kumar posted a HIGH priority problem.",
        link: "/tickets",
      },
      {
        userId: vet.id,
        type: "vaccination_new",
        title: "3 new vaccination requests",
        message: "Farmers have requested vaccinations — review pending queue.",
        link: "/vaccination-requests",
      },
      {
        userId: admin.id,
        type: "ticket_new",
        title: "2 new tickets need attention",
        message: "High priority problems posted by farmers.",
        link: "/tickets",
      },
    ],
  });

  console.log(`Created notifications`);
  console.log(`\nSeed completed!`);
  console.log(`\nDemo login credentials:`);
  console.log(`  Farmer: farmer@demo.in / demo123`);
  console.log(`  Vet:    vet@demo.in / demo123`);
  console.log(`  Admin:  admin@demo.in / demo123`);
}

main()
  .catch((e) => {
    console.error(e);
    process.exit(1);
  })
  .finally(async () => {
    await db.$disconnect();
  });
