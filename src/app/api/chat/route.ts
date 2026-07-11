import { NextRequest, NextResponse } from "next/server";
import ZAI from "z-ai-web-dev-sdk";
import { breeds } from "@/data/breeds";
import { diseases } from "@/data/health";
import { govtSchemes } from "@/data/schemes";

export const runtime = "nodejs";
export const maxDuration = 60;

// Build a compact context string of bovine knowledge
function buildContext(): string {
  const breedList = breeds
    .map(
      (b) =>
        `- ${b.name} (${b.type}, ${b.category}): ${b.origin}. Milk ${b.milkYieldKgPerLactation} kg/lactation, ${b.fatContent}% fat. Use: ${b.primaryUse}. Heat tolerance: ${b.heatTolerance}.`
    )
    .join("\n");

  const diseaseList = diseases
    .map(
      (d) =>
        `- ${d.name} (${d.category}, severity ${d.severity}): Symptoms include ${d.symptoms
          .slice(0, 4)
          .join(", ")}. Prevention: ${d.prevention[0]}.`
    )
    .join("\n");

  const schemeList = govtSchemes
    .map(
      (s) =>
        `- ${s.name} (${s.category}, ${s.subsidyPct}): ${s.summary}`
    )
    .join("\n");

  return `INDIAN BOVINE KNOWLEDGE BASE
==========================

BREEDS (${breeds.length} total):
${breedList}

DISEASES (${diseases.length} total):
${diseaseList}

GOVERNMENT SCHEMES (${govtSchemes.length} total):
${schemeList}

KEY FACTS ABOUT INDIAN BOVINE SECTOR:
- India has the world's largest cattle and buffalo population (~303 million bovines)
- India is the world's largest milk producer (230+ million tonnes/year)
- Average milk yield of indigenous cattle is ~3 kg/day vs crossbred ~8 kg/day
- Buffalo milk contributes 49% of India's total milk production
- Average dairy herd size in India is 2-3 animals
- Major dairy states: Uttar Pradesh, Rajasthan, Madhya Pradesh, Andhra Pradesh, Gujarat, Punjab, Haryana
- Common feeding: 60% roughage (green + dry) + 40% concentrate mix
- Important vaccines: FMD (every 6 months), HS (annual), BQ (annual), Brucellosis (once in female calves)`;
}

const SYSTEM_PROMPT = `You are PashuMitra, an expert AI assistant for Indian bovine (cattle and buffalo) management. You help farmers, dairy owners, and veterinarians with practical, actionable advice on:

1. Breed identification, characteristics, and selection
2. Health management, disease diagnosis support, and treatment guidance
3. Nutrition, feeding practices, and fodder cultivation
4. Reproduction, breeding, AI, and pregnancy management
5. Milk production optimisation
6. Government schemes, subsidies, and insurance
7. Market prices and economics of dairy farming
8. Calf rearing, heifer management, and animal welfare

GUIDELINES:
- Always respond in the user's language (Hindi/English/Hinglish — match their style)
- Be practical and actionable for Indian smallholder and medium dairy farmers
- Reference specific Indian breeds (Gir, Sahiwal, Murrah, etc.) and Indian conditions
- For disease symptoms, always recommend consulting a veterinarian for diagnosis
- For treatments, mention both modern veterinary medicine and traditional practices where appropriate
- Mention costs in Indian Rupees (₹) when relevant
- Reference Indian government schemes, NDDB, NDRI, ICAR, state AH departments
- Be concise but thorough — give complete, usable answers
- If you don't know something, admit it and suggest contacting local veterinary officer
- Always prioritise animal welfare and food safety

${buildContext()}`;

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { message, history = [] } = body;

    if (!message || typeof message !== "string") {
      return NextResponse.json(
        { error: "Message is required and must be a string." },
        { status: 400 }
      );
    }

    const zai = await ZAI.create();

    // Build conversation history
    const messages: Array<{ role: "system" | "user" | "assistant"; content: string }> = [
      { role: "system", content: SYSTEM_PROMPT },
      ...history.slice(-6).map((h: { role: string; content: string }) => ({
        role: h.role === "user" ? ("user" as const) : ("assistant" as const),
        content: h.content,
      })),
      { role: "user", content: message },
    ];

    const response = await zai.chat.completions.create({
      messages,
      temperature: 0.7,
      max_tokens: 1024,
      thinking: { type: "disabled" },
    });

    const reply =
      response.choices[0]?.message?.content ||
      "I'm sorry, I couldn't generate a response. Please rephrase your question.";

    return NextResponse.json({
      response: reply,
      timestamp: new Date().toISOString(),
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("Chat API error:", message);
    return NextResponse.json(
      { error: `Chat failed: ${message}` },
      { status: 500 }
    );
  }
}
