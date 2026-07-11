import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";

export const runtime = "nodejs";

export async function GET() {
  try {
    const posts = await db.forumPost.findMany({
      orderBy: { createdAt: "desc" },
      include: { _count: { select: { replies: true } } },
    });
    return NextResponse.json({ posts });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { authorName, authorRole, topic, breedTag, body: postBody } = body;

    if (!authorName || !topic || !postBody) {
      return NextResponse.json(
        { error: "authorName, topic, and body are required." },
        { status: 400 }
      );
    }

    const post = await db.forumPost.create({
      data: {
        authorName,
        authorRole: authorRole || "Farmer",
        topic,
        breedTag: breedTag || null,
        body: postBody,
      },
    });
    return NextResponse.json({ post });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
