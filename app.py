import os
import io
import json
import uuid
import hashlib
import requests
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import pytesseract
import google.generativeai as genai
import re

# Configure Gemini API
genai.configure(api_key="AIzaSyDOv1dpaKoxyT66F53p5HCpIRjFE")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

session_text_store = {}
summary_store = {}

# ---------- Utility Functions ----------

def get_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()

def remove_degrees_and_clean_entities(text: str) -> str:
    text = re.sub(r'\b(B\.?Tech|M\.?Tech|Ph\.?D|MBA|MD|MBBS|LLB|BSc|MSc)\b', '', text)
    return re.sub(r'\s+', ' ', text)

def get_metrics():
    if not os.path.exists("metrics.json"):
        return {"uploads": 0, "feedbacks": 0, "avg_confidence": 0.0}
    with open("metrics.json", "r") as f:
        return json.load(f)

def save_metrics(metrics):
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

def update_upload_metrics(confidence):
    metrics = get_metrics()
    metrics["uploads"] += 1
    metrics["avg_confidence"] = round(
        ((metrics["avg_confidence"] * (metrics["uploads"] - 1)) + confidence) / metrics["uploads"], 2
    )
    save_metrics(metrics)

def update_feedback_metrics():
    metrics = get_metrics()
    metrics["feedbacks"] += 1
    save_metrics(metrics)

# ---------- Routes ----------

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/metrics", response_class=HTMLResponse)
async def show_metrics(request: Request):
    data = get_metrics()
    return templates.TemplateResponse("metrics.html", {"request": request, "metrics": data})

@app.post("/summarize/")
async def summarize_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = extract_text(file, content)

        if not text or len(text.strip()) < 5:
            return JSONResponse(status_code=400, content={"error": "No text extracted from file."})

        text = remove_degrees_and_clean_entities(text)
        text_hash = get_text_hash(text)

        if text_hash in summary_store:
            stored = summary_store[text_hash]
            return {
                "filename": file.filename,
                "summary": stored["summary"],
                "confidence": stored["confidence"],
                "original_text": text,
                "session_id": stored["session_id"]
            }

        summary = summarize_with_gemini(text)

        if not summary or summary == "Summary generation failed.":
            return JSONResponse(status_code=500, content={"error": "Gemini failed to generate summary."})

        confidence = get_semantic_confidence(text, summary)
        update_upload_metrics(confidence)

        session_id = str(uuid.uuid4())
        session_text_store[session_id] = text
        summary_store[text_hash] = {
            "summary": summary,
            "confidence": confidence,
            "session_id": session_id
        }

        return {
            "filename": file.filename,
            "summary": summary,
            "confidence": confidence,
            "original_text": text,
            "session_id": session_id
        }

    except Exception as e:
        print("Summarize error:", str(e))
        return JSONResponse(status_code=500, content={"error": f"Summary error: {str(e)}"})

@app.post("/feedback/")
async def receive_feedback(request: Request):
    try:
        feedback = await request.json()
        with open("feedback_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback) + "\n")
        update_feedback_metrics()
        return {"message": "Feedback recorded"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to save feedback: {str(e)}"})

class ChatRequest(BaseModel):
    session_id: str
    question: str

@app.post("/chat/")
async def chat_with_document(request: Request):
    try:
        data = await request.json()
        session_id = data.get("session_id")
        question = data.get("question")

        if not session_id or not question:
            return JSONResponse(status_code=400, content={"error": "session_id and question required"})

        context = session_text_store.get(session_id, "")
        if not context:
            return JSONResponse(status_code=400, content={"error": "No document text found for this session."})

        prompt = f"""Use the following document content to answer the user's question.

Document:
{context}

Question:
{question}

Answer in this format:
Answer: [your answer]
Source: [mention page number or section title if available]
Snippet: [exact sentence or phrase from document used to generate the answer]
"""

        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 512}
        )

        raw_answer = response.text.strip() if response.text else "Sorry, no answer generated."
        parts = raw_answer.split("\n\n")
        answer_text = parts[0].replace("Answer:", "").strip() if len(parts) > 0 else raw_answer
        source_text = parts[1].replace("Source:", "").strip() if len(parts) > 1 else "Not specified"
        snippet_text = parts[2].replace("Snippet:", "").strip() if len(parts) > 2 else ""

        return {
            "answer": answer_text,
            "source": source_text,
            "snippet": snippet_text
        }

    except Exception as e:
        print("Chat error:", str(e))
        return JSONResponse(status_code=500, content={"error": f"Chat error: {str(e)}"})

# ---------- Core Logic Functions ----------

def summarize_with_gemini(text: str) -> str:
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    trimmed_text = text[:3000]

    try:
        keyword_prompt = f"""
Extract important project and technical terms from the document below.
Avoid filler words. No duplicates.

Document:
{trimmed_text}

Output:
Keywords: [comma-separated list of unique, non-repeating keywords]
"""
        keyword_response = model.generate_content(keyword_prompt)
        raw_keywords = keyword_response.text.strip().replace("Keywords:", "").strip()
        unique_keywords = list(dict.fromkeys([kw.strip() for kw in raw_keywords.split(",") if kw.strip()]))
        filtered_keywords = [kw for kw in unique_keywords if len(kw) > 2]
        keywords = ", ".join(filtered_keywords)

        summary_prompt = f"""
Write a concise 5-7 sentence summary covering:
- Project idea
- Technologies/methods used
- Problems solved
- Use these keywords once: {keywords}

Avoid using filler words. Use clear language.

Document:
{trimmed_text}

Output format:
Summary:
[your summary here]
"""
        summary_response = model.generate_content(summary_prompt)
        summary_text = summary_response.text.strip()

        if summary_text.lower().startswith("summary:"):
            summary_text = summary_text[8:].strip()

        summary_text = remove_degrees_and_clean_entities(summary_text)
        return summary_text

    except Exception as e:
        print("Gemini error:", str(e))
        return "Summary generation failed."

def get_semantic_confidence(text: str, summary: str) -> float:
    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
        prompt = f"""
Evaluate how well the summary captures the meaning of the document.

Document:
{text[:3000]}

Summary:
{summary}

Give a numeric score from 0.0 (poor) to 1.0 (excellent), based on completeness and relevance.
Only output a number.
"""
        response = model.generate_content(prompt)
        score = response.text.strip()
        return round(float(score), 2) if score.replace('.', '', 1).isdigit() else 0.5
    except Exception as e:
        print("Semantic confidence error:", str(e))
        return 0.5

def extract_text(file: UploadFile, content: bytes) -> str:
    ext = file.filename.split(".")[-1].lower()
    text = ""
    try:
        if ext == "pdf":
            with open("temp.pdf", "wb") as f:
                f.write(content)
            reader = PdfReader("temp.pdf")
            for page in reader.pages:
                text += page.extract_text() or ""
            os.remove("temp.pdf")

        elif ext == "docx":
            with open("temp.docx", "wb") as f:
                f.write(content)
            doc = Document("temp.docx")
            text = "\n".join([para.text for para in doc.paragraphs])
            os.remove("temp.docx")

        elif ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
            response = requests.post(
                "https://api.ocr.space/parse/image",
                files={"file": (file.filename, io.BytesIO(content))},
                data={"apikey": "K83146264688957", "language": "eng"},
            )
            result = response.json()
            if result.get("IsErroredOnProcessing"):
                raise ValueError("OCR API error: " + result.get("ErrorMessage", ["Unknown error"])[0])
            parsed_results = result.get("ParsedResults", [])
            text = parsed_results[0]["ParsedText"] if parsed_results else ""

        else:
            raise ValueError("Unsupported file type.")

        return text.strip()

    except Exception as e:
        print(f"Text extraction error: {e}")
        return ""