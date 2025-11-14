import os
import re
import uuid
import fitz  # PyMuPDF
import json
import requests
from pydantic import BaseModel, ValidationError, Field
from typing import List, Dict, Any, Optional
from google.auth import default
from google.auth.transport.requests import Request

from google.cloud import storage, firestore
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic import PredictionServiceClient, IndexEndpointServiceClient
# Note: we no longer rely on IndexEndpointServiceClient.upsert_datapoints
# but keep it for other uses if required.

import functions_framework
from cloudevents.http import CloudEvent
from google.events.cloud.storage import StorageObjectData

# --- Configuration ---
PROJECT_ID = "rezumai-478204"
REGION = "us-central1"
CANDIDATE_COLLECTION = "candidates"
VECTOR_SEARCH_ENDPOINT_ID = "9074104537291161600"
VECTOR_SEARCH_DEPLOYED_INDEX_ID = "resume_index_v1_1763142505680"
VECTOR_SEARCH_INDEX_ID = "9048727808922091520"

# --- Initialize Google Cloud Clients (Global Scope) ---
try:
    aiplatform.init(project=PROJECT_ID, location=REGION)

    storage_client = storage.Client()
    db = firestore.Client()

    # Full resource name used in REST call too:
    vs_endpoint_name = f"projects/{PROJECT_ID}/locations/{REGION}/indexEndpoints/{VECTOR_SEARCH_ENDPOINT_ID}"
    vs_client_options = {"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
    # Keep the client if you need other client methods; not used for upsert here.
    vs_endpoint_client = IndexEndpointServiceClient(client_options=vs_client_options)
    embedding_client = PredictionServiceClient(client_options=vs_client_options)

    credentials, _ = default()

except Exception as e:
    print(f"Error initializing Google Cloud clients: {e}")
    raise

# ---------- Pydantic models ----------
class WorkExperience(BaseModel):
    company: str = ""
    title: str = ""
    start_date: str = ""
    end_date: str = ""
    description: str = ""


class Education(BaseModel):
    institution: str = ""
    degree: str = ""
    graduation_date: str = ""


class Project(BaseModel):
    name: str = ""
    description: str = ""


class ResumeSchema(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    summary: str = ""
    skills: List[str] = Field(default_factory=list)
    work_experience: List[WorkExperience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)


# --- Prompt Constants ---
INGESTION_RESUME_PARSING_PROMPT = """
You are an expert HR resume parser. Analyze the following resume text.

Extract the information and return **only** a valid JSON object matching this exact schema. If a field or value is not present, return an empty string `""` or an empty list `[]`.

**SCHEMA:**
{{
  "name": "string",
  "email": "string",
  "phone": "string",
  "summary": "string",
  "skills": ["string", "string"],
  "work_experience": [
    {{
      "company": "string",
      "title": "string",
      "start_date": "string",
      "end_date": "string",
      "description": "string"
    }}
  ],
  "education": [
    {{
      "institution": "string",
      "degree": "string",
      "graduation_date": "string"
    }}
  ],
  "projects": [
    {{
      "name": "string",
      "description": "string"
    }}
  ]
}}

**RESUME TEXT:**
---
{resume_text}
---

**JSON OUTPUT:**
"""


# --- Helper Functions ---

def parse_gcs_path(path: str) -> tuple:
    """
    Parses the GCS path to get RECRUITER_UUID and BATCH_TAG.
    Expected path: [RECRUITER_UUID]/[BATCH_TAG]/resume.pdf
    """
    parts = path.split('/')
    if len(parts) >= 3:
        recruiter_uuid = parts[0]
        batch_tag = parts[1]
        return recruiter_uuid, batch_tag
    return None, None


def extract_text_from_pdf(bucket_name: str, file_name: str) -> str:
    """Downloads PDF from GCS and extracts text using PyMuPDF."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    try:
        pdf_bytes = blob.download_as_bytes()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        print(f"Successfully extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        print(f"Error extracting text from {file_name}: {e}")
        raise


# ---------- Robust JSON extraction ----------
def _extract_first_balanced_json(s: str) -> Optional[str]:
    """
    Find the first balanced {...} JSON object in s.
    Handles escape sequences and braces inside strings properly.
    """
    start = s.find('{')
    if start == -1:
        return None

    in_string = False
    escape_next = False
    stack = 0

    for i in range(start, len(s)):
        ch = s[i]

        if escape_next:
            escape_next = False
            continue

        if ch == '\\':
            escape_next = True
            continue

        if ch == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if ch == '{':
                stack += 1
            elif ch == '}':
                stack -= 1
                if stack == 0:
                    return s[start:i + 1]

    return None


def parse_resume_with_gemini(raw_text: str) -> ResumeSchema:
    """
    Calls Gemini 1.5 Flash via generateContent REST API endpoint.
    Handles proper request/response format and JSON extraction.
    """
    prompt = INGESTION_RESUME_PARSING_PROMPT.format(resume_text=raw_text)

    print("[Gemini] Step 1: Building request payload...")

    request_payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generation_config": {
            "temperature": 0.0,
            "response_mime_type": "application/json"
        }
    }

    api_url = (
        f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/"
        f"locations/{REGION}/publishers/google/models/gemini-2.0-flash-001:generateContent"
    )

    print(f"[Gemini] Step 2: Calling API endpoint: {api_url}")

    raw_response_text = ""
    cleaned_json_text = ""
    parsed_json = {}

    try:
        credentials.refresh(Request())
        headers = {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json"
        }

        print("[Gemini] Step 3: Sending HTTP request to generateContent endpoint...")
        response = requests.post(api_url, json=request_payload, headers=headers, timeout=60)

        print(f"[Gemini] Step 4: Received response status code: {response.status_code}")

        if response.status_code != 200:
            print(f"[Gemini] Error response body: {response.text}")
            raise Exception(f"API returned status {response.status_code}: {response.text}")

        response_json = response.json()
        print(f"[Gemini] Step 5: Response structure keys: {list(response_json.keys())}")

        if "candidates" not in response_json or len(response_json["candidates"]) == 0:
            print("[Gemini] Error: No candidates in response")
            print(f"[Gemini] Full response: {json.dumps(response_json, ensure_ascii=False)[:1000]}")
            raise Exception("No candidates in Gemini response")

        candidate = response_json["candidates"][0]
        print(f"[Gemini] Step 6: Candidate structure keys: {list(candidate.keys())}")

        if "content" not in candidate or "parts" not in candidate["content"]:
            print("[Gemini] Error: No content/parts in candidate")
            print(f"[Gemini] Candidate structure: {json.dumps(candidate, ensure_ascii=False)[:1000]}")
            raise Exception("No content/parts in candidate")

        parts = candidate["content"]["parts"]
        if len(parts) == 0 or "text" not in parts[0]:
            print("[Gemini] Error: No text in parts")
            raise Exception("No text content in response parts")

        raw_response_text = parts[0]["text"]
        print(f"[Gemini] Step 7: Extracted raw text ({len(raw_response_text)} chars)")
        print(f"[Gemini] First 500 chars of response: {raw_response_text[:500]}")

        print("[Gemini] Step 8: Extracting JSON from response...")

        m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_response_text, re.IGNORECASE)

        if m:
            print("[Gemini] Found fenced JSON code block")
            candidate_json = m.group(1).strip()
            balanced = _extract_first_balanced_json(candidate_json)
            cleaned_json_text = balanced if balanced else candidate_json
        else:
            print("[Gemini] No fenced block found, searching for balanced JSON...")
            balanced = _extract_first_balanced_json(raw_response_text)
            if balanced:
                print("[Gemini] Found balanced JSON object")
                cleaned_json_text = balanced
            else:
                print("[Gemini] No balanced JSON found, using entire response")
                cleaned_json_text = raw_response_text.strip()

        cleaned_json_text = cleaned_json_text.strip()
        print(f"[Gemini] Step 9: Cleaned JSON text ({len(cleaned_json_text)} chars)")
        print(f"[Gemini] First 300 chars of cleaned JSON: {cleaned_json_text[:300]}")

        print("[Gemini] Step 10: Attempting to parse JSON...")
        try:
            parsed_json = json.loads(cleaned_json_text)
            print("[Gemini] JSON parsed successfully")
        except json.JSONDecodeError as e:
            print(f"[Gemini] JSON decode failed: {e}")
            # Attempt common fixes (smart quotes -> straight quotes)
            tentative = cleaned_json_text.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")
            tentative = tentative.strip()
            try:
                parsed_json = json.loads(tentative)
                print("[Gemini] JSON parsed after simple quote fixes")
            except Exception as e2:
                print(f"[Gemini] Quote fix failed: {e2}")
                # Provide helpful debug info before propagating
                start = max(0, e.pos - 50) if hasattr(e, 'pos') else 0
                end = min(len(cleaned_json_text), (e.pos + 50) if hasattr(e, 'pos') else len(cleaned_json_text))
                context = cleaned_json_text[start:end]
                print(f"[Gemini] Context around original error: ...{context}...")
                raise json.JSONDecodeError(
                    f"Could not parse JSON: {e}",
                    cleaned_json_text,
                    e.pos if hasattr(e, 'pos') else 0
                )

        print("[Gemini] Step 11: Validating JSON structure...")
        if isinstance(parsed_json, list) and len(parsed_json) > 0 and isinstance(parsed_json[0], dict):
            print("[Gemini] Response is a list, extracting first element")
            parsed_json = parsed_json[0]

        if not isinstance(parsed_json, dict):
            print(f"[Gemini] Warning: Parsed JSON is {type(parsed_json)}, coercing to empty dict")
            parsed_json = {}

        print("[Gemini] Step 12: Creating ResumeSchema from parsed JSON...")
        validated_data = ResumeSchema.parse_obj(parsed_json)
        print(f"[Gemini] Successfully created ResumeSchema: name='{validated_data.name}'")

        return validated_data

    except ValidationError as e:
        print(f"[Gemini] Pydantic ValidationError: {e}")
        print(f"[Gemini] Parsed JSON (truncated): {json.dumps(parsed_json, ensure_ascii=False)[:1000]}")
        raise
    except Exception as e:
        print(f"[Gemini] Unexpected error: {type(e).__name__}: {e}")
        print(f"[Gemini] Raw response (truncated): {raw_response_text[:1000]}")
        print(f"[Gemini] Cleaned JSON (truncated): {cleaned_json_text[:1000]}")
        raise


def chunk_parsed_resume(candidate_id: str, parsed_data: ResumeSchema) -> List[Dict[str, str]]:
    """Creates logical text chunks from the PARSED JSON data."""
    chunks = []

    if parsed_data.summary:
        chunks.append({
            "id": f"{candidate_id}_summary",
            "text_content": f"Candidate: {parsed_data.name}. Summary: {parsed_data.summary}"
        })

    if parsed_data.skills:
        chunks.append({
            "id": f"{candidate_id}_skills",
            "text_content": f"Skills for {parsed_data.name}: {', '.join(parsed_data.skills)}"
        })

    for i, job in enumerate(parsed_data.work_experience):
        job_text = (
            f"Job: {job.title} at {job.company}. "
            f"Dates: {job.start_date} to {job.end_date}. "
            f"Description: {job.description}"
        )
        chunks.append({"id": f"{candidate_id}_work_{i}", "text_content": job_text})

    for i, edu in enumerate(parsed_data.education):
        edu_text = (
            f"Education: {edu.degree} from {edu.institution}. "
            f"Graduated: {edu.graduation_date}"
        )
        chunks.append({"id": f"{candidate_id}_edu_{i}", "text_content": edu_text})

    for i, project in enumerate(parsed_data.projects):
        project_text = (
            f"Project: {project.name}. "
            f"Description: {project.description}"
        )
        chunks.append({"id": f"{candidate_id}_project_{i}", "text_content": project_text})

    return chunks


def get_embeddings(text_chunks: List[str]) -> List[List[float]]:
    """Gets text embeddings from Vertex AI using text-embedding-004 (768 dims)."""

    api_url = (
        f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/"
        f"locations/{REGION}/publishers/google/models/text-embedding-004:predict"
    )

    instances = [{"content": text} for text in text_chunks]
    request_payload = {
        "instances": instances
    }

    try:
        print(f"[Embeddings] Step 1: Requesting embeddings for {len(text_chunks)} chunks...")

        credentials.refresh(Request())
        headers = {
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json"
        }

        response = requests.post(api_url, json=request_payload, headers=headers, timeout=60)

        print(f"[Embeddings] Step 2: Received response status code: {response.status_code}")

        if response.status_code != 200:
            print(f"[Embeddings] Error: {response.status_code} - {response.text}")
            raise Exception(f"Embedding API returned status {response.status_code}: {response.text}")

        response_json = response.json()

        print(f"[Embeddings] Step 3: Parsing {len(response_json.get('predictions', []))} predictions...")

        embeddings = []
        for idx, pred in enumerate(response_json.get("predictions", [])):
            if "embeddings" in pred:
                # Some models return embeddings in different shapes; handle common ones.
                values = pred["embeddings"].get("values", None)
                if values is None:
                    # maybe the embedding is directly a list under 'embeddings'
                    maybe_list = pred.get("embeddings")
                    if isinstance(maybe_list, list):
                        float_values = [float(v) for v in maybe_list]
                    else:
                        raise Exception(f"Unrecognized embedding format at prediction {idx}: {pred}")
                else:
                    float_values = [float(v) for v in values]

                embeddings.append(float_values)
                if idx == 0:
                    print(f"[Embeddings] First embedding dimension: {len(float_values)}")
                    print(f"[Embeddings] First 5 values: {float_values[:5]}")
            else:
                # Another possible format: prediction is directly a list of floats
                if isinstance(pred, list):
                    float_values = [float(v) for v in pred]
                    embeddings.append(float_values)
                else:
                    print(f"[Embeddings] Warning: Prediction {idx} has unexpected format: {pred}")
                    raise Exception(f"Unexpected prediction format at index {idx}: {pred}")

        print(f"[Embeddings] Step 4: Successfully generated {len(embeddings)} embeddings")

        return embeddings

    except Exception as e:
        print(f"[Embeddings] Error getting embeddings: {e}")
        import traceback
        traceback.print_exc()
        raise


def upsert_to_vector_search(datapoints: List[Dict[str, Any]]):
    """
    Upserts datapoints to Vertex AI Vector Search using the REST API.
    """
    try:
        if not datapoints:
            print("[Vector Search] No datapoints provided to upsert.")
            return

        # Refresh token
        credentials.refresh(Request())
        token = credentials.token

        # Correct INDEX resource path
        index_resource = f"projects/{PROJECT_ID}/locations/{REGION}/indexes/{VECTOR_SEARCH_INDEX_ID}"
        url = f"https://{REGION}-aiplatform.googleapis.com/v1/{index_resource}:upsertDatapoints"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        body_datapoints = []
        for dp in datapoints:

            feature_vector = [float(x) for x in dp["feature_vector"]]

            restricts_input = dp.get("restricts", [])
            rest_serialized = []
            for r in restricts_input:
                namespace = r.get("namespace")
                allow_list = r.get("allow_list") or []
                rest_serialized.append({
                    "namespace": namespace,
                    "allowList": list(allow_list)
                })

            body_datapoints.append({
                "datapointId": dp["datapoint_id"],
                "featureVector": feature_vector,
                "restricts": rest_serialized
            })

        request_body = {
            "datapoints": body_datapoints
        }

        print(f"[Vector Search] Upserting {len(body_datapoints)} datapoints to index {VECTOR_SEARCH_INDEX_ID} ...")
        response = requests.post(url, headers=headers, json=request_body, timeout=60)

        print(f"[Vector Search] REST call status: {response.status_code}")
        if response.status_code not in (200, 201):
            print(f"[Vector Search] ERROR response: {response.text}")
            raise Exception(f"Vector Search upsert failed: {response.status_code}: {response.text}")

        print("[Vector Search] Upsert successful.")
        return response.json() if response.text else {}

    except Exception as e:
        print(f"[Vector Search] Error upserting to Vector Search: {e}")
        import traceback
        traceback.print_exc()
        raise


# --- Cloud Function Entrypoint ---

@functions_framework.cloud_event
def process_resume_upload(cloud_event: CloudEvent):
    """
    This function is triggered by a GCS 'object.finalized' event.
    """
    bucket_name = cloud_event.data.get("bucket")
    file_name = cloud_event.data.get("name")

    if not file_name:
        print("No file name in event. Exiting.")
        return

    print(f"Received event for file: gs://{bucket_name}/{file_name}")

    try:
        # 1. Parse Path
        recruiter_uuid, batch_tag = parse_gcs_path(file_name)
        if not recruiter_uuid or not batch_tag:
            print(f"File path {file_name} does not match expected format. Skipping.")
            return

        print(f"Parsed path - Recruiter: {recruiter_uuid}, Batch: {batch_tag}")

        # 2. Extract Text
        print(f"Step 2: Extracting text from {file_name}...")
        raw_text = extract_text_from_pdf(bucket_name, file_name)

        if not raw_text or raw_text.strip() == "":
            print(f"File {file_name} is empty or unreadable. Skipping.")
            return

        # 3. Parse with LLM (Gemini Call #1)
        print("Step 3: Parsing resume with Gemini...")
        parsed_data = parse_resume_with_gemini(raw_text)

        # 4. Save to Firestore
        print("Step 4: Saving full profile to Firestore...")
        candidate_id = f"cnd_{uuid.uuid4()}"
        resume_gcs_url = f"gs://{bucket_name}/{file_name}"

        firestore_doc = {
            "candidate_id": candidate_id,
            "recruiter_uuid": recruiter_uuid,
            "batch_tag": batch_tag,
            "resume_gcs_url": resume_gcs_url,
            **parsed_data.dict()
        }

        db.collection(CANDIDATE_COLLECTION).document(candidate_id).set(firestore_doc)
        print(f"Saved candidate document: {candidate_id}")

        # 5. Chunk Resume
        print("Step 5: Chunking parsed resume data...")
        chunks = chunk_parsed_resume(candidate_id, parsed_data)

        if not chunks:
            print("Warning: No chunks generated from resume. Skipping vector indexing.")
            return

        chunk_texts = [chunk['text_content'] for chunk in chunks]
        chunk_ids = [chunk['id'] for chunk in chunks]

        print(f"Generated {len(chunks)} chunks")

        # 6. Embed Chunks
        print("Step 6: Generating embeddings for chunks...")
        embeddings = get_embeddings(chunk_texts)

        if len(embeddings) != len(chunk_ids):
            raise Exception(f"Mismatch: {len(embeddings)} embeddings vs {len(chunk_ids)} chunks")

        # 7. Save to Vector Search
        print("Step 7: Preparing datapoints for Vector Search...")

        # Create restrictions as plain dicts
        vector_restrictions = [
            {"namespace": "recruiter_uuid", "allow_list": [recruiter_uuid]},
            {"namespace": "batch_tag", "allow_list": [batch_tag]},
            {"namespace": "candidate_id", "allow_list": [candidate_id]}
        ]

        datapoints = []
        for i, embedding in enumerate(embeddings):
            # Ensure embedding is list of floats
            if not isinstance(embedding, list):
                embedding = list(embedding)

            embedding = [float(v) for v in embedding]

            if i == 0:
                print(f"[Datapoint] Creating datapoint 0:")
                print(f"  - ID: {chunk_ids[i]}")
                print(f"  - Embedding length: {len(embedding)}")
                print(f"  - Embedding type: {type(embedding)}, first value type: {type(embedding[0])}")
                print(f"  - Restrictions: {len(vector_restrictions)} restrictions")

            datapoint = {
                "datapoint_id": chunk_ids[i],
                "feature_vector": embedding,
                "restricts": vector_restrictions
            }
            datapoints.append(datapoint)

        if datapoints:
            print(f"Step 8: Upserting {len(datapoints)} datapoints to Vector Search...")
            upsert_resp = upsert_to_vector_search(datapoints)
            print(f"Upsert response (truncated): {json.dumps(upsert_resp)[:1000]}")

        print(f"✓ Successfully processed and ingested candidate {candidate_id}.")

    except Exception as e:
        print(f"✗ Internal Server Error processing {file_name}: {e}")
        import traceback
        traceback.print_exc()
        return
