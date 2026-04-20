# Fullerton Assessment OCR API

FastAPI OCR extraction service for the assessment documents. The API accepts a PDF, JPG, or PNG upload, classifies the document type, and returns structured JSON fields.

## Folder Structure

The folder structure is intentionally designed to mirror production-ready layouts. For a small assessment, this may be overkill, but if scaled, this helps to keep the codebase readable and easy to extend.

```text
.
|-- app/
|   |-- api/
|   |   +-- routes.py              # FastAPI routes and request validation
|   |-- core/
|   |   +-- configs.py             # Application configuration
|   |-- schemas/
|   |   |-- error.py               # Error response schema
|   |   |-- health.py              # Health response schema
|   |   +-- ocr.py                 # OCR request/response and document schemas
|   |-- services/
|   |   +-- ocr.py                 # OCR pipeline and OpenAI integration
|   |-- constants.py               # Supported file types and model prompts
|   +-- main.py                    # FastAPI app creation
|-- docs/
|   +-- fullerton_ocr.postman_collection.json
|-- tests/
|   |-- test_ocr.py                # API endpoint tests
|   |-- test_ocr_schemas.py        # Schema validation tests
|   +-- test_ocr_service.py        # OCR service orchestration tests
|-- pyproject.toml
+-- README.md
```

## Setup With uv
If you don't have `uv` installed, you can install it with pip:

```bash
pip install uv
```
If not, you can install uv [here](https://docs.astral.sh/uv/).

Install dependencies from `pyproject.toml`:

```bash
uv sync
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

For Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

## Run The API

Start the server:

```bash
uv run uvicorn app.main:app --reload
```

Open the Swagger docs:

```text
http://127.0.0.1:8000/docs
```

Swagger supports file uploads directly. Open `POST /ocr`, click **Try it out**, choose a PDF/JPG/PNG file from your machine, then click **Execute**.

## Endpoints

`GET /health`

Returns a simple health response. This endpoint serves to check health and connectivity.

`POST /ocr`

Accepts multipart form data with one `file` field. Supported uploads:

- `application/pdf`
- `image/jpeg`
- `image/png`

Supported document types: (as of now)

- `referral_letter`
- `medical_certificate`
- `receipt`

## How To Extend To New Document Types

The current implementation only supports three document types, but the pipeline is intentionally split into two stages:

1. Document classification
2. Field extraction

For three document types, this may look more structured than strictly necessary. The reason for the split is scalability. In a production setting, the service should avoid spending expensive extraction compute on documents that are unsupported in the first place, especially considering most likely that this will become some form of a batched pipeline that uses a message queue and asynchronous workers.

### 1. Add A Low-Cost Classification Stage

The first stage should be a relatively low-cost document classifier. Based on the intended direction, this could be a supervised document-image classification model such as a Document Image Transformer-style model or a similar architecture trained on the supported document types from labelled data.

The classifier receives the uploaded document, either PDF pages or image files, and returns:

- predicted document type
- confidence score

The service then applies a confidence threshold/gating mechanism:

- If the confidence is high enough, the pipeline continues to extraction.
- If the confidence is too low, or the predicted class is unsupported, the service returns `unsupported_document_type`.

This prevents the system from wasting extraction-model tokens, latency, and compute on documents that are outside the supported scope.

The tradeoff is that this classifier needs supervised learning. It must be trained with representative examples of each supported document type so it can learn both content cues and document-layout structure.

### 2. Use A Heavier Extraction Model Only After Classification

The second stage is field extraction. This can use a more capable and more expensive vision-language model, such as the current OpenAI GPT model.

Other VLMs can also be considered, especially open-source or self-hosted models, when privacy, PDPA, or deployment constraints are important. Examples include the recent Genma4 or other document-capable VLMs that can run in a controlled environment. Hosting can be done via a separate microservice. The key is that the extraction model should be reserved for documents that have already been classified as supported types.

The key design goal is to use structured outputs for extraction. Instead of writing custom business rules for each document type, each supported document type is represented as a Pydantic schema. The model is asked to return data that conforms to that schema.

This gives a few benefits:

- Output fields are constrained by the schema.
- The extraction layer is reusable across document types.
- Adding a new document type mostly means adding a new schema, not writing a new parser.
- The model output can be validated and normalized consistently.
- A confidence threshold can be applied per extracted value.

The confidence threshold acts as a gating mechanism:

- High-confidence values are accepted.
- Low-confidence values are returned as `"unsure"`, which for now, will be returned as `"null"` as well, but can be adjusted based on use case. 
- Missing or unsupported fields remain `null`.

This keeps the extraction process schema-driven rather than rule-driven.

### Adding A New Document Type

To add a new document type:

1. Add the new document type literal to the supported document type list.
2. Add a description for the new document type so the classifier has clear semantic guidance.
3. Create a new Pydantic schema for the fields that should be extracted.
4. Register the schema in the OCR service document schema mapping.
5. Add classifier training data or examples for the new document type.
6. Add tests for classification, extraction schema generation, validation, and response shape.

At scale, the classifier and extractor can evolve independently:

- The classifier optimizes routing and rejection of unsupported documents.
- The extractor optimizes accurate field-level parsing for supported documents.

This abstraction is especially useful when balancing compute cost, latency, model accuracy, and privacy requirements, which are key concerns when scaling this OCR service in a production environment.

## Curl Examples

Health check:

```bash
curl "http://127.0.0.1:8000/health"
```

Upload a PDF:

```bash
curl -X POST "http://127.0.0.1:8000/ocr" \
  -H "accept: application/json" \
  -F "file=@medical_certificate.pdf;type=application/pdf"
```

Upload a PNG:

```bash
curl -X POST "http://127.0.0.1:8000/ocr" \
  -H "accept: application/json" \
  -F "file=@sample.png;type=image/png"
```

Upload a JPG:

```bash
curl -X POST "http://127.0.0.1:8000/ocr" \
  -H "accept: application/json" \
  -F "file=@sample.jpg;type=image/jpeg"
```

## Postman

Import the collection:

```text
docs/fullerton_ocr.postman_collection.json
```

The collection includes:

- `GET /health`
- `POST /ocr`

For the OCR request, select the `file` form-data row in Postman and choose a local PDF/JPG/PNG before sending.

The collection uses this variable:

```text
baseUrl = http://127.0.0.1:8000
```

## Tests

```bash
uv run pytest
```

The test suite covers the API layer, schema validation, and OCR service orchestration.

`tests/test_ocr.py`

- Verifies `GET /health` returns the expected health response.
- Verifies `POST /ocr` accepts multipart file uploads and returns the expected response shape.
- Covers upload validation for missing files, empty files, invalid MIME types, and unsupported extensions.
- Covers API error mapping for low-confidence classification and unexpected service failures.

`tests/test_ocr_schemas.py`

- Verifies medical certificate date fields accept the required `DD/MM/YYYY` format.
- Verifies equivalent date inputs such as ISO dates, `DD-MM-YYYY`, `date`, and `datetime` values are normalized to `DD/MM/YYYY`.
- Verifies invalid dates are rejected.
- Verifies `provider_name` values containing `Fullerton Health` are normalized to `null`.
- Verifies amount fields round decimals to the nearest integer instead of stripping decimal points.
- Verifies referral letter signature presence defaults to `false` when omitted.

`tests/test_ocr_service.py`

- Verifies the OCR service uses the two-step OpenAI flow: classification first, extraction second.
- Verifies image payloads are sent to OpenAI as base64 data URLs.
- Verifies Pydantic structured output models are passed to the OpenAI client.
- Verifies confidence-threshold post-processing converts low-confidence extracted fields to `"unsure"`.
- Verifies PDF uploads are rendered into page images before model calls.
- Verifies low-confidence document classification is rejected.
