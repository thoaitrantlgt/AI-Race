# Implementation Plan for Viettel AI Race Medical Challenge

## 0. Goal

Build a reproducible offline pipeline for Round 1 of Viettel AI Race Medical Challenge.

The task is a structured clinical information extraction problem:

- Input: clinical note text files under `input/`.
- Output: `output.zip`.
- After unzip:

```text
output/
  1.json
  2.json
  ...
  100.json
```

Each JSON should contain extracted medical concepts with:

- raw mention text copied exactly from the note,
- exact character offsets,
- concept type,
- assertion/context label,
- optional ontology candidate IDs.

The solution should prioritize:

1. correct concept type,
2. exact span boundary and raw text,
3. conservative candidate linking,
4. stable assertion prediction,
5. reproducible offline inference.

Do not build an LLM-only pipeline. Build a hybrid pipeline with:

- section parser,
- rule-based span extractor,
- optional NER model hook,
- ontology/gazetteer linker,
- assertion classifier/rules,
- offset-safe postprocessing,
- output validation.

---

## 1. Expected Repository Structure

Create this structure:

```text
viettel-med-ie/
  README.md
  requirements.txt
  run_inference.sh

  configs/
    default.yaml

  data/
    input/
    terminology/
      rxnorm_2026/
      icd10_vi/
      snomed/
      custom_aliases/
    processed/

  output/

  src/
    main.py

    io/
      read_input.py
      write_output.py
      schema.py
      validate_output.py

    preprocess/
      text_normalizer.py
      offset_mapper.py
      section_parser.py
      sentence_splitter.py

    extraction/
      base.py
      rule_extractor.py
      medication_extractor.py
      diagnosis_symptom_extractor.py
      model_extractor.py
      span_merger.py

    linking/
      terminology_store.py
      rxnorm_loader.py
      snomed_loader.py
      alias_loader.py
      candidate_generator.py
      candidate_ranker.py
      linker.py

    assertion/
      cue_lexicons.py
      assertion_rules.py
      assertion_classifier.py

    postprocess/
      type_resolver.py
      overlap_resolver.py
      candidate_filter.py
      offset_validator.py

    eval/
      local_eval.py
      metrics.py

    utils/
      logging.py
      unicode.py
      json_utils.py

  tests/
    test_offsets.py
    test_section_parser.py
    test_rule_extractor.py
    test_linker.py
    test_json_schema.py
```

The first working version must be a strong rule-based baseline. Model-based components should be optional and should not break inference if model weights are absent.

---

## 2. Global Design Principles

### 2.1 Offset safety is mandatory

Never modify the raw input text directly when producing final spans.

Maintain two views:

```text
raw_text        -> used for offsets and final mention text
normalized_text -> used for matching, searching, candidate generation
```

Every span must be validated with:

```python
raw_text[start:end] == predicted_text
```

If validation fails, either fix the boundary or drop the span.

### 2.2 Conservative prediction

Because wrong concept type can be heavily penalized, prefer high precision over high recall.

Candidate linking should also be conservative:

- high-confidence mention: output top-1 candidate,
- ambiguous mention: output top-2 only if close and useful,
- low-confidence mention: output empty candidate list if schema allows.

Avoid outputting long candidate lists.

### 2.3 Section-aware extraction

Clinical notes contain headings such as:

```text
Lý do nhập viện
Bệnh sử hiện tại
Tiền sử bệnh
Tiền sử dùng thuốc
Triệu chứng hiện tại
Đánh giá tại bệnh viện
Chẩn đoán
Điều trị
Thuốc
```

Use these sections to improve:

- span type,
- assertion label,
- medication history detection,
- diagnosis/current condition detection.

### 2.4 Offline reproducibility

The final inference must run with one command:

```bash
bash run_inference.sh data/input output.zip
```

No external API calls during inference.

### 2.5 Terminology version requirements

Use fixed offline terminology snapshots for reproducibility:

- **RxNorm:** use a **2026 release snapshot**. Store it under `data/terminology/rxnorm_2026/`. The loader should record the release folder/date in logs and metadata.
- **ICD-10:** use the **Vietnamese ICD-10 version** for diagnosis/problem linking. Store it under `data/terminology/icd10_vi/`.
- Keep terminology metadata in the output run log, but do not add extra fields to the final JSON unless the competition schema allows them.
- Do not silently mix multiple RxNorm or ICD-10 versions in one run. If multiple terminology folders exist, fail with a clear warning unless the config explicitly selects one.

---

## 3. Config File

Create `configs/default.yaml`:

```yaml
paths:
  terminology_dir: data/terminology
  rxnorm_dir: data/terminology/rxnorm_2026
  icd10_vi_dir: data/terminology/icd10_vi
  snomed_dir: data/terminology/snomed
  alias_dir: data/terminology/custom_aliases

pipeline:
  use_rule_extractor: true
  use_model_extractor: false
  use_sapbert_ranker: false
  use_assertion_classifier: false

extraction:
  min_span_length: 2
  max_span_length: 120
  allow_overlapping_spans: false

linking:
  max_candidates: 2
  high_confidence_threshold: 0.88
  medium_confidence_threshold: 0.72
  enable_rxnorm: true
  rxnorm_version: "2026"
  enable_icd10_vi: true
  icd10_language: "vi"
  enable_snomed: false

assertion:
  default_label: "present"
  historical_sections:
    - "tiền sử"
    - "tiền sử bệnh"
    - "tiền sử dùng thuốc"
    - "bệnh sử"
  negation_cues:
    - "không"
    - "không ghi nhận"
    - "không có"
    - "chưa ghi nhận"
    - "phủ nhận"
  speculation_cues:
    - "nghi ngờ"
    - "theo dõi"
    - "khả năng"
    - "có thể"
  family_cues:
    - "gia đình"
    - "bố"
    - "mẹ"
    - "anh"
    - "chị"
    - "em"

output:
  output_dir: output
  zip_name: output.zip
  ensure_ascii: false
  indent: 2
```

---

## 4. Input Reader

Implement `src/io/read_input.py`.

Requirements:

- Read all `.txt` files in input directory.
- Sort files numerically by filename stem.
- Preserve UTF-8.
- Return records:

```python
@dataclass
class InputRecord:
    record_id: str
    filename: str
    raw_text: str
```

Example:

```python
records = read_input_dir("data/input")
```

Expected output:

```python
[
  InputRecord(record_id="1", filename="1.txt", raw_text="..."),
  ...
]
```

---

## 5. Output Schema

Implement `src/io/schema.py`.

Create internal dataclasses:

```python
@dataclass
class Candidate:
    id: str
    score: float | None = None
    source: str | None = None

@dataclass
class SpanPrediction:
    text: str
    start: int
    end: int
    concept_type: str
    assertion: str
    candidates: list[Candidate]
    confidence: float
    source: str
```

Important:

The final JSON format must follow the competition sample exactly. Therefore:

1. Inspect sample submission if available.
2. Create a conversion function:

```python
def prediction_to_submission_json(pred: SpanPrediction) -> dict:
    ...
```

3. Keep internal schema stable, but make final JSON configurable.

If exact field names are unknown, use this internal-to-output mapping and adjust later:

```json
{
  "text": "...",
  "start": 10,
  "end": 20,
  "type": "drug",
  "assertion": "isHistorical",
  "candidates": ["Rx308135"]
}
```

---

## 6. Text Normalization and Offset Mapping

Implement `src/preprocess/text_normalizer.py`.

Create normalization only for matching:

```python
def normalize_for_matching(text: str) -> str:
    """
    Lowercase.
    Normalize Unicode.
    Normalize repeated whitespace.
    Remove unnecessary punctuation only in matching view.
    Do not use this text for final offsets.
    """
```

Implement `src/preprocess/offset_mapper.py`.

For version 1, avoid complex offset mapping by doing extraction on raw text whenever possible.

Rules:

- Regex extraction runs on raw text.
- Matching may use normalized windows.
- Final span must always come from raw text slice.

Create utility:

```python
def validate_span(raw_text: str, start: int, end: int, text: str) -> bool:
    return raw_text[start:end] == text
```

Also create:

```python
def trim_span_to_valid_boundary(raw_text, start, end):
    """
    Trim leading/trailing whitespace and punctuation.
    Return corrected start, end, raw_text[start:end].
    """
```

Do not trim dosage/frequency from medication mentions unless obviously outside the span.

---

## 7. Section Parser

Implement `src/preprocess/section_parser.py`.

Goal: detect sections and assign every character offset to a section label.

Create heading lexicon:

```python
SECTION_PATTERNS = {
    "admission_reason": [
        r"lý do nhập viện",
        r"lí do nhập viện"
    ],
    "current_illness": [
        r"bệnh sử hiện tại",
        r"triệu chứng hiện tại",
        r"diễn tiến bệnh"
    ],
    "past_history": [
        r"tiền sử",
        r"tiền sử bệnh",
        r"tiền căn"
    ],
    "medication_history": [
        r"tiền sử dùng thuốc",
        r"thuốc đang dùng",
        r"toa thuốc",
        r"medications?"
    ],
    "diagnosis": [
        r"chẩn đoán",
        r"đánh giá",
        r"assessment",
        r"diagnosis"
    ],
    "treatment": [
        r"điều trị",
        r"plan",
        r"kế hoạch"
    ],
    "family_history": [
        r"tiền sử gia đình",
        r"gia đình"
    ]
}
```

Output:

```python
@dataclass
class SectionSpan:
    name: str
    start: int
    end: int
    heading_text: str
```

Function:

```python
def parse_sections(raw_text: str) -> list[SectionSpan]:
    ...
```

Rules:

- Detect headings at line starts.
- Support formats:
  - `Tiền sử bệnh:`
  - `# Tiền sử`
  - `- Tiền sử:`
  - uppercase headings
- Section ends at next heading.
- If no heading detected, assign whole note to `"unknown"`.

Create helper:

```python
def get_section_for_offset(sections, char_start, char_end) -> str:
    ...
```

---

## 8. Rule-Based Span Extraction

Implement base interface in `src/extraction/base.py`:

```python
class BaseExtractor:
    def extract(self, record: InputRecord, sections: list[SectionSpan]) -> list[SpanPrediction]:
        raise NotImplementedError
```

---

## 9. Medication Extractor

Implement `src/extraction/medication_extractor.py`.

Medication extraction should combine:

1. known drug dictionary from RxNorm/custom aliases,
2. regex for dosage and route,
3. list-style medication sections,
4. English generic names.

### 9.1 Medication patterns

Detect medication mentions like:

```text
metoprolol 25mg po bid
doxycycline
atenolol
levofloxacin
tylenol
amlodipine 10mg
insulin
aspirin
```

Regex pieces:

```python
DRUG_NAME = r"[A-Za-z][A-Za-z0-9\-]+(?:\s+[A-Za-z][A-Za-z0-9\-]+){0,3}"
STRENGTH = r"\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|ml|iu|units?|%)"
ROUTE = r"(?:po|iv|im|sc|uống|tiêm|truyền|ngậm)"
FREQ = r"(?:qd|bid|tid|qid|qhs|prn|mỗi ngày|ngày\s*\d+\s*lần|x\s*\d+)"
```

Useful combined patterns:

```python
r"{DRUG_NAME}\s+{STRENGTH}(?:\s+{ROUTE})?(?:\s+{FREQ})?"
r"{DRUG_NAME}\s+(?:{ROUTE})\s+(?:{FREQ})"
```

### 9.2 Dictionary matching

Create a drug alias list:

```text
aspirin
amlodipine
metoprolol
atenolol
doxycycline
levofloxacin
tylenol
paracetamol
acetaminophen
insulin
warfarin
heparin
ceftriaxone
omeprazole
atorvastatin
simvastatin
furosemide
losartan
metformin
```

Later load aliases from RxNorm and custom files.

### 9.3 Medication span boundaries

When a medication name is followed by dosage/frequency, include dosage/frequency if the sample labels do so. If labels only annotate the drug name, add a config flag:

```yaml
extraction:
  medication_include_dose: true
```

Default should be `true` because WER rewards exact surface mention and medication strings may include strength/route.

### 9.4 Medication type

Set:

```python
concept_type = "drug"
```

or whatever exact label name appears in sample submission.

---

## 10. Diagnosis and Symptom Extractor

Implement `src/extraction/diagnosis_symptom_extractor.py`.

This should be dictionary + section + cue based.

### 10.1 Vietnamese symptom lexicon

Start with a local lexicon:

```text
sốt
ho
đau ngực
khó thở
đau bụng
buồn nôn
nôn
chóng mặt
mệt mỏi
phù
đau đầu
tiêu chảy
táo bón
sụt cân
chán ăn
đau lưng
đau họng
khò khè
hồi hộp
ngất
co giật
yếu liệt
tê bì
```

### 10.2 Diagnosis lexicon

Start with:

```text
tăng huyết áp
đái tháo đường
tiểu đường
suy tim
nhồi máu cơ tim
bệnh mạch vành
viêm phổi
hen phế quản
copd
bệnh phổi tắc nghẽn mạn tính
suy thận
bệnh thận mạn
xơ gan
viêm gan
tai biến mạch máu não
đột quỵ
nhiễm trùng
nhiễm khuẩn
rối loạn lipid máu
ung thư
thiếu máu
```

### 10.3 Type decision rules

Use section and phrase patterns:

- In `diagnosis` section: default type = diagnosis/problem.
- In `current_illness` section:
  - acute subjective findings = symptom.
  - named disease = diagnosis.
- In `past_history` section:
  - named disease = diagnosis with historical assertion.
- In medication section:
  - avoid extracting diagnosis unless strongly matched.

Create function:

```python
def infer_problem_type(span_text: str, section_name: str, context_window: str) -> str:
    ...
```

Keep the type set configurable because exact competition labels may differ.

---

## 11. Generic Rule Extractor

Implement `src/extraction/rule_extractor.py`.

This orchestrates:

```python
class RuleExtractor(BaseExtractor):
    def __init__(self, terminology_store, config):
        self.med_extractor = MedicationExtractor(...)
        self.problem_extractor = DiagnosisSymptomExtractor(...)

    def extract(self, record, sections):
        spans = []
        spans += self.med_extractor.extract(record, sections)
        spans += self.problem_extractor.extract(record, sections)
        return spans
```

---

## 12. Optional Model Extractor Hook

Implement `src/extraction/model_extractor.py`.

Do not require model training in version 1.

Create a safe interface:

```python
class ModelExtractor(BaseExtractor):
    def __init__(self, model_path=None):
        self.enabled = model_path is not None and os.path.exists(model_path)

    def extract(self, record, sections):
        if not self.enabled:
            return []
        ...
```

Expected future models:

- PhoBERT token classifier for Vietnamese NER,
- XLM-R token classifier for mixed-language notes.

For now, return empty list if no model exists.

---

## 13. Span Merger

Implement `src/extraction/span_merger.py`.

Goal: merge spans from rule extractor and optional model extractor.

Rules:

1. Validate offset.
2. Drop empty/too-short spans.
3. Normalize duplicate spans:
   - same start/end/type: keep highest confidence.
4. Resolve overlap:
   - if same type and overlapping, keep longer span if confidence close.
   - if medication overlaps with diagnosis, prefer medication if dosage/route present.
   - if type conflict and confidence low, drop lower confidence.
5. Do not allow nested duplicate predictions.

Function:

```python
def merge_spans(spans: list[SpanPrediction], raw_text: str) -> list[SpanPrediction]:
    ...
```

Overlap policy:

```python
def overlap_ratio(a_start, a_end, b_start, b_end):
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union
```

If overlap ratio > 0.5, resolve.

---

## 14. Terminology Store

Implement `src/linking/terminology_store.py`.

Create object:

```python
@dataclass
class TermEntry:
    concept_id: str
    name: str
    normalized_name: str
    source: str
    semantic_type: str | None
    aliases: list[str]
    version: str | None = None
    language: str | None = None
```

Store indexes:

```python
exact_index: dict[str, list[TermEntry]]
token_index: dict[str, list[TermEntry]]
alias_index: dict[str, list[TermEntry]]
```

Methods:

```python
class TerminologyStore:
    def add_entry(self, entry: TermEntry): ...
    def lookup_exact(self, mention: str) -> list[TermEntry]: ...
    def lookup_fuzzy(self, mention: str, limit: int = 20) -> list[TermEntry]: ...
    def lookup_by_tokens(self, mention: str, limit: int = 20) -> list[TermEntry]: ...
```

Use Python standard libraries first. Optional dependency:

```text
rapidfuzz
```

for fuzzy matching.

---

## 15. RxNorm Loader

Implement `src/linking/rxnorm_loader.py`.

RxNorm release files are usually RRF files. Support loading from:

```text
RXNCONSO.RRF
```

Important fields in RXNCONSO.RRF:

- RXCUI,
- LAT,
- TS,
- LUI,
- STT,
- SUI,
- ISPREF,
- RXAUI,
- SAUI,
- SCUI,
- SDUI,
- SAB,
- TTY,
- CODE,
- STR,
- SRL,
- SUPPRESS,
- CVF.

For version 1, parse line by splitting `|`.

Extract:

```python
concept_id = "Rx" + RXCUI
name = STR
source = "RxNorm"
semantic_type = TTY
```

Only keep English terms:

```python
LAT == "ENG"
```

Prefer common TTY values:

```text
IN   ingredient
PIN  precise ingredient
BN   brand name
SCD  semantic clinical drug
SBD  semantic branded drug
GPCK generic pack
BPCK branded pack
```

Add every term to terminology store.

Version requirement:

- The expected RxNorm source folder is `data/terminology/rxnorm_2026/`.
- The loader must treat this as the selected **RxNorm 2026 snapshot**.
- Add `version="2026"` and `language="en"` to every RxNorm `TermEntry`.
- Log the exact folder name and, if available, release date/file metadata.
- If the configured RxNorm folder name or metadata does not indicate 2026, print a clear warning.

If RxNorm files are absent, fallback to custom aliases.

---


## 16. ICD-10 Vietnamese Loader

Implement `src/linking/icd10_vi_loader.py`.

This loader is used for diagnosis/problem candidate linking. It should use the **Vietnamese ICD-10 version**, stored locally under:

```text
data/terminology/icd10_vi/
```

Support one or more simple offline formats because the available ICD-10 Vietnamese source may come as CSV, TSV, Excel-exported TSV, or normalized text:

```text
code<TAB>vietnamese_name<TAB>english_name_optional<TAB>chapter_optional
```

Minimum required fields:

- ICD-10 code, for example `I10`, `E11`, `J18`;
- Vietnamese disease name;
- optional English synonym;
- optional chapter/category.

Create entries like:

```python
concept_id = "ICD10:" + code
name = vietnamese_name
source = "ICD10_VI"
semantic_type = "diagnosis"
version = "ICD-10 Vietnamese"
language = "vi"
```

Function:

```python
def load_icd10_vi_to_store(path, store):
    ...
```

Implementation requirements:

- Normalize Vietnamese accents only in the matching index, not in final spans.
- Preserve the original Vietnamese disease name in `name`.
- Add common Vietnamese aliases when obvious, for example `đái tháo đường` and `tiểu đường`.
- Prefer ICD-10 Vietnamese candidates for diagnosis/problem spans.
- Do not use ICD-10 for medication spans.
- If ICD-10 Vietnamese files are absent, do not crash; continue with custom aliases and empty diagnosis candidates.

---

## 17. SNOMED Loader

Implement `src/linking/snomed_loader.py`.

Make this optional.

Expected input can be either:

1. official SNOMED description file, or
2. custom TSV:

```text
concept_id<TAB>term<TAB>semantic_type
```

Function:

```python
def load_snomed_to_store(path, store):
    ...
```

If files absent, do not crash.

---

## 18. Custom Alias Loader

Implement `src/linking/alias_loader.py`.

Support simple TSV files:

```text
alias<TAB>canonical_name<TAB>concept_id<TAB>source<TAB>semantic_type
```

Example:

```text
tiểu đường	đái tháo đường	ICD10:E11	ICD10_VI	diagnosis
đái tháo đường	đái tháo đường	ICD10:E11	ICD10_VI	diagnosis
tăng huyết áp	tăng huyết áp vô căn	ICD10:I10	ICD10_VI	diagnosis
tylenol	acetaminophen	Rx161	RxNorm	drug
```

If `concept_id` is empty, keep alias for span extraction but no candidate output.

---

## 18. Candidate Generator

Implement `src/linking/candidate_generator.py`.

Input:

```python
mention_text
concept_type
context_window
```

Candidate generation strategy:

1. exact lookup,
2. normalized exact lookup,
3. alias lookup,
4. fuzzy lookup,
5. token-set lookup.

Return:

```python
@dataclass
class CandidateMatch:
    concept_id: str
    name: str
    source: str
    score: float
    match_type: str
```

Scoring heuristic:

```text
exact match: 1.00
alias exact: 0.95
normalized exact: 0.92
fuzzy >= 95: 0.88
fuzzy >= 90: 0.82
token overlap strong: 0.70
otherwise drop
```

For medications:

- Prefer RxNorm candidates.
- If mention includes dosage/form and RxNorm has matching clinical drug, prefer SCD/SBD.
- If mention is only ingredient, prefer IN/PIN.
- Do not hallucinate dosage-specific candidate when dosage is missing.

For diagnosis/symptom:

- Prefer ICD-10 Vietnamese candidates first when concept type is diagnosis/problem.
- Use SNOMED/custom candidates only as optional fallback if configured.
- If no ontology is available, return an empty candidate list.

---

## 19. Candidate Ranker

Implement `src/linking/candidate_ranker.py`.

Version 1: heuristic ranker.

Sort by:

1. candidate score,
2. source priority,
3. semantic type compatibility,
4. shorter normalized name distance,
5. exact dosage/form compatibility.

Function:

```python
def rank_candidates(mention, concept_type, candidates, context_window):
    ...
```

Optional future version:

- SapBERT embedding reranker.
- PubMedBERT reranker.
- Small local LLM reranker under 9B.

But version 1 must work without them.

---

## 20. Candidate Filter

Implement `src/postprocess/candidate_filter.py`.

Because Jaccard punishes overprediction, filter candidates aggressively.

Policy:

```python
def filter_candidates(ranked_candidates, config):
    if not ranked_candidates:
        return []

    top = ranked_candidates[0]

    if top.score >= high_confidence_threshold:
        return [top]

    if top.score >= medium_confidence_threshold:
        if len(ranked_candidates) >= 2:
            second = ranked_candidates[1]
            if abs(top.score - second.score) <= 0.05:
                return [top, second]
        return [top]

    return []
```

Default max candidates = 2.

Never return more than config value.

---

## 21. Linker

Implement `src/linking/linker.py`.

Function:

```python
def link_spans(spans, raw_text, sections, terminology_store, config):
    for span in spans:
        context = get_context_window(raw_text, span.start, span.end, window_chars=120)
        candidates = generate_candidates(...)
        ranked = rank_candidates(...)
        filtered = filter_candidates(...)
        span.candidates = filtered
    return spans
```

---

## 22. Assertion Prediction

Implement `src/assertion/assertion_rules.py`.

Assertion labels must match competition schema. If exact labels are unknown, create mapping in config:

```yaml
assertion_labels:
  present: "isPresent"
  historical: "isHistorical"
  negated: "isNegated"
  possible: "isPossible"
  family: "isFamily"
```

Adjust according to sample JSON.

### 22.1 Rule priority

Use this order:

1. family history,
2. negation,
3. speculation,
4. historical,
5. present/default.

### 22.2 Family history

If section is `family_history` or context contains family cue:

```text
gia đình
bố
mẹ
cha
mẹ
anh trai
chị gái
em
```

then assertion = family, if label exists.

### 22.3 Negation

If left context within 60 chars contains:

```text
không
không có
không ghi nhận
chưa ghi nhận
phủ nhận
âm tính
```

then assertion = negated.

Avoid false negation if cue is far from span or belongs to previous sentence.

### 22.4 Speculation

If context contains:

```text
nghi ngờ
theo dõi
khả năng
có thể
chưa loại trừ
rule out
r/o
```

then assertion = possible.

### 22.5 Historical

If section is:

```text
past_history
medication_history
```

or nearby context contains:

```text
tiền sử
đã từng
trước đây
trước khi nhập viện
đang dùng tại nhà
```

then assertion = historical.

### 22.6 Present

Default:

```text
present
```

### 22.7 Implementation

Function:

```python
def predict_assertion(span, raw_text, section_name, config):
    context_left, context_right = get_context(...)
    sentence = get_sentence_containing_span(...)
    ...
```

---

## 23. Optional Assertion Classifier

Implement `src/assertion/assertion_classifier.py`.

Do not require it in baseline.

Create stub:

```python
class AssertionClassifier:
    def __init__(self, model_path=None):
        self.enabled = model_path is not None and os.path.exists(model_path)

    def predict(self, span, raw_text, section_name):
        if not self.enabled:
            return None
```

Rules remain default.

---

## 24. Postprocessing

Implement these modules:

### 24.1 Type Resolver

`src/postprocess/type_resolver.py`

Goal: prevent obvious type mistakes.

Rules:

- If span has dosage/route/frequency, force type = drug.
- If span appears in medication section and matches drug alias, force type = drug.
- If span is disease phrase in diagnosis section, force diagnosis/problem.
- If low confidence and conflicting type, drop span.

### 24.2 Overlap Resolver

`src/postprocess/overlap_resolver.py`

Rules:

- Remove duplicate spans.
- Resolve nested spans.
- Prefer more informative medication span if dosage is part of mention.
- Prefer exact dictionary match over fuzzy long noisy span.
- Do not output two spans with same start/end but different type.

### 24.3 Offset Validator

`src/postprocess/offset_validator.py`

Every final prediction must pass:

```python
assert raw_text[start:end] == text
```

If not:

1. try to repair by searching exact text near predicted offset,
2. if exactly one nearby match exists, fix offset,
3. otherwise drop span.

Nearby window:

```python
raw_text[max(0, start-20): min(len(raw_text), end+20)]
```

---

## 25. Output Writer

Implement `src/io/write_output.py`.

Function:

```python
def write_record_json(record_id, predictions, output_dir):
    ...
```

Rules:

- Create output directory if absent.
- Write one JSON per record.
- UTF-8.
- `ensure_ascii=False`.
- Use exact competition schema.
- Sort predictions by `start`, then `end`.

Create zip:

```python
def create_output_zip(output_dir, zip_path):
    ...
```

Zip structure must be:

```text
output/
  1.json
  2.json
  ...
```

not:

```text
output.zip/output/...
```

and not:

```text
1.json
2.json
```

unless competition sample says otherwise.

---

## 26. Output Validator

Implement `src/io/validate_output.py`.

Check:

- number of JSON files equals number of input records,
- each filename is `{record_id}.json`,
- JSON is valid UTF-8,
- each span has required fields,
- start/end are integers,
- `0 <= start < end <= len(raw_text)`,
- `raw_text[start:end] == text`,
- candidates list length <= max candidates,
- no duplicate same start/end/type,
- no invalid assertion labels,
- no invalid concept types.

Run validation before zipping.

If validation fails, print exact record and span.

---

## 27. Local Evaluation

Implement `src/eval/metrics.py`.

If ground truth labels are available, implement approximate local metrics:

### 27.1 Span exact match

Compare:

```text
record_id, start, end, concept_type
```

### 27.2 Text WER

Implement word-level edit distance:

```python
def wer(ref_text, hyp_text):
    ...
```

### 27.3 Assertion Jaccard

If assertions are sets:

```python
intersection / union
```

If single label:

```python
1 if same else 0
```

### 27.4 Candidate Jaccard

```python
len(set(pred_candidates) & set(gold_candidates)) / len(set(pred_candidates) | set(gold_candidates))
```

### 27.5 Error report

Create error report:

```text
false_positive_spans.tsv
false_negative_spans.tsv
type_confusion.tsv
bad_offsets.tsv
candidate_errors.tsv
assertion_errors.tsv
```

This will guide improvement.

---

## 28. Main Pipeline

Implement `src/main.py`.

CLI:

```bash
python -m src.main \
  --input_dir data/input \
  --output_zip output.zip \
  --config configs/default.yaml
```

Main flow:

```python
def run_pipeline(input_dir, output_zip, config_path):
    config = load_config(config_path)

    records = read_input_dir(input_dir)

    terminology_store = TerminologyStore()
    load_rxnorm_if_available(config.paths.rxnorm_dir, terminology_store)
    load_icd10_vi_if_available(config.paths.icd10_vi_dir, terminology_store)
    load_snomed_if_available(config.paths.snomed_dir, terminology_store)
    load_custom_aliases(config.paths.alias_dir, terminology_store)

    rule_extractor = RuleExtractor(terminology_store, config)
    model_extractor = ModelExtractor(config.model.path) if config.pipeline.use_model_extractor else None

    all_outputs = {}

    for record in records:
        sections = parse_sections(record.raw_text)

        spans = []
        if config.pipeline.use_rule_extractor:
            spans += rule_extractor.extract(record, sections)

        if model_extractor is not None:
            spans += model_extractor.extract(record, sections)

        spans = merge_spans(spans, record.raw_text)
        spans = resolve_types(spans, record.raw_text, sections, config)
        spans = resolve_overlaps(spans, record.raw_text, config)
        spans = link_spans(spans, record.raw_text, sections, terminology_store, config)
        spans = predict_assertions_for_spans(spans, record.raw_text, sections, config)
        spans = validate_and_repair_offsets(spans, record.raw_text, config)
        spans = final_candidate_filter(spans, config)

        all_outputs[record.record_id] = spans

    write_all_json(all_outputs, output_dir=config.output.output_dir)
    validate_output(...)
    create_output_zip(config.output.output_dir, output_zip)
```

---

## 29. Shell Script

Create `run_inference.sh`:

```bash
#!/usr/bin/env bash
set -e

INPUT_DIR=${1:-data/input}
OUTPUT_ZIP=${2:-output.zip}
CONFIG=${3:-configs/default.yaml}

python -m src.main \
  --input_dir "$INPUT_DIR" \
  --output_zip "$OUTPUT_ZIP" \
  --config "$CONFIG"
```

Make executable:

```bash
chmod +x run_inference.sh
```

---

## 30. Requirements

Create `requirements.txt`:

```text
pyyaml
regex
rapidfuzz
tqdm
numpy
pandas
```

Optional later:

```text
torch
transformers
sentence-transformers
scikit-learn
```

Do not require optional heavy packages for baseline inference.

---

## 31. Testing Plan

### 31.1 Offset tests

`tests/test_offsets.py`

Test:

```python
raw = "Bệnh nhân dùng amlodipine 10mg mỗi ngày."
span = "amlodipine 10mg"
start = raw.index(span)
end = start + len(span)
assert raw[start:end] == span
```

Also test trimming whitespace.

### 31.2 Section parser tests

Input:

```text
Lý do nhập viện:
Đau ngực

Tiền sử bệnh:
Tăng huyết áp
```

Expected:

- `admission_reason`,
- `past_history`.

### 31.3 Medication extractor tests

Input:

```text
BN đang dùng metoprolol 25mg po bid và aspirin.
```

Expected:

- `metoprolol 25mg po bid`,
- `aspirin`.

### 31.4 Assertion tests

Input:

```text
Tiền sử bệnh:
Tăng huyết áp.
```

Expected:

- `Tăng huyết áp` assertion = historical.

Input:

```text
Không ghi nhận đau ngực.
```

Expected:

- `đau ngực` assertion = negated.

### 31.5 Linker tests

If alias exists:

```text
amlodipine -> Rx308135
```

Expected candidate contains `Rx308135`.

### 31.6 JSON schema tests

For each output JSON:

- valid JSON,
- correct fields,
- offset valid,
- no more than max candidates.

---

## 32. Development Order

Follow this exact order.

### Phase 1: Working baseline

Implement:

1. input reader,
2. output writer,
3. output validator,
4. section parser,
5. simple medication extractor,
6. simple symptom/diagnosis extractor,
7. span merger,
8. assertion rules,
9. custom alias linker,
10. `run_inference.sh`.

Deliverable:

```bash
bash run_inference.sh data/input output.zip
```

must produce valid output.

### Phase 2: Improve span extraction

Add:

1. better medication regex,
2. Vietnamese symptom lexicon,
3. diagnosis lexicon,
4. overlap resolver,
5. type resolver,
6. offset repair.

Deliverable:

- fewer bad spans,
- fewer type conflicts,
- all offsets valid.

### Phase 3: Add ontology linking

Add:

1. RxNorm 2026 loader,
2. ICD-10 Vietnamese loader,
3. custom alias TSV support,
4. candidate generator,
5. candidate ranker,
6. candidate filter.

Deliverable:

- medication spans get conservative RxNorm 2026 candidate IDs,
- diagnosis/problem spans get ICD-10 Vietnamese candidate IDs when available,
- no long candidate lists.

### Phase 4: Add local evaluation

If gold labels are available, implement:

1. local metrics,
2. error reports,
3. ablation configs.

Deliverable:

```bash
python -m src.eval.local_eval --gold data/gold --pred output
```

### Phase 5: Optional model-based NER

Only after rule baseline is stable:

1. train PhoBERT token classifier,
2. export model,
3. add model extractor,
4. merge model spans with rule spans,
5. compare local eval.

Model extractor must be optional.

### Phase 6: Optional biomedical reranker

Only after candidate generation is stable:

1. add SapBERT-style embedding reranker,
2. cache ontology embeddings,
3. rerank top candidates,
4. keep final candidate list small.

---

## 33. Concrete Baseline Heuristics

Use these default confidence values:

```text
medication regex with dosage: 0.92
medication exact dictionary match: 0.88
medication fuzzy dictionary match: 0.74
diagnosis exact lexicon in diagnosis section: 0.86
diagnosis exact lexicon in past history section: 0.84
symptom exact lexicon in current illness section: 0.82
symptom exact lexicon outside useful section: 0.68
```

Drop spans below:

```text
0.65
```

unless they have exact ontology match.

---

## 34. Candidate Linking Heuristics

### 34.1 Medication

If mention:

```text
amlodipine 10mg
```

candidate generation should search:

```text
amlodipine 10mg
amlodipine
```

Prefer:

1. exact clinical drug with strength,
2. ingredient if no exact clinical drug,
3. brand/alias match.

### 34.2 Brand names

Map known brands to ingredients through alias file:

```text
tylenol -> acetaminophen/paracetamol
```

### 34.3 Vietnamese aliases

Use custom aliases:

```text
tăng huyết áp -> hypertension
đái tháo đường -> diabetes mellitus
tiểu đường -> diabetes mellitus
nhồi máu cơ tim -> myocardial infarction
đột quỵ -> stroke
viêm phổi -> pneumonia
```

Even if no candidate ID is available, aliases help extraction/type.

---

## 35. Error Analysis Checklist

After each run, inspect:

### 35.1 Bad offsets

Any prediction where:

```python
raw_text[start:end] != text
```

must be fixed immediately.

### 35.2 Type confusion

Check examples where:

- drug predicted as diagnosis,
- diagnosis predicted as symptom,
- symptom predicted as diagnosis.

Add section/type rules.

### 35.3 Overprediction

If output has too many low-confidence spans, raise threshold.

### 35.4 Candidate overprediction

If candidate lists are long, reduce max candidates to 1.

### 35.5 Assertion errors

If many current conditions are marked historical, refine section parser.

If negation spreads too far, restrict negation cue to same sentence.

---

## 36. README Requirements

Create `README.md` with:

```text
# Viettel Medical IE Pipeline

## Setup

pip install -r requirements.txt

## Data

Place input files in:

data/input/

Optional terminology:

data/terminology/rxnorm_2026/
data/terminology/icd10_vi/
data/terminology/snomed/
data/terminology/custom_aliases/

## Run inference

bash run_inference.sh data/input output.zip

## Output

Creates:

output/
output.zip

## Reproducibility

No external API calls are used.
All terminology files are loaded locally.
RxNorm must use a 2026 release snapshot under data/terminology/rxnorm_2026/.
Diagnosis/problem linking should use the Vietnamese ICD-10 terminology under data/terminology/icd10_vi/.
```

---

## 37. Acceptance Criteria

The implementation is successful when:

1. `bash run_inference.sh data/input output.zip` runs without error.
2. `output.zip` has the required structure.
3. Each input record has exactly one JSON output.
4. Every output span has valid offsets.
5. No JSON file contains invalid UTF-8.
6. Candidate list length is never greater than configured max.
7. Pipeline works even without model weights.
8. Pipeline works even without RxNorm/ICD-10/SNOMED files, using fallback aliases.
9. RxNorm 2026 and ICD-10 Vietnamese folders are configurable and loaded offline when present.
10. Code is modular enough to later add PhoBERT/SapBERT.
11. README explains how to reproduce.

---

## 38. Important Implementation Notes

Do not normalize text before computing offsets.

Do not use online APIs.

Use RxNorm 2026 for drug linking and Vietnamese ICD-10 for diagnosis/problem linking.

Do not output too many candidates.

Do not depend on a large LLM for core extraction.

Do not crash if ontology files are missing.

Do not hardcode only 100 records; support any number of `.txt` files.

Do not assume labels before checking sample submission. Put label mapping in config.

Do not include debugging fields in final JSON unless competition schema allows them.

---

## 39. First Codex Task

Start by implementing the baseline pipeline only.

Specifically implement these files first:

```text
src/io/read_input.py
src/io/schema.py
src/io/write_output.py
src/io/validate_output.py
src/preprocess/section_parser.py
src/preprocess/text_normalizer.py
src/extraction/medication_extractor.py
src/extraction/diagnosis_symptom_extractor.py
src/extraction/rule_extractor.py
src/extraction/span_merger.py
src/assertion/assertion_rules.py
src/linking/terminology_store.py
src/linking/alias_loader.py
src/linking/icd10_vi_loader.py
src/linking/candidate_generator.py
src/linking/linker.py
src/postprocess/offset_validator.py
src/main.py
run_inference.sh
requirements.txt
README.md
```

The first version does not need PhoBERT, XLM-R, SapBERT, or PubMedBERT.

After the baseline produces valid `output.zip`, then add model-based improvements.
