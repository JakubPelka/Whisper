# MobileNotes

## Status

**Idea / future project** — planned after VildaLeder / PoznajKraj.

This document collects the product concept, architecture decisions, privacy assumptions, risks and implementation roadmap for turning the existing `Whisper` repository into a private iOS-first meeting/service-note application and, only later, a commercial product.

The immediate goal is **not** to build accounts, subscriptions or a public SaaS backend. The first goal is much simpler:

> Record a meeting directly on an iPhone, keep the audio on the phone, transcribe locally, send only text for the existing Luna → Terra note pipeline, and receive a grounded service note back on the phone.

---

# 1. Product concept

The useful product is not merely another "AI meeting recorder".

The stronger concept is:

```text
record
  ↓
local transcription
  ↓
optional organizational context
  ↓
grounded professional note
```

The application should be able to understand the user's preferred note structure and supporting context without confusing those materials with evidence from the actual meeting.

Possible optional inputs:

- **Mall / template** — controls structure and style of the output.
- **Agenda** — helps organize topics and use correct headings.
- **Reference material** — supplies terminology, project context, earlier decisions, names or background information.
- **Transcript** — remains the primary evidence of what was actually said during the meeting.

The existing repository already contains an important part of this concept: transcription output with timestamped segments, a structured Luna draft, a separate Terra verification pass, evidence references to transcript segments and API usage tracking.

---

# 2. Core architecture

Preferred architecture for the mobile product:

```text
iPhone
  │
  ├── record audio locally
  │
  ├── transcribe locally
  │
  ├── optional template / agenda / reference material
  │
  └── user presses Generate
          │
          │ text only
          ▼
     private note API
          │
          ├── Luna: draft
          ├── Terra: verification
          │
          ▼
     grounded note
          │
          ▼
       iPhone
```

Default rule:

> **Audio should never need to leave the phone.**

The backend should receive only the transcript text and, if explicitly selected by the user, the text extracted from template, agenda or reference materials.

A server-side Whisper / KB-Whisper path can remain as an optional fallback if Apple's local transcription quality is not sufficient.

---

# 3. Why local transcription first

For the first iOS version, prefer Apple's on-device speech transcription APIs over running a GPU transcription service.

Expected advantages:

- no audio upload,
- lower infrastructure cost,
- simpler backend,
- better privacy story,
- no GPU queue,
- no need to operate large Whisper models for every user,
- transcription can work before the user decides whether anything should be sent to the API.

However, Swedish transcription quality must be tested against the current local KB-Whisper workflow.

The first meaningful benchmark should therefore use real recordings, not synthetic examples.

Suggested test set:

- 10–20 real Swedish recordings,
- clean audio,
- conference-table audio,
- several speakers,
- fast conversational Swedish,
- names and abbreviations,
- GIS / municipal / technical terminology,
- poor acoustics.

Compare:

```text
Apple on-device transcription
vs
KB-Whisper large
```

The practical question is not simply which system has the lowest formal word error rate.

The useful question is:

> Is Apple's transcript accurate enough for Luna + Terra to produce a correct professional note?

If yes, the production architecture becomes much simpler.

If no, keep an optional future path such as:

```text
Local transcription
Enhanced transcription
```

where enhanced transcription may use a server-side model.

---

# 4. Source semantics and grounding

This distinction is critical.

| Source | Allowed purpose |
|---|---|
| Transcript | Evidence of what was actually said |
| Template / Mall | Structure and style |
| Agenda | Topic organization and headings |
| Reference material | Context, terminology and background |

A template, agenda or reference document must **not** silently become evidence that something happened during the meeting.

Example:

If the agenda contains:

```text
Decision on detailed plan
```

but no decision is actually made during the recorded meeting, the generated note must not state that a decision was made.

Recommended prompt/input separation:

```text
<transcript>
...
</transcript>

<template>
...
</template>

<agenda>
...
</agenda>

<reference_material>
...
</reference_material>
```

Terra should ultimately verify not only whether a statement is supported, but also **what type of source supports it**.

Possible future modes:

### Strict note

Facts must be supported by the transcript.

### Context enhanced

Reference material may supply additional context, but those facts must remain distinguishable from information actually stated during the meeting.

---

# 5. Private alpha philosophy

Before building a commercial product, build a version usable by one person on one iPhone.

For the private alpha deliberately avoid:

- external user accounts,
- organizations,
- teams,
- StoreKit,
- subscriptions,
- credit balances,
- public registration,
- cloud storage of recordings,
- public upload API,
- GPU infrastructure,
- Android,
- web dashboard.

The backend can initially know exactly one user.

Authentication can be a simple random private bearer token stored in iOS Keychain.

The OpenAI API key must remain on the server and should never be embedded in the mobile app.

---

# 6. Alpha 0.1 — target milestone

The first tangible milestone is:

> I press REC on my iPhone → lock the screen → finish the meeting → receive a local Swedish transcript → press Generate → receive the existing Luna/Terra grounded service note on the phone.

This is the definition of **Alpha 0.1**.

---

# 7. Roadmap

## Phase 0 — Freeze the existing note engine

Do not rewrite Luna/Terra logic in Swift.

Keep the current `generate_meeting_note.py` flow as the source of truth:

```text
timestamped transcript
  ↓
Luna draft
  ↓
Terra verification
  ↓
structured grounded note
```

Add a thin HTTP API around it.

Example conceptual endpoint:

```http
POST /note
```

Input:

```json
{
  "transcript": [],
  "language": "sv",
  "template": null,
  "agenda": null,
  "references": []
}
```

Output should include:

```text
note
verification warnings
usage
estimated / actual cost
```

Important backend rule:

> `/note` should not accept audio.

---

## Phase 1 — Native iOS recorder

Build the smallest useful iOS application.

Initial screen:

```text
New recording

      ● REC

00:42:17

[ Stop ]
```

Required functions:

- start recording,
- pause,
- resume,
- stop,
- recording name,
- creation date,
- local list of recordings,
- local delete,
- recording while screen is locked,
- reasonable interruption recovery,
- safe local persistence.

The recording remains inside the application's local storage.

Do **not** start by integrating Voice Memos import.

The primary workflow should be:

```text
meeting starts
→ open MobileNotes
→ REC
```

---

## Phase 2 — Local transcription on iPhone

After Stop:

```text
Recording finished

Transcribing locally...
██████████ 72%
```

Generate a timestamped transcript locally.

Preferred normalized format:

```json
[
  {
    "start": 0.0,
    "end": 4.8,
    "text": "..."
  }
]
```

This should be intentionally close to the segment format already used by the current Python pipeline.

After transcription:

```text
Meeting 22 Aug

Audio:        stored locally
Transcript:   ready

[ Transcript ]

[ Generate note ]
```

At this point **nothing has been uploaded yet**.

---

## Phase 3 — Apple vs KB-Whisper benchmark

Before extending the app too far, benchmark local iOS transcription against KB-Whisper using real Swedish recordings.

Test both transcription quality and downstream note quality.

Evaluation should include:

- important names,
- decisions,
- dates,
- numbers,
- technical terms,
- action items,
- whether transcript errors change the final Luna/Terra note.

Decision gate:

### Apple good enough

Use on-device transcription as the standard architecture.

### Apple not good enough

Keep local transcription for privacy / quick mode but consider enhanced server-side transcription later.

Do not build the server-side GPU path before this decision is necessary.

---

## Phase 4 — First backend call: Generate note

Only after the user presses:

```text
Generate note
```

send transcript text to the private backend.

Architecture:

```text
iPhone
  │
  │ transcript text only
  ▼
Private note API
  │
  ├── Luna
  ├── Terra
  │
  ▼
iPhone
```

For the private alpha:

- one user,
- private bearer token,
- token stored in Keychain,
- no registration,
- no password reset,
- no account UI.

---

## Phase 5 — Native note view

The first output should be shown directly in the application.

Example:

```text
NOTE

Ärende
...

Sammanfattning
...

Beslut
...

Åtgärder
...

⚠ 2 punkter kräver kontroll
```

Keep transcript grounding metadata.

A very useful feature can be:

```text
tap note statement
→ show supporting transcript segment(s)
```

This can start as a debugging feature and later become a major trust / transparency feature of the product.

Do not make DOCX the core UI format.

Export can come later:

```text
Share
→ PDF
→ DOCX
→ copy
→ email
```

---

## Phase 6 — Mall / template + agenda

Add these before generic reference material because they are simpler and likely very useful.

Possible generation screen:

```text
Note type
Tjänsteanteckning ▼

Template
[ Standard ] >

Agenda
[ + Add agenda ]

Reference material
[ + Add ]
```

Initially a template can simply be locally stored text describing preferred structure and style.

Agenda can also be plain text.

Rules:

- template affects structure,
- agenda affects organization,
- neither is evidence that an event or decision occurred.

---

## Phase 7 — Reference material

Add optional reference materials.

Initial supported forms can be deliberately limited to:

- TXT,
- Markdown,
- PDF,
- pasted text.

Example:

```text
Reference material

✓ Projektbeskrivning.pdf
✓ Förra mötet.txt
✓ Terminologi.md
```

Prefer extracting text locally and sending the extracted text rather than uploading the original file whenever practical.

Desired privacy model:

```text
audio           NEVER uploaded by default
original files  preferably NEVER uploaded
extracted text  uploaded only after Generate
```

The UI must clearly state what will leave the device.

---

## Phase 8 — Cost preflight

Before building actual credits, introduce cost estimation.

Once transcription and optional materials are known, estimate:

- transcript tokens,
- agenda tokens,
- reference tokens,
- Luna input,
- expected Luna output,
- Terra input,
- expected Terra output.

Example UI:

```text
Ready to generate

Transcript       14,230 tokens
Agenda              620
References         4,810

Estimated API cost
Luna              ~$0.xx
Terra             ~$0.xx
Total             ~$0.xx

[ Generate ]
```

After completion:

```text
Estimated   $0.084
Actual      $0.079
```

The current note pipeline already collects usage information from both API requests, so that mechanism should become the source for actual cost accounting.

Do not hard-code model prices permanently in the app. Pricing/configuration belongs on the backend.

---

## Phase 9 — Long-use reliability tests

Before thinking about public launch, test the application as an actual daily tool.

Minimum scenarios:

| Scenario | Expected behavior |
|---|---|
| 5 minute recording | works |
| 90 minute recording | works |
| screen locked | recording continues |
| app in background | recording continues as designed |
| temporary loss of internet | audio and transcript remain safe |
| no internet during Generate | retry later |
| Luna succeeds, Terra fails | retry only the failed stage if possible |
| app closes during transcription | recover / restart safely |
| low storage | warn before or during recording |
| incoming call / audio interruption | recover predictably |
| backend temporarily unavailable | note generation can be retried |

Also test:

- battery consumption,
- local storage growth,
- deletion behavior,
- large transcripts,
- unusual encodings / punctuation,
- mixed Swedish / English terminology,
- API timeout handling.

---

# 8. Personal-tool stopping point

After the phases above, stop feature development and use the application in real work for a while.

Target architecture at that point:

```text
RECORD
   ↓
LOCAL AUDIO
   ↓
LOCAL TRANSCRIPTION
   ↓
optional:
 template
 agenda
 references
   ↓
COST PREVIEW
   ↓
send TEXT only
   ↓
LUNA
   ↓
TERRA
   ↓
GROUNDED NOTE
   ↓
save locally
```

Use it on roughly 20–50 real meetings before deciding whether the concept deserves commercial infrastructure.

The important test is simple:

> Do I continue voluntarily using this instead of going back to manual recording/export/note workflows?

If yes, proceed toward productization.

If not, learn why before building accounts and billing.

---

# 9. Commercialization gate

Only after the private alpha proves useful should the project move from:

```text
personal tool
```

to:

```text
commercial product
```

Then add, approximately in this order:

```text
accounts
→ secure multi-user auth
→ storage / retention policy
→ organizations / teams if needed
→ processing ledger
→ credits
→ StoreKit
→ subscriptions
→ external TestFlight
→ App Store
```

---

# 10. Future credit model

Do not expose raw API dollars as the core product abstraction.

Prefer internal **processing units**.

Why:

- model pricing can change,
- Luna/Terra may later be replaced,
- reasoning effort may differ,
- some note types may use one verification pass and others two,
- reference-heavy notes cost more,
- server-side transcription may become optional.

The backend can convert predicted model usage to processing units before generation and settle against actual usage after generation.

Possible future subscription concept:

```text
Monthly plan
→ included processing allowance

Optional top-up
→ additional processing units
```

Exact App Store billing mechanics should be designed only when commercialization begins.

---

# 11. Critical risks

## 11.1 Transcription quality

This is the first technical risk.

If local transcription damages names, decisions, numbers or technical terms badly enough, no later LLM verification can reliably reconstruct what was actually said.

Mitigation:

- real benchmark,
- retain timestamps,
- preserve original audio locally for user review,
- support corrections,
- potentially offer enhanced transcription later.

## 11.2 Speaker attribution

The current repository intentionally removed diarization because it was slow, GPU-heavy and fragile.

That is reasonable for the first product.

Therefore the app must not initially promise reliable speaker attribution.

Possible later options:

- manual speaker labels,
- user correction,
- optional diarization,
- speaker attribution only when explicitly supported by transcript/context.

## 11.3 Privacy

Privacy is one of the main product risks and also potentially one of the strongest differentiators.

Preferred default:

```text
Audio stays on device.
Only text selected for generation leaves the device.
```

The user must understand exactly what is being sent.

Especially important for:

- meeting transcripts,
- reference documents,
- municipal / professional information,
- personal data,
- confidential work material.

Future commercial work will require a proper privacy architecture, retention policy, legal review and clear user controls.

## 11.4 Reference material becoming false evidence

This is a subtle but serious failure mode.

Mitigation:

- explicit source typing,
- strict prompt separation,
- Terra verification,
- evidence metadata,
- visible distinction between transcript-supported and context-derived statements.

## 11.5 iOS long-running behavior

Recording and transcription must survive realistic phone use:

- screen locking,
- app backgrounding,
- interruptions,
- long recordings,
- battery constraints.

This must be tested early on a physical iPhone rather than assumed from simulator behavior.

## 11.6 Backend security

Even for the private alpha:

- OpenAI key stays server-side,
- mobile app uses a separate token,
- rate-limit requests,
- reject unexpected payloads,
- never expose provider credentials to the client.

Commercialization later requires proper user authentication and abuse protection.

---

# 12. Features deliberately postponed

Do **not** build these before the private alpha proves itself:

- diarization,
- Android,
- web dashboard,
- team administration,
- user organizations,
- shared workspaces,
- cloud recording storage,
- public API,
- user registration,
- billing,
- StoreKit,
- subscriptions,
- complex sync,
- custom GPU infrastructure.

Every one of these can become a project of its own and distract from validating the core workflow.

---

# 13. Potential differentiator

The most interesting direction is not automatic summarization by itself.

It is:

> **A note generator that understands the user's structure, agenda and domain context while remaining explicit about what was actually supported by the recorded meeting.**

That combination gives the application a more professional character than a generic meeting summarizer.

The existing Luna → Terra → evidence verification pipeline is already a useful prototype of this trust layer.

---

# 14. First implementation task when this project resumes

Do not start with billing or backend redesign.

Start with a physical iPhone and implement only:

```text
REC
→ local file
→ local transcription
→ timestamped transcript view
```

Then benchmark that transcript against the existing KB-Whisper workflow.

Only after that should `/note` be connected.

This keeps the first decision small, measurable and reversible.
