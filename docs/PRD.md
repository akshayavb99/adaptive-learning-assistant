# Product Requirements Document (PRD)

# AI Testing Sandwich Knowledge Engine

**Version:** 0.5
**Status:** Draft

---

# 1. Vision

Build an AI-powered learning platform that transforms a Markdown knowledge base into adaptive assessments based on the **Testing Sandwich** learning methodology.

The platform supports any knowledge domain where information can be represented as Markdown files, including:

* Programming
* Mathematics
* History
* Science
* Engineering
* Law
* Medicine
* Personal knowledge systems

The system uses Markdown as the **source of truth** and creates an intelligent assessment layer on top.

```text
Markdown Knowledge Base
          ↓
Concept Model
          ↓
AI Generated Assessments
          ↓
Learning Analytics
          ↓
Unique Question Repository
```

The platform continuously improves by building a high-quality question bank while ensuring each learner receives fresh, adaptive assessments.

---

# 2. Problem Statement

Traditional study workflows have several problems:

* Notes are passive and do not test understanding.
* Learners cannot easily measure knowledge gaps.
* AI-generated questions are often repetitive.
* Generated questions are discarded after use.
* Existing question banks are disconnected from personal notes.
* Learning progress is difficult to quantify.
* Knowledge bases change over time, but learning systems often become stale.
* AI usage and cost are difficult to monitor.

This product solves these problems by:

* Converting Markdown notes into structured knowledge.
* Generating assessments directly from selected topics.
* Adapting difficulty based on learner performance.
* Maintaining a repository of unique assessment questions.
* Synchronizing with changing Markdown sources.
* Providing learning and AI operational analytics.

---

# 3. Product Principles

## 3.1 Markdown is the Source of Truth

Markdown files are the authoritative knowledge source.

The system derives:

* Subjects
* Topics
* Concepts
* Learning objectives
* Questions
* Tests
* Analytics

from Markdown.

Questions never replace or override source knowledge.

---

## 3.2 Concepts are the Core Learning Unit

Questions are generated from concepts, not directly from documents.

Architecture:

```text
Document
    ↓
Concept
    ↓
Learning Objective
    ↓
Question
    ↓
Test
```

---

## 3.3 Tests are Ephemeral

A test represents one assessment session.

Tests are:

* Generated dynamically
* Not reused
* Not considered learning assets

Questions generated during a test may become question bank entries.

---

## 3.4 Always Generate Fresh Questions

Every assessment generates new questions.

The question bank is **not used to populate tests**.

The question bank is used for:

* Duplicate detection
* Coverage analysis
* Question analytics
* Future export
* Repository quality management

---

## 3.5 Question Bank Quality Over Quantity

The question bank should contain only unique, useful assessment items.

Similar questions should not accumulate.

---

## 3.6 Incremental Knowledge Synchronization

The system should detect changes in Markdown sources and update only affected knowledge.

Users should not need to rebuild the entire knowledge base after every change.

---

# 4. Goals

## 4.1 Functional Goals

The system should:

* Import Markdown knowledge bases.
* Detect Markdown changes.
* Refresh only modified content.
* Extract concepts automatically.
* Generate learning objectives.
* Allow subject and topic selection.
* Generate fresh assessments.
* Support multiple question types.
* Adapt question difficulty.
* Evaluate answers.
* Maintain a question repository.
* Track learning progress.
* Provide analytics dashboards.

---

## 4.2 Non-Functional Goals

The system should be:

* Domain independent.
* Cost observable.
* Efficient with LLM usage.
* Scalable to large knowledge bases.
* Extensible.
* Maintainable.

---

# 5. User Workflow

```text
Import Markdown Knowledge Base
              ↓
AI Extracts Knowledge Model
              ↓
User Selects Subject
              ↓
User Selects Topics
              ↓
User Selects Test Type
              ↓
AI Generates Fresh Assessment
              ↓
User Completes Test
              ↓
AI Evaluates Answers
              ↓
Question Bank Insertion Pipeline
              ↓
Analytics Updated
```

---

# 6. High-Level Architecture

```text
                    Markdown Repository
                            │
                            ▼
                    Change Detection
                            │
                            ▼
                    Markdown Processor
                            │
                            ▼
                    Concept Extraction
                            │
                            ▼
                    Knowledge Store
                            │
                            ▼
                    Question Generator
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
       Current Test                Question Bank
                                      Insertion
                                          │
                                          ▼
                               Duplicate Detection
                                          │
                              ┌───────────┴───────────┐
                              │                       │
                           Unique                Duplicate
                              │                       │
                              ▼                       ▼
                       Store Question             Discard
```

---

# 7. Core Data Model

## 7.1 Document

Represents a Markdown source file.

Fields:

```text
id

title

path

hash

source_last_updated_date

last_processed_date

processing_status
```

---

## 7.2 Subject

Fields:

```text
id

name
```

Example:

```text
Programming
History
Mathematics
```

---

## 7.3 Topic

Fields:

```text
id

subject_id

title

description
```

---

## 7.4 Concept

The smallest meaningful learning unit.

Fields:

```text
id

topic_id

document_id

title

description

importance

difficulty

relationships
```

---

## 7.5 Learning Objective

Fields:

```text
id

concept_id

objective

cognitive_level
```

Examples:

* Define ownership.
* Explain borrowing.
* Compare algorithms.
* Apply principles.

---

## 7.6 Question

Reusable assessment item.

Fields:

```text
id

concept_ids

learning_objective_id

question_type

cognitive_level

difficulty

question_text

answer

explanation

embedding

status

created_at
```

Question status:

```text
active

needs_review

outdated
```

---

## 7.7 Test

Temporary assessment.

Fields:

```text
id

subject

topics

test_type

created_at
```

---

## 7.8 Attempt

Fields:

```text
question_id

answer

score

confidence

time_taken

notes
```

---

## 7.9 Progress

Fields:

```text
concept_id

mastery_score

average_score

attempt_count
```

---

# 8. Knowledge Base Refresh

## Purpose

Allow users to update the AI knowledge model when Markdown files change.

---

## User Action

User selects:

**Refresh Knowledge Base**

Options:

### Refresh Changed Files

Default.

Only processes files where:

```text
source_last_updated_date >
last_processed_date
```

---

### Full Refresh

Processes all Markdown files.

Used when:

* AI extraction logic changes.
* Knowledge structure changes.
* User wants a rebuild.

---

# 9. Refresh Pipeline

```text
User Requests Refresh
          ↓
Scan Markdown Repository
          ↓
Compare File Metadata
          ↓
Identify Changed Files
          ↓
Process Changed Files
          ↓
Update Concepts
          ↓
Update Learning Objectives
          ↓
Update Question Relationships
          ↓
Refresh Analytics
```

---

# 10. Changed Content Handling

When a Markdown file changes:

The system compares:

* Previous extracted concepts
* New extracted concepts

Identify:

* Added concepts
* Modified concepts
* Removed concepts

---

## New Concept

Action:

* Create concept.
* Generate objectives.
* Include in future assessments.

---

## Modified Concept

Action:

* Update concept.
* Preserve learning history.
* Review affected questions.

---

## Removed Concept

Action:

* Mark inactive.
* Preserve historical data.

Do not immediately delete.

---

# 11. Question Impact from Knowledge Changes

When concepts change:

Associated questions are evaluated.

Possible states:

## Active

Still valid.

---

## Needs Review

Potentially affected.

---

## Outdated

No longer aligned with source knowledge.

---

Questions are not automatically deleted.

---

# 12. Assessment Generation

Questions are generated from:

```text
Concept

↓

Learning Objective

↓

Question Type

↓

Difficulty

↓

LLM
```

Questions are never generated directly from Markdown files.

---

# 13. Supported Question Types (MVP)

* Multiple Choice Questions
* Short Answer Questions
* Long Answer Questions
* True / False
* Fill in the Blank
* Comparison
* Explanation
* Application
* Scenario

---

# 14. Test Configuration

Users select:

## Subject

Example:

```text
Programming
```

---

## Topics

Example:

```text
Rust

- Ownership
- Borrowing
- Lifetimes
```

---

## Test Type

### Pre-study

Purpose:

Measure baseline knowledge.

Characteristics:

* Broad coverage
* Lower difficulty
* Recall focused

---

### Post-study

Purpose:

Measure learning improvement.

Characteristics:

* More difficult
* More application
* More reasoning

---

### Review

Purpose:

Measure retained knowledge.

Characteristics:

* Adaptive difficulty
* Targets weak concepts

---

## Test Settings

Users configure:

* Number of questions
* Question type distribution
* Topic coverage
* Difficulty mode

---

# 15. Adaptive Difficulty (MVP)

Difficulty adapts based on:

* Previous scores
* Concept mastery
* Question history
* Incorrect answers
* Confidence ratings

Difficulty levels:

```text
Beginner

Easy

Medium

Hard

Expert
```

Adaptation occurs per concept.

Example:

```text
Ownership: Hard

Lifetimes: Easy
```

within the same test.

---

# 16. Test Generation Pipeline

Every test generates fresh questions.

```text
Select Topics
      ↓
Identify Concepts
      ↓
Generate Questions
      ↓
Deliver Test
      ↓
Evaluate Answers
      ↓
Question Bank Insertion
```

Rules:

* Existing questions are never reused for tests.
* Questions are generated specifically for the current learner.
* Duplicate detection occurs only during question bank insertion.
* A duplicate question can still appear in the current test.

---

# 17. Question Bank Insertion Pipeline

Only unique questions are stored.

```text
Generated Question
        ↓
Extract Metadata
        ↓
Generate Embedding
        ↓
Search Similar Questions
        ↓
Similarity Check
        ↓
LLM Verification
        ↓
Insert or Reject
```

---

# 18. Duplicate Detection

Compare against existing questions with:

* Same concepts
* Same learning objective
* Similar cognitive level
* Similar question type

---

Similarity threshold:

```text
0.95 cosine similarity
```

A similarity match creates a duplicate candidate.

---

LLM verification asks:

> Are these questions testing the same learning objective?

---

Outcome:

## Same Intent

Reject.

---

## Different Intent

Insert.

---

# 19. Coverage Model

Coverage is measured by:

```text
Concept

×

Learning Objective

×

Question Type

×

Cognitive Level
```

Example:

| Concept   | Recall | Explain | Apply | Analyze |
| --------- | ------ | ------- | ----- | ------- |
| Ownership | ✓      | ✓       | ✓     | ✗       |
| Borrowing | ✓      | ✓       | ✗     | ✗       |

---

# 20. Answer Evaluation

## MCQ

* Correctness

## SAQ

* Accuracy
* Completeness
* Missing concepts

## LAQ

Evaluation:

* Structure
* Reasoning
* Concept coverage
* Accuracy
* Missing details

---

# 21. Analytics Dashboard

## Learning Analytics

Metrics:

* Average score
* Tests completed
* Questions answered
* Subjects studied

---

## Subject Progress

Example:

```text
Programming
82%

History
76%
```

---

## Topic Progress

Example:

```text
Ownership
88%

Borrowing
74%

Lifetimes
61%
```

---

## Question Type Performance

Example:

```text
MCQ
91%

SAQ
78%

LAQ
64%
```

---

## Test Type Performance

Example:

```text
Pre
45%

Post
82%

Review
79%
```

---

# 22. AI Usage Analytics

## Usage

Track:

* LLM requests
* Requests by operation
* Prompt tokens
* Completion tokens

Operations:

* Concept extraction
* Question generation
* Answer evaluation
* Duplicate verification

---

## Cost

Track:

* Total cost
* Cost per test
* Cost per question
* Cost per evaluated answer
* Cost by subject/topic

---

## Performance

Track:

* Generation latency
* Evaluation latency
* Duplicate detection latency

---

## Knowledge Base Analytics

Track:

* Markdown files
* Last refresh date
* Files processed
* Concepts added
* Concepts modified
* Concepts removed
* Questions requiring review

---

# 23. Success Metrics

## Learning

* Pre-test to post-test improvement
* Topic improvement
* Subject improvement
* Long-answer evaluation reliability

---

## Question Quality

Targets:

* <5% duplicate questions stored
* > 90% valid questions
* Balanced coverage
* Effective difficulty adaptation

---

## System

Targets:

* Predictable AI cost
* Efficient refresh processing
* Low duplicate detection latency
* Growing question coverage

---

# 24. MVP Scope

## Included

### Knowledge Management

* Markdown ingestion
* Incremental refresh
* Changed file detection
* Concept extraction
* Learning objective generation
* Topic/subject organization

---

### Assessment

* Subject selection
* Topic selection
* Pre/Post/Review tests
* Fresh question generation
* Adaptive difficulty
* MCQ
* SAQ
* LAQ
* AI answer evaluation

---

### Question Bank

* Question persistence
* Embeddings
* Duplicate detection
* LLM verification
* Coverage tracking
* Question status tracking

---

### Analytics

* Learning dashboard
* Subject/topic progress
* Question type analytics
* LLM usage metrics
* Cost tracking
* Knowledge refresh analytics

---

## Excluded

* Spaced repetition
* Flashcards
* Mobile applications
* Collaboration
* Community question sharing
* Multimedia questions
* Human moderation
* Fine-tuned models

---

# 25. Key Architectural Decisions

1. Markdown files are the canonical knowledge source.
2. Concepts are the fundamental learning units.
3. Questions are always generated fresh for assessments.
4. Tests are temporary.
5. Question bank insertion happens after test generation.
6. Duplicate detection occurs immediately before question persistence.
7. Duplicate detection does not affect the current test.
8. Semantic similarity identifies candidates; LLM verification decides duplicates.
9. Adaptive difficulty is included in the MVP.
10. Knowledge refresh uses file timestamps and metadata to process only changed Markdown sources.
11. Learning analytics and AI cost analytics are first-class features.
12. Question history and learner progress are preserved even when source knowledge evolves.
