# Northeastern Aladdin – Specialized AI Assistant

## Overview
**Northeastern Aladdin** is a specialized AI assistant project developed for the *Prompt Engineering* course at Northeastern University.  
The assistant replicates aspects of **BlackRock’s Aladdin platform** for portfolio management, risk analytics, trading, and operations — but redesigned as an **educational, student-focused quant-mentor**.  

The project leverages prompt-engineering techniques, schema-constrained responses, uploaded datasets (PitchBook, MSCI, FRED, etc.), and a custom validation framework.  
Its goals are to **simulate financial advising**, integrate ESG and behavioral dimensions, and maintain strict schema + policy compliance.

---

## Repository Structure
NORTHEASTERN-ALADDIN-TESTS/
│
├── artifacts/                  # Raw failing responses (for debugging)
├── responses/                  # AI-generated responses (Txx.txt, FNLxx.txt)
├── runs/
│   ├── specialized-ai-assistant.jsonl        # JSONL test cases (id, name, prompt, expects)
│   ├── specialized-ai-assistant_summary.csv  # CSV summary of validation results
│
├── validate.py                 # Inline validator (basic checks)
├── validate_files.py           # File-based validator (advanced checks)
└── README.md                   # Project documentation

---

## Assistant Persona & Goals
**Persona**  
Northeastern Aladdin is a **precise, empathetic quant-mentor**, adapting to both formal (`"We recommend diversified allocation…"`) and casual (`"Let’s mix in green stocks!"`) tones.  
It links finance to psychology (loss aversion), society (equity audits), math (stochastic models), and nature (climate resilience).  

**Core Goals**
1. **Portfolio Optimization** – Mean-variance optimization with Sharpe maximization, long-only, 0.1% pro-rata fees.  
2. **Risk Simulation** – Monte Carlo (1,000 paths), VaR (95%), climate stress testing, behavioral + bias flagging.  
3. **Trading & Operations Support** – API-like flows with compliance alignment (SEC, MiFID II) and reflective prompts.

---

## Core Assistant Functions
The project implements the following core modules (mapped to assignment requirements):

1. **Knowledge Q&A Module** – Layered explanations of financial concepts (Sharpe ratio, mean-variance, etc.).  
2. **Step-by-Step Process Guide** – Numbered workflows (mean–variance procedure, execution flow).  
3. **Real-World Connections** – Case examples (e.g., diversification during 2020 market crash).  
4. **Critical Thinking & Reflection** – Assistant prompts users with thoughtful questions to refine constraints.  
5. **Resource Curation** – Summarizes uploaded files (PitchBook, MSCI methodology, FRED datasets).  
6. **Visual Aids Creation** – Text-based diagrams and efficient frontier chart descriptions.  
7. **Personalized Assessment** – Rubric-based evaluation of allocations and feedback.  
8. **Action Planning** – Generates structured plans with turnover caps and fee reminders.  
9. **Skill Development Tracking** – Week-by-week progress trackers.  
10. **Content Templates** – Portfolio review memos and ops checklists.

---

## Validation Framework
To ensure assistant responses follow the **format contract** and **policy rules**, two validators are provided:

### `validate.py` (Inline Validator)
- Basic JSONL test runner (reads inline responses).  
- Checks JSON validity, word limits, disclaimers, and noise.  

### `validate_files.py` (File-Based Validator)
- Advanced validator for `responses/` directory.  
- Extracts and parses JSON blocks + prose summary.  
- Enforces domain rules (e.g., no shorts, climate stress references, MSCI citations).  
- Generates `runs/specialized-ai-assistant_summary.csv` with pass/fail status.  
- Stores raw failing outputs in `artifacts/` for review.  

**Example Run**
```bash
# Validate AI responses
python validate_files.py \
  --tests runs/specialized-ai-assistant.jsonl \
  --responses_dir responses \
  --summary_csv runs/specialized-ai-assistant_summary.csv \
  --artifacts_dir artifacts

Latest run summary (Python 3.11.9):

Total Tests: 100
Passed: 79
Failed: 21
Pass Rate: 79%
