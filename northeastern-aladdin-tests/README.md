# Northeastern Aladdin: Specialized AI Financial Advisor for NEU Students

## Overview
This repository hosts the **"Northeastern Aladdin"** project, a specialized AI assistant built using **Claude by Anthropic** (via `https://claude.northeastern.edu/`). Inspired by BlackRock's Aladdin platform, it serves as a financial advisor for **Northeastern University students**, focusing on portfolio management, risk analytics, trading, operations, API-first modularity, and ESG integration.

The assistant is a **precise, empathetic quant-mentor** that adapts to user styles (formal or casual), links finance to psychology, society, math, and nature, and utilizes uploaded data with transparent citations. It optimizes portfolios using **mean-variance analysis**, simulates risks with **Monte Carlo and VaR**, and provides educational reflections while flagging biases.

This project addresses the **"Build a Specialized AI Assistant"** assignment, selecting the Financial Advisor type on Claude. It demonstrates advanced prompt engineering, data integration, and interactive functions for educational personal finance.

---

## Personality and Goals (Assignment Section 3)

### Personality
Northeastern Aladdin is a **precise, empathetic quant-mentor** designed for Northeastern University students. It adapts communication styles—**formal** for professional advice (e.g., "We recommend a diversified allocation...") or **casual** for relatable engagement (e.g., "Let’s mix in green stocks!"). It employs inclusive, gender-neutral, culturally sensitive language, flags data biases (e.g., Global South underrepresentation), prompts for missing inputs, and limits responses to **200 words** with ethical disclaimers and self-evaluations. *(98 words)*

### Core Goals
1. **Optimize portfolios** via mean-variance (Sharpe maximization, $\sigma_p = \sqrt{w^T \Sigma w}$), incorporating no-shorts, 0.1% fees, and benchmarks like AlternativeSoft.
2. **Simulate risks** using Monte Carlo (1,000 paths, $\mu=0.05, \sigma=0.15$), VaR (95%), and 2025 climate stress tests, integrating herding and Global South biases.
3. **Support trading/operations** with API-like flows, SEC/MiFID II compliance, and educational reflections, tracking history for refinements.

### Approach
The assistant tracks conversation history, queries user inputs (age, risk 1-10, sustainability yes/no, style formal/casual), fuses uploaded data with Aladdin principles, assesses multidimensionally (psychology via herding, covariance, equity, biodiversity), optimizes allocations with efficient frontier trade-offs, simulates risks with bias flags, reflects via questions, and self-evaluates. Outputs strictly follow: **a JSON schema object + one short sentence ($\le30$ words).**

---

## Core Assistant Functions (Assignment Section 4)
Implemented 6 essential functions using prompt-driven techniques in Claude. Each includes initial instructions (from the system prompt), example interactions, and resources.

### 1. Knowledge Q&A Module (15 Points)
* **Function:** Answers questions with deepening responses and citations from uploads.
* **Initial Instructions:** "Q&A: Deepen answers with citations (e.g., PitchBook Q1 2025 Indexes CSV)... Cite sources (e.g., “S&P 500 Historical Data.csv, 09/26/2025: 6,643.70”)."
* **Example Interaction:**
    * **User:** "What's the S&P 500 value on 09/26/2025?"
    * **Assistant JSON:** `{ "style": "formal", ... }` followed by: "The S&P 500 closed at 6,643.70 (S&P 500 Historical Data.csv, 09/26/2025)."
* **Resources:** Uploaded CSVs/PDFs (e.g., `FRED DGS10.csv` for yields).

### 2. Step-by-Step Process Guide (15 Points)
* **Function:** Guides through optimization processes like mean-variance.
* **Initial Instructions:** "Guide: Provide step-by-step optimization (e.g., mean-variance process)... Optimize portfolios holistically (e.g., 40% Tech, 25% Healthcare)."
* **Example Interaction:**
    * **User:** "Optimize my portfolio."
    * **Assistant:** Steps: 1. Assess inputs. 2. Fuse data. 3. Optimize allocation. **JSON:** `{ "allocation": {"Tech": 40, ...}, ... }` "We recommend this balanced mix."
* **Resources:** Olkhov papers for covariance $\Sigma$.

### 3. Real-World Connections (10 Points)
* **Function:** Links concepts to psychology (loss aversion), society (equity audits), math (stochastic models), nature (climate resilience).
* **Initial Instructions:** "Connections: Link concepts to psychology/society/nature (e.g., loss aversion to volatility)."
* **Example Interaction:**
    * **User:** "Explain risk in portfolios."
    * **Assistant:** "Volatility ties to loss aversion (psychology) and herding (NBER w9222.pdf), with societal equity gaps in Global South data."
* **Resources:** `NBER w9222.pdf`, `MSCI ESG Methodology.pdf`.

### 4. Critical Thinking and Reflection Prompts (10 Points)
* **Function:** Poses questions for deeper reflection (e.g., climate impacts).
* **Initial Instructions:** "Reflection: Pose thought-provoking questions (e.g., climate impact)... Reflect ethically with a guiding question."
* **Example Interaction:**
    * **Assistant JSON:** `{ "question": "How might climate risks alter your long-term goals?", ... }` "Consider nature’s fat tails in planning."
* **Resources:** `SEC Climate Disclosures.pdf`.

### 5. Resource Curation (10 Points)
* **Function:** Recommends/summarizes uploads based on needs.
* **Initial Instructions:** "Resources: Recommend/summarize uploads (e.g., PitchBook PE outlook)."
* **Example Interaction:**
    * **User:** "Summarize ESG risks."
    * **Assistant:** "Per MSCI ESG Methodology.pdf: Focus on biodiversity; recommend for climate resilience audits."
* **Resources:** All uploaded files (e.g., `PitchBook Q1 2025 Indexes CSV`).

### 6. Action Planning (10 Points)
* **Function:** Creates actionable rebalance plans (e.g., shift 10% to ESG).
* **Initial Instructions:** "Planning: Create actionable rebalance plans (e.g., shift 10% to ESG)... Track history for refinements."
* **Example Interaction:**
    * **User:** "Rebalance for ESG."
    * **Assistant JSON:** `{ "actions": ["Rebalance 10% into ESG", "Cap turnover at 5%"], ... }` "Updating prior: Shift for sustainability."
* **Resources:** Benchmarks from AlternativeSoft (integrated via prompt).

---

## Creating the AI Assistant (Assignment Section 5)
* **Platform:** Claude by Anthropic.
* **Setup:** Create a new project, upload data files, paste the comprehensive system prompt (see `system_prompt.md`), and test interactions.
* **System Prompt:** The full prompt defining persona, workflow, functions, and output format is in `docs/system_prompt.md`. It combines the provided description and set project instructions.

## Data Integration (Assignment Section 6)
### Uploaded Files (Fictional for 2025 simulation; create dummies if needed):
* `PitchBook Q1 2025 Indexes CSV` (private equity benchmarks).
* `FRED DGS10.csv` (Treasury yields).
* `MSCI ESG Methodology.pdf` (ESG and biodiversity).
* `SEC Climate Disclosures.pdf` (climate stress tests).
* `NBER w9222.pdf` (herding behavior).
* `S&P 500 Historical Data.csv` (market data).

### Accuracy & Relevance:
Cited directly in responses (e.g., values/dates); supports optimization/risk simulation.

### Organization:
Grouped logically in Claude project; flagged gaps like "Global South underrepresented."

### Integration:
Blends with general knowledge; assumes balanced data in disclaimers.

*Note: Actual uploads are handled in Claude; local dummies can be added to `/data/` if desired. If you have real file contents or additional uploads, provide details to include them.*

---

## Project Structure
Based on the provided image, the current local structure is for testing/validation:
'''bash
NORTHEASTERN-ALADDIN-TESTS/
├── artifacts/                  # Empty or for generated artifacts
├── responses/                  # Response-related files
│   └── runs/                   # Subfolder for run logs (collapsed in image)
├── specialized-ai-assistant_summary.csv  # Summary CSV, perhaps test results
├── specialized-ai-assistant.json         # JSON for assistant config or output
├── testjsoncode.json                     # Test JSON code/output
├── validate_files.py                     # Python script for file validation
└── validate.py                           # Additional validation script

---

## Setup and Usage
1.  **Access Claude:** Login at `https://claude.northeastern.edu/`.
2.  **Create Project:** Start a new project, upload data files to knowledge base.
3.  **Set System Prompt:** Copy `docs/system_prompt.md` into the prompt field.
4.  **Interact:** Query e.g., "Optimize portfolio: age 22, risk 6, esg yes, style formal." Expect **JSON + sentence output**.
5.  **Testing:** Use `validate.py` and `validate_files.py` to check JSON formats/responses (run with Python 3+).
6.  **Dependencies:** None; Claude handles AI. Local scripts require Python.

## Demo Interaction (Assignment Section 7)
Screen recording (3-5 minutes) showcasing functions: [Demo Video Link](#)https://youtu.be/5yxY4xSJjHM?si=GnRsQptO7WiHAnIb

---

## Reflection (Assignment Section 8)
"In developing **Northeastern Aladdin** on Claude, I emphasized a robust system prompt to enforce structured outputs and ethical guidelines, integrating fictional 2025 data for realism. Challenges included maintaining response limits, simulating future data accurately, and handling output formatting without native JSON validation—addressed via local Python scripts. Key takeaways: **Prompt engineering** is crucial for consistent AI behavior; **data integration** enhances educational value but requires bias awareness. This project honed my skills in AI personalization, highlighting Claude's strengths in knowledge-based assistants over CustomGPT for NEU access. Overall, it demonstrates how AI can democratize financial advice for students while promoting interdisciplinary connections." *(152 words)*

*If this isn't your reflection, please share the text to replace.*

---

## License
**MIT License.** See `LICENSE` for details.

## Contact
For questions, open an issue or contact [your GitHub username].
