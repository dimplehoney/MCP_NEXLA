# Example Interaction Log

These outputs were captured by running `python scripts/demo.py` against the
indexed corpus (TCS 2020, Berkshire Hathaway 2021, Tencent Music 2021,
Salesforce 2020, ASX_AJY 2020). They are real model outputs — not hand-edited.

---

## Q1 — Single-document factual lookup

**Question:** What was Salesforce's total revenue in fiscal year 2020?

**Answer:**
> Salesforce's total revenue in fiscal year 2020 was $17.1 billion.

**Sources:**
- `NYSE_CRM_2020` — page 45
  > "Total fiscal 2020 revenue was $17.1 billion, an increase of 29 percent year-over-year."

---

## Q2 — Cross-document comparison

**Question:** Compare the risk factors discussed by Salesforce and Berkshire Hathaway.

**Answer:**
> Berkshire faces long-term insurance risk; Salesforce deals with stock price volatility.

**Sources:**
- `NYSE_BRK-A_2021` — page 139
  > "Mistakes in assessing insurance risks can be huge..."
- `NYSE_CRM_2020` — page 37
  > "The market price of our common stock is likely to be volatile..."

---

## Q3 — Specific factual lookup

**Question:** What was TCS's attrition rate in IT Services in FY2020?

**Answer:**
> TCS's IT services attrition rate in FY 2020 was 12.1%.

**Sources:**
- `OTC_TCS_2020` — page 90
  > "TCS' IT services attrition rate was 12.1%."
- `OTC_TCS_2020` — page 5
  > "Our attrition Commitment and Trust in IT services in FY 2020 was 12.1%."

---

## Q4 — Out-of-scope (hallucination guard)

**Question:** What is the company's stated policy on cryptocurrency or blockchain investments?

**Answer:**
> Not found in documents.

**Sources:** (none)

> No retrieved chunk passed the distance threshold (< 0.7), so the LLM was
> never called — the fallback was returned directly. This confirms the
> hallucination control is working as intended.
