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
> Berkshire Hathaway discusses the inherent risks in the insurance business,
> emphasizing that underwriting profits are not guaranteed and that significant
> mistakes in risk assessment can lead to substantial losses, potentially from
> unforeseen events like cyber attacks or natural disasters. They acknowledge
> that while they have had a strong underwriting record, future profits are
> uncertain and major catastrophes could impact their financials significantly
> (Source 1, Page 139; Source 4, Page 139). In contrast, Salesforce highlights
> the volatility of its common stock, which is influenced by various operational
> metrics and market conditions. They note that their financial success is
> dependent on liquidity events and that their investments carry risks of
> partial or total loss (Source 7, Page 67; Source 8, Page 37). Overall, while
> both companies recognize significant risks, Berkshire focuses on underwriting
> and catastrophic events in insurance, whereas Salesforce emphasizes stock
> price volatility and investment risks.

**Sources:**
- `NYSE_BRK-A_2021` — page 139
  > "Mistakes in assessing insurance risks can be huge and can take many years – even decades – to surface and ripen."
- `NYSE_BRK-A_2021` — page 139
  > "We will most certainly not have an underwriting profit in 16 of the next 17 years."
- `NYSE_CRM_2020` — page 67
  > "the financial success of our investment in any company is typically dependent on a liquidity event."
- `NYSE_CRM_2020` — page 37
  > "the market price of our common stock is likely to be volatile and could subject us to litigation."

---

## Q3 — Specific factual lookup

**Question:** What does TCS say about its workforce or employee headcount?

**Answer:**
> As of FY 2020, TCS had a total of 448,464 employees, with 36.2% being women
> and representing 144 nationalities. The average age of employees was 31 years.
> Additionally, TCS reported an attrition rate of 12.1% in its IT services sector.

**Sources:**
- `OTC_TCS_2020` — page 85
  > "448,464 employees, 36.2% women, 144 nationalities, 31 years average age, 12.1% Attrition in IT services"

---

## Q4 — Out-of-scope (hallucination guard)

**Question:** What is the company's stated policy on cryptocurrency or blockchain investments?

**Answer:**
> Not found in documents.

**Sources:** (none)

> No retrieved chunk passed the distance threshold (< 0.7), so the LLM was
> never called — the fallback was returned directly. This confirms the
> hallucination control is working as intended.
