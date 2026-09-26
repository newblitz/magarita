# Amazon ML Challenge 2026 — Business Entity Resolution
## Dataset Analysis, Candidate Generation & Modeling Insights

**Status:** Working research report  
**Last updated:** 2026-09-25

---

## 1. Executive Summary

This project addresses the **Amazon ML Challenge 2026 – Business Entity Resolution Challenge**. The objective is to resolve businesses across three independently collected, noisy sources:

- **Source 1 (S1):** deduplicated reference entities
- **Source 2 (S2):** noisy business records
- **Source 3 (S3):** another noisy business-record source

For every S1 entity, the task is to identify **all matching S2 and S3 records**. An S1 entity can have zero, one, or many matches.

The analysis so far establishes:

1. This is a **multi-match entity-resolution problem**, not top-1 classification.
2. Training contains **7,638,365 positive links** across **2,206,821 S1 entities**.
3. About **94.4%** of S1 entities have at least one match; 5.6% are true singletons.
4. Multiple matches are common; **3 matches per S1** is the most common total.
5. Exact normalized `name + address` is extremely precise but has very low recall.
6. Exact normalized name has useful recall but poor precision due to common business names.
7. Exact normalized address is much more precise but misses noisy address variants.
8. Structural blocking using house numbers, name prefixes, and address tokens substantially increases recall.
9. `country + house number + exact name` currently provides a very strong high-confidence signal, with about **97–98% precision** and **15% recall**.
10. `country + house number + name prefix` reaches about **51–54% recall**, but with a much larger candidate set.
11. The union of all tested deterministic component blocks previously reached about **85.8% candidate recall**, but generated roughly **1.3 billion candidate pairs**.
12. Therefore the next goal is a high-recall but computationally manageable retrieval layer, followed by a precision-oriented pair classifier.

Proposed architecture:

```text
S1
 ↓
Multi-block candidate generation
 ↓
Approximate retrieval for hard cases
 ↓
Similarity / structural features
 ↓
Pairwise ML classifier
 ↓
Threshold calibrated for macro F0.5
 ↓
All S2/S3 matches per S1
```

---

## 2. Dataset Scale

Approximate training records:

| File | Records |
|---|---:|
| Train S1 | 2,206,821 |
| Train S2 | 5,034,616 |
| Train S3 | 5,285,603 |
| Ground truth | 2,206,821 |

Total training records: **~12.53M**

Test records:

| File | Records |
|---|---:|
| Test S1 | 1,732,544 |
| Test S2 | 4,887,273 |
| Test S3 | 5,082,316 |

Total test records: **~11.70M**

---

## 3. Why Blocking Is Mandatory

The approximate test pair space is:

```text
1,732,544 × (4,887,273 + 5,082,316)
≈ 17.3 trillion pairs
```

Therefore brute-force comparison is impossible.

The system must reduce:

```text
~17.3 trillion possible pairs
        ↓
candidate generation
        ↓
manageable candidate set
        ↓
pairwise matcher
```

Candidate recall is the ceiling for final recall:

> A true pair that never enters the candidate set can never be recovered by the final model.

---

## 4. Source 1 EDA

Train S1:

- **2,206,821 records**
- No missing core fields
- US: **1,323,633 (59.979%)**
- India: **883,188 (40.021%)**
- Mean business-name length: **24.03**
- Mean address length: **52.07**

S1 is the clean/reference source.

---

## 5. Source 2 EDA

Train S2:

- **5,034,616 records**
- Address missing: **168,967 (3.356%)**
- US: **3,016,817 (59.921%)**
- India: **2,017,799 (40.079%)**
- Mean name length: **25.10**
- Mean address length: **46.23**

---

## 6. Source 3 EDA

Train S3:

- **5,285,603 records**
- Address missing: **175,916 (3.328%)**
- US: **3,170,056 (59.975%)**
- India: **2,115,547 (40.025%)**
- Mean name length: **25.20**
- Mean address length: **46.71**

---

## 7. Test Countries and Open-Set Requirement

The test set contains:

| Country | Test S1 | Test S2 | Test S3 |
|---|---:|---:|---:|
| India | 46.751% | 47.318% | 47.321% |
| US | 38.274% | 38.290% | 38.284% |
| France | 14.975% | 14.392% | 14.395% |

France is unseen in training.

Therefore:

**Country must be handled as an open-set attribute.**

Country equality can be used as a useful constraint, but the implementation must not assume only US and India exist.

---

## 8. Ground-Truth Match Distribution

There are:

**7,638,365 positive links**

across **2,206,821 S1 entities**.

Total matches per S1:

| Matches | S1 entities | Percentage |
|---:|---:|---:|
| 0 | 123,247 | 5.585% |
| 1 | 119,157 | 5.399% |
| 2 | 375,212 | 17.002% |
| 3 | 530,841 | 24.055% |
| 4 | 484,115 | 21.937% |
| 5 | 321,957 | 14.589% |
| 6 | 164,868 | 7.471% |
| 7 | 63,968 | 2.899% |
| 8 | 18,680 | 0.846% |
| 9 | 4,205 | 0.191% |
| 10 | 534 | 0.024% |
| 11 | 37 | 0.002% |

Key conclusion:

**94.4% of S1 entities have at least one match, but multiple matches are the norm.**

The most common S1 has three total matches.

---

## 9. S2/S3 Match Distribution

### S2 matches per S1

| S2 matches | S1 entities | Percentage |
|---:|---:|---:|
| 0 | 287,745 | 13.039% |
| 1 | 789,108 | 35.758% |
| 2 | 652,779 | 29.580% |
| 3 | 333,957 | 15.133% |
| 4 | 119,078 | 5.396% |
| 5 | 24,154 | 1.095% |

### S3 matches per S1

| S3 matches | S1 entities | Percentage |
|---:|---:|---:|
| 0 | 266,276 | 12.066% |
| 1 | 716,417 | 32.464% |
| 2 | 668,375 | 30.287% |
| 3 | 372,443 | 16.877% |
| 4 | 145,116 | 6.576% |
| 5 | 35,378 | 1.603% |
| 6 | 2,816 | 0.128% |

This rules out a top-1 matching strategy.

---

## 10. Evaluation Metric

The competition uses macro-averaged **F0.5 per S1**:

```text
F0.5 =
(1.25 × Precision × Recall)
/
(0.25 × Precision + Recall)
```

The metric is precision-heavy.

The final model therefore needs both:

- high candidate recall, because missed candidates cannot be recovered;
- conservative final decisions, because false matches are costly.

Singletons are important: if an S1 truly has no match, an empty prediction is correct, while an added false match is harmful.

---

## 11. Normalization

Initial normalization:

```python
text = str(text).strip().lower()
text = unicodedata.normalize("NFKC", text)
text = re.sub(r"[^\w\s]", " ", text)
text = re.sub(r"\s+", " ", text)
```

This handles case, Unicode compatibility, punctuation and whitespace variation.

Normalization is useful for blocking but cannot resolve semantic or structural address variation.

---

## 12. Exact Matching Results

### S2

| Block | Candidates | True positives | Precision | Recall |
|---|---:|---:|---:|---:|
| Exact name | 10,346,281 | 792,009 | 7.6550% | 21.4426% |
| Exact address | 564,294 | 460,928 | 81.6822% | 12.4790% |
| Exact name + address | 99,772 | 99,772 | 100% | 2.7012% |

### S3

| Block | Candidates | True positives | Precision | Recall |
|---|---:|---:|---:|---:|
| Exact name | 11,414,330 | 876,784 | 7.6814% | 22.2266% |
| Exact address | 205,509 | 170,441 | 82.9360% | 4.3207% |
| Exact name + address | 211 | 211 | 100% | 0.0053% |

### Interpretation

**Exact name:** useful recall, poor precision because generic names collide.

**Exact address:** strong precision, poor recall because addresses are noisy/missing.

**Exact name + address:** nearly perfect precision but very low recall.

These are therefore blocking/signaling features, not sufficient final rules.

---

## 13. Country-Conditioned Blocking

Adding country to exact name/address blocks changed very little in recall.

This indicates that true matches generally already share country.

Country is therefore useful for:

- eliminating impossible comparisons;
- open-set handling;
- feature engineering;

but it is not a powerful discriminator by itself.

---

## 14. Name ∪ Address Blocking

The union of exact normalized name and address produced:

### S2

- Candidates: **10,810,803**
- True positives: **1,153,165**
- Precision: **10.6668%**
- Recall: **31.2205%**

### S3

- Candidates: **11,619,628**
- True positives: **1,047,014**
- Precision: **9.0107%**
- Recall: **26.5420%**

Combining complementary blocks improves recall, but still leaves too many true matches unretrieved.

---

## 15. Structural Components

The next analysis extracted:

### House number

First numeric component in the address.

Example:

```text
123 Main Street
→ 123
```

### Name prefix

First four characters of normalized name after removing spaces.

### Strong address tokens

Address tokens with length >= 4, excluding generic terms such as:

```text
road
street
avenue
lane
drive
building
floor
unit
...
```

The goal is to find stable structural clues that survive noisy formatting.

---

## 16. Structural Blocking Results

### A — country + number + exact name

Previously measured:

**S2**

- Candidates: 561,363
- True positives: 549,214
- Precision: **97.8358%**
- Recall: **14.8693%**

**S3**

- Candidates: 618,944
- True positives: 599,290
- Precision: **96.8246%**
- Recall: **15.1921%**

This is an excellent high-confidence signal.

---

### B — country + number + name prefix

**S2**

- Candidates: 21,552,441
- True positives: 1,981,730
- Precision: 9.1949%
- Recall: **53.6528%**

**S3**

- Candidates: 28,209,452
- True positives: 2,053,964
- Precision: 7.2811%
- Recall: **52.0683%**

This is a strong recall-oriented block.

---

### C — country + address token + name prefix

**S2**

- Candidates: 84,891,162
- True positives: 2,520,049
- Precision: 2.9686%
- Recall: **68.2271%**

**S3**

- Candidates: 118,341,441
- True positives: 2,463,304
- Precision: 2.0815%
- Recall: **62.4452%**

This is useful for recall but expensive.

---

### country + number + address token

**S2**

- Candidates: 538,662,493
- True positives: 2,306,738
- Precision: 0.4282%
- Recall: **62.4520%**

**S3**

- Candidates: 534,997,494
- True positives: 2,366,564
- Precision: 0.4424%
- Recall: **59.9928%**

This is extremely broad and appears computationally inefficient.

---

### country + name + first number

**S2**

- Candidates: 15,350,114
- True positives: 1,861,582
- Precision: 12.1275%
- Recall: **50.3999%**

**S3**

- Candidates: 20,571,317
- True positives: 1,931,388
- Precision: 9.3887%
- Recall: **48.9610%**

This is another useful moderate-width block.

---

## 17. All-Component Union

The previous union of all tested component blocks produced:

### S2

- Candidates: **649,348,169**
- True positives: **3,170,282**
- Recall: **85.8313%**

### S3

- Candidates: **686,464,618**
- True positives: **3,384,021**
- Recall: **85.7855%**

Combined:

```text
~1.336 billion candidate pairs
~85.8% candidate recall
```

This demonstrates that deterministic structural blocking can recover most matches, but the candidate volume is too large for naive downstream pairwise inference.

---

## 18. Latest Marginal Experiment

The current experiment tests cumulative combinations:

```text
A
A + B
A + B + C
A + B + C + D
```

where:

### A

```text
country + house number + exact normalized name
```

### B

```text
country + house number + name prefix
```

### C

```text
country + address token + name prefix
```

### D

```text
country + exact normalized address
```

Latest observed individual results:

### A

```text
Candidates : 1,145,366
True pairs : 1,121,539
Precision  : 97.919704%
Recall     : 14.682972%
```

### B

```text
Candidates : 33,411,859
True pairs : 3,894,694
Precision  : 11.656622%
Recall     : 50.988582%
```

A and B here are being evaluated as combined S2+S3 candidate sets.

### Interpretation

A is extremely selective and high precision.

B produces roughly:

```text
3.47× as many true pairs
```

as A, but approximately:

```text
29× as many candidates
```

Therefore:

- A is a strong high-confidence block.
- B is primarily a high-recall candidate generator.
- The cumulative A+B result is more important than B alone because overlap between the blocks must be measured.

The next required output is the marginal contribution of C and D.

---

## 19. Candidate Generation Strategy

The likely final candidate system should contain multiple complementary blocks rather than relying on one.

A promising hierarchy is:

### High-confidence block

```text
country
+ house number
+ exact name
```

### Medium-width block

```text
country
+ house number
+ name prefix
```

### Selective recall fallback

```text
country
+ address token
+ name prefix
```

Additional blocks may include:

```text
exact normalized name
exact normalized address
country + first name token + number
postal/ZIP/PIN + name component
```

The final selection should be driven by marginal recall per candidate cost.

---

## 20. Why Approximate Retrieval Is Needed

Even the union of many deterministic blocks previously reached only about:

```text
85.8% recall
```

Therefore approximately 14% of true pairs can remain outside the deterministic candidate set.

These hard matches are likely caused by:

- spelling changes;
- abbreviations;
- address formatting;
- token reordering;
- missing address;
- punctuation;
- transliteration/Unicode variation;
- more substantial source-specific noise.

The next retrieval layer should investigate approximate methods.

---

## 21. Character N-Gram Retrieval

A promising approach is character n-gram TF-IDF retrieval.

Character n-grams are useful because they are robust to:

- small spelling errors;
- punctuation changes;
- whitespace differences;
- abbreviations;
- partial token overlap.

Potentially build separate retrieval indexes for:

```text
business name
address
name + address
```

Then retrieve only the top-K approximate candidates per S1.

This can recover hard pairs without constructing hundreds of millions of broad deterministic candidates.

---

## 22. Pairwise Feature Engineering

After candidate generation, compute features such as:

### Name

```text
exact normalized match
Levenshtein similarity
Jaro-Winkler similarity
character n-gram cosine
token Jaccard
token overlap
prefix agreement
length difference
```

### Address

```text
exact normalized match
character similarity
token Jaccard
house-number agreement
postal/ZIP/PIN agreement
numeric token overlap
length difference
```

### Metadata

```text
country match
source indicator
address missingness
name/address lengths
```

Interactions are also useful:

```text
name similarity × address similarity
house number match × name similarity
exact name + partial address
```

---

## 23. Pairwise ML Model

A tree-based model is a strong starting point:

- LightGBM
- XGBoost
- CatBoost

The training data should contain:

```text
positive = ground-truth matched pair
negative = candidate pair not present in ground truth
```

The negatives should come from actual blocking/retrieval output.

This creates **hard negatives**, which are far more informative than random unrelated businesses.

---

## 24. Avoid Random Pair-Level Splitting

A random split of pairs can create leakage.

A better validation design is:

```text
split S1 entities
        ↓
train S1 entities
validation S1 entities
```

All candidate pairs belonging to an S1 should stay in the same partition.

The validation procedure should reproduce the competition's macro F0.5 evaluation.

---

## 25. Final Matching Must Allow Multiple Matches

Do not use:

```text
top_1(candidate)
```

because the ground truth frequently contains multiple matches.

Instead:

```text
score all candidates
keep candidates above calibrated threshold
```

The threshold should be selected using the actual validation metric.

Potentially separate thresholds can be investigated for:

- S2/S3;
- deterministic vs approximate candidates;
- records with/without addresses.

But thresholds should be learned from validation results rather than manually chosen.

---



## 27. Why an LLM Should Not Be the Default Matcher

The dataset scale is too large for LLM inference over arbitrary pairs.

The sensible hierarchy is:

```text
normalization
 ↓
deterministic blocking
 ↓
approximate retrieval
 ↓
compact pairwise features
 ↓
tree-based matcher
 ↓
threshold calibration
```

A language model, if used at all, should be restricted to a small hard-case subset rather than the full candidate universe.

---

## 28. Important Risks

### Candidate recall ceiling

Missed candidates can never be recovered.

### Candidate explosion

High recall can create hundreds of millions of candidates.

### Easy-negative bias

Random negatives can make a classifier appear much better than it is.

### Top-1 assumption

Incorrect because multiple matches are common.

### Closed-set country encoding

France is unseen during training and must still work.

### External data

The challenge prohibits external lookup/augmentation. The solution should rely only on supplied data.

---

## 29. Recommended Next Steps

### Step 1
Finish:

```text
A
A+B
A+B+C
A+B+C+D
```

and inspect the **new true pairs** added by each block.

### Step 2
Build an optimized integer/array-based candidate generator.

### Step 3
Investigate character n-gram approximate retrieval to recover the remaining hard matches.

### Step 4
Construct positive and hard-negative training pairs.

### Step 5
Train LightGBM/XGBoost/CatBoost.

### Step 6
Tune the decision threshold for **macro F0.5 per S1**.

### Step 7
Evaluate separately for US, India and France.

### Step 8
Generate:

```text
matching_results.tsv
candidate_pairs.tsv
```

with final candidate-set consistency checks.

---

## 30. Overall Conclusion

The dataset analysis indicates that this challenge is best viewed as a **retrieval + ranking entity-resolution problem**.

No individual field is sufficient:

```text
exact name
    → broad but noisy

exact address
    → precise but incomplete

name + address
    → very precise but low recall

country
    → useful constraint but weak discriminator

house number
    → strong structural signal

name prefix
    → strong recall expansion

address token
    → additional recall but high candidate cost
```

The strongest current design is therefore:

```text
                    S1 REFERENCE
                         │
                         ▼
                  NORMALIZATION
                         │
                         ▼
              MULTI-BLOCK RETRIEVAL
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          Exact       Structural   Approximate
          blocks       blocks       retrieval
             └───────────┼───────────┘
                         ▼
                  CANDIDATE PAIRS
                         │
                         ▼
               SIMILARITY FEATURES
                         │
                         ▼
                 PAIRWISE ML MODEL
                         │
                         ▼
                THRESHOLD CALIBRATION
                         │
                         ▼
                MULTIPLE MATCHES/S1
```

The most important current engineering target is to improve candidate recall beyond the approximately **85.8% deterministic level** while keeping candidate volume manageable.

The marginal A/B/C/D experiment is therefore the correct immediate next step before committing to the final retrieval architecture.
