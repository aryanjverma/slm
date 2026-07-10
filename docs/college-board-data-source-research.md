# College Board sources for an APUSH LEQ grader dataset

_Research date: July 10, 2026. Factual claims below use only first-party College Board sources._

## Bottom line

- The best public, consistently packaged corpus is **2023–2025**: AP Central currently publishes two exam sets per year, with LEQ 2, 3, and 4 sample-response packets for each set. That is **18 LEQ packets and 54 scored student responses** (each packet contains three samples) with prompt, scoring guideline, row-level scores, total score, and scoring commentary. [APUSH past exam questions](https://apcentral.collegeboard.org/courses/ap-united-states-history/exam/past-exam-questions) and a representative [2025 LEQ 2 Set 1 packet](https://apcentral.collegeboard.org/media/pdf/ap25-apc-us-history-leq2-set-1.pdf).
- Keep **2023** separate from **2024–2025** by rubric version. The 2024 LEQ complexity point broadened the stated pathways from the 2023 formulation to “sophisticated argumentation and/or effective use of evidence,” including multiple themes/perspectives and effective use of at least four pieces of evidence. Compare the official [2023 scoring guidelines](https://apcentral.collegeboard.org/media/pdf/ap23-sg-us-history-set-1.pdf) with the [2024 scoring guidelines](https://apcentral.collegeboard.org/media/pdf/ap24-sg-us-history-set-1.pdf); the [2025 guidelines](https://apcentral.collegeboard.org/media/pdf/ap25-sg-us-history-set-1.pdf) retain the 2024 structure.
- Public access is **not a dataset or model-training license**. College Board's educator terms prohibit downloading, modifying, reusing, reproducing, scraping, or data-mining its content without express written permission and direct users to a permission form. Obtain written permission before creating or distributing a training dataset. [Educator Legal Terms](https://privacy.collegeboard.org/educator-legal-terms); [Permission Request Form](https://privacy.collegeboard.org/copyright-trademark/request-form).

## What the official packets contain

- **Free-response question PDFs:** the released question wording and exam set. For 2023–2025, each set includes three LEQ choices (questions 2–4). Examples: [2025 Set 1](https://apcentral.collegeboard.org/media/pdf/ap25-frq-us-history-set-1.pdf) and [2025 Set 2](https://apcentral.collegeboard.org/media/pdf/ap25-frq-us-history-set-2.pdf).
- **Scoring guidelines:** the six-point LEQ rubric plus question-specific acceptable thesis, context, evidence, reasoning, and complexity examples. Examples: [2023 Set 1](https://apcentral.collegeboard.org/media/pdf/ap23-sg-us-history-set-1.pdf), [2024 Set 1](https://apcentral.collegeboard.org/media/pdf/ap24-sg-us-history-set-1.pdf), and [2025 Set 1](https://apcentral.collegeboard.org/media/pdf/ap25-sg-us-history-set-1.pdf).
- **Sample responses and scoring commentary:** a representative 2025 packet supplies complete Samples A, B, and C, row scores for thesis, contextualization, evidence, and analysis/reasoning, total scores of 6, 4, and 2, and prose explaining each awarded or missed point. [2025 LEQ 2 Set 1](https://apcentral.collegeboard.org/media/pdf/ap25-apc-us-history-leq2-set-1.pdf).
- **Supporting annual files:** the archive also links Chief Reader reports, scoring statistics, and score distributions. These are useful for calibration context, but they do not add essay-level labels. [APUSH past exam questions](https://apcentral.collegeboard.org/courses/ap-united-states-history/exam/past-exam-questions).

## Availability by year

| Year | Officially observable availability as of research date | Dataset use |
|---|---|---|
| 2026 | The archive lists the [released FRQ PDF](https://apcentral.collegeboard.org/media/pdf/ap26-frq-us-history.pdf), but no scoring guidelines or sample/commentary packets yet. | Prompts only; no gold labels yet. |
| 2025 | Two FRQ sets, two scoring guides, and sample/commentary packets for LEQ 2–4 in both sets. | 18 labeled responses; strongest current source. |
| 2024 | Same complete two-set structure as 2025. | 18 labeled responses; same relevant LEQ rubric language as 2025. |
| 2023 | Same complete two-set structure. | 18 labeled responses; tag as the pre-2024 complexity rubric. |
| 2022, 2021, 2019, 2018 | Legacy FRQ, scoring-guide, and LEQ sample PDFs still resolve on College Board's domain (for example [2022 FRQ](https://apcentral.collegeboard.org/media/pdf/ap22-frq-us-history.pdf), [2022 scoring guidelines](https://apcentral.collegeboard.org/media/pdf/ap22-sg-us-history.pdf), [2021 LEQ 2 samples](https://apcentral.collegeboard.org/media/pdf/ap21-apc-us-history-leq2.pdf), [2019 LEQ 4 samples](https://apcentral.collegeboard.org/media/pdf/ap19-apc-us-history-leq4.pdf), and [2018 LEQ 2 samples](https://apcentral.collegeboard.org/media/pdf/ap18-apc-us-history-leq2.pdf)). They are no longer listed in the current three-year archive. | Potential expansion set, but inventory and rubric-version checks must be done file by file; legacy direct URLs should not be treated as a stable archive contract. |
| Earlier years / 2020 | No complete, current public LEQ packet inventory was established from the official archive. College Board says comprehensive prior-year questions are available to authorized educators in secure AP Classroom. | Do not claim completeness; do not extract secure AP Classroom material without permission. |

AP Central explicitly states that it provides the **three most recent years** of released materials and that authorized educators can access comprehensive, secure prior-year questions in AP Classroom. The page currently retains 2023 while 2026 has questions only, so the completed public sample corpus is 2023–2025. [APUSH past exam questions](https://apcentral.collegeboard.org/courses/ap-united-states-history/exam/past-exam-questions).

## Rubric and format boundaries

- **2023 versus 2024:** preserve the original annual scoring guide and assign a `rubric_version`. The most consequential change for an LEQ grader is the expanded 2024 description of how the second analysis/reasoning point can be earned. [2023 guide](https://apcentral.collegeboard.org/media/pdf/ap23-sg-us-history-set-1.pdf); [2024 guide](https://apcentral.collegeboard.org/media/pdf/ap24-sg-us-history-set-1.pdf).
- **2024 versus 2025:** the six-point rows and the relevant complexity language remain aligned, making these years suitable for a common rubric version after checking question-specific notes. [2024 guide](https://apcentral.collegeboard.org/media/pdf/ap24-sg-us-history-set-1.pdf); [2025 guide](https://apcentral.collegeboard.org/media/pdf/ap25-sg-us-history-set-1.pdf).
- **May 2027 format:** students will answer one broad LEQ instead of choosing among three; the question adds an orienting statement and lets students select relevant developments from a broad period. College Board says the FRQ scoring rubrics and point criteria remain unchanged. [AP History Exam Updates](https://apcentral.collegeboard.org/courses/ap-history-exam-updates); [APUSH CED, effective Fall 2026](https://apcentral.collegeboard.org/media/pdf/ap-us-history-course-and-exam-description.pdf); [CED clarifications](https://apcentral.collegeboard.org/media/pdf/ap-us-history-course-and-exam-description-clarifications.pdf).

## Recommended dataset design

1. Use 2023–2025 as the clean public core; retain `year`, `set`, `leq_number`, prompt, response, four row scores, total score, and commentary.
2. Store each year's scoring guide verbatim as provenance, but keep rubric examples and commentary out of held-out grader inputs to prevent answer leakage.
3. Use rubric labels such as `2023_leq` and `2024_2026_leq`; treat 2027 as a prompt-format shift even though College Board says scoring criteria are unchanged.
4. Add legacy years only after validating every PDF's question, rubric, sample count, and commentary layout; record the exact source URL and file hash.
5. Seek written College Board permission before automated collection, transformation, model training, publication, or redistribution.

## Licensing and usage constraints

College Board's educator terms state that its text, images, materials, and data are protected content; they prohibit distribution, downloading, modification, reuse, reproduction, reposting, scraping, and data-mining without express written permission. The terms also grant only a limited, nonexclusive, revocable, nontransferable license to access digital services and prohibit commercial exploitation outside intended use. [Educator Legal Terms](https://privacy.collegeboard.org/educator-legal-terms).

Therefore, the archive's download links establish **availability, not permission for corpus construction or model training**. The conservative path is to submit the official [Permission Request Form](https://privacy.collegeboard.org/copyright-trademark/request-form) describing extraction, storage, training/evaluation, access controls, derivative outputs, and redistribution. This is a source-based risk summary, not legal advice.
