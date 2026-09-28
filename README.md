## Evaluation

The RAG system was evaluated at both the retrieval and generation levels.

### Retrieval Evaluation

Retrieval performance was measured using Precision@5, Recall@5, and Mean Reciprocal Rank (MRR).

| Retrieval Method | Precision@5 | Recall@5 | MRR |
|---|---:|---:|---:|
| Dense Retrieval | 0.1300 | 0.5667 | 0.6140 |
| BM25 | 0.2000 | 0.8500 | 0.7280 |
| Hybrid Retrieval | 0.2100 | 0.9000 | 0.7458 |
| Hybrid + MMR | 0.2100 | 0.9000 | 0.7458 |

The evaluation indicates that combining semantic and lexical retrieval improved retrieval performance compared with dense retrieval alone.

### Generation Evaluation

Generation evaluation used answerable and unanswerable research questions to assess grounded response generation and citation behavior.

| Evaluation Result | Value |
|---|---:|
| Total queries | 8 |
| Successful generations | 3 |
| API failures | 5 |
| Answerable queries | 5 |
| Unanswerable queries | 3 |
| Expected-page retrieval coverage | 1.0000 |
| Citation presence accuracy* | 1.0000 |

\*Citation presence accuracy was calculated for successful answerable generations only.

The generation evaluation was partially limited by temporary Gemini API availability issues. Therefore, the results above should not be interpreted as an overall generation-quality score.

### Faithfulness Evaluation

An LLM-as-a-judge evaluation was designed to assess:

- Faithfulness
- Citation correctness
- Answer relevance

The evaluation separates generation failures from evaluator failures. Faithfulness scores are not reported where API availability prevented reliable evaluation.