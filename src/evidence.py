import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)


def extract_evidence(question, retrieved_chunks, sentence_ranker, top_k=5, min_words=5):
    """
    Extract the most relevant evidence sentences from retrieved chunks.

    Input:
    - question: user's question
    - retrieved_chunks: top chunks selected by chunk reranker
    - sentence_ranker: CrossEncoder model
    - top_k: number of evidence sentences to return

    Output:
    - list of evidence sentence dictionaries
    """

    all_sentences = []

    for chunk in retrieved_chunks:
        text = chunk.get("text", "")
        page = chunk.get("page", "Unknown")
        chunk_id = chunk.get("chunk_id", chunk.get("id", "Unknown"))

        sentences = sent_tokenize(text)

        for sentence in sentences:
            sentence = sentence.strip()

            if len(sentence.split()) < min_words:
                continue

            all_sentences.append({
                "text": sentence,
                "page": page,
                "chunk_id": chunk_id
            })

    if len(all_sentences) == 0:
        return []

    pairs = [
        (question, item["text"])
        for item in all_sentences
    ]

    scores = sentence_ranker.predict(pairs)

    ranked = sorted(
        zip(all_sentences, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_evidence = []

    for item, score in ranked[:top_k]:
        top_evidence.append({
            "text": item["text"],
            "page": item["page"],
            "chunk_id": item["chunk_id"],
            "score": float(score)
        })

    return top_evidence