import re
from pathlib import Path
from typing import Dict, List

import chromadb


def parse_qa_pairs(file_path: str) -> List[Dict[str, str]]:
    """Parse support ticket Q&A entries from a text file."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    pairs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
    qa_pairs: List[Dict[str, str]] = []

    for idx, pair in enumerate(pairs):
        question = ""
        answer = ""

        for line in pair.splitlines():
            if line.startswith("Question:"):
                question = line.replace("Question:", "", 1).strip()
            elif line.startswith("Answer:"):
                answer = line.replace("Answer:", "", 1).strip()

        if question and answer:
            qa_pairs.append({"question": question, "answer": answer, "id": f"hr_qa_{idx}"})

    return qa_pairs


def prepare_hr_support_documents(file_path: str) -> Dict[str, List[str]]:
    """Convert HR Q&A entries into ChromaDB-ready document payloads."""
    qa_pairs = parse_qa_pairs(file_path)

    documents = []
    metadatas = []
    ids = []

    for qa in qa_pairs:
        document_text = (
            f"Question: {qa['question']}\n"
            f"Answer: {qa['answer']}\n\n"
            "This support ticket entry provides HR guidance for employees."
        )

        keywords = re.findall(r"\b\w+\b", (qa["question"] + " " + qa["answer"]).lower())

        metadata = {
            "question": qa["question"],
            "answer": qa["answer"],
            "question_length": len(qa["question"]),
            "answer_length": len(qa["answer"]),
            "keywords": " ".join(sorted(set(keywords))),
            "topic": "hr_support_ticket",
        }

        documents.append(document_text)
        metadatas.append(metadata)
        ids.append(qa["id"])

    return {"documents": documents, "metadatas": metadatas, "ids": ids}


def setup_support_ticket_chromadb(
    file_path: str,
    collection_name: str = "hr_support_qna",
) -> chromadb.Collection:
    """Create and populate a ChromaDB collection from the HR support ticket file."""
    repo_root = Path(__file__).resolve().parents[1]
    chroma_path = repo_root / "chroma"
    client = chromadb.PersistentClient(str(chroma_path))

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "HR support ticket Q&A knowledge base"},
    )

    data = prepare_hr_support_documents(file_path)
    collection.add(
        documents=data["documents"],
        metadatas=data["metadatas"],
        ids=data["ids"],
    )

    print(f"Added {len(data['documents'])} HR support Q&A entries to '{collection_name}'")
    return collection


def query_support_ticket(
    question: str,
    collection_name: str = "hr_support_qna",
    n_results: int = 3,
):
    """Query the HR support knowledge base."""
    repo_root = Path(__file__).resolve().parents[1]
    chroma_path = repo_root / "chroma"
    client = chromadb.PersistentClient(str(chroma_path))
    collection = client.get_collection(name=collection_name)

    return collection.query(query_texts=[question], n_results=n_results)


if __name__ == "__main__":
    support_file = Path(__file__).resolve().parents[1] / "data" / "support_ticket.txt"
    setup_support_ticket_chromadb(str(support_file), collection_name="hr_support_qna")
