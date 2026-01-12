import csv
import os
from typing import List
from dotenv import load_dotenv
from neo4j import GraphDatabase

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain.agents import create_agent

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_neo4j import Neo4jVector

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASS)
)

def ingest_faq_csv(path: str):
    with open(path, encoding="utf-8") as f, driver.session() as session:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question"].strip()
            a = row["answer"].strip()
            vector = embeddings.embed_query(q + " " + a)

            session.run(
                """
                MERGE (q:Question {text: $q})
                MERGE (a:Answer {text: $a})
                MERGE (q)-[:HAS_ANSWER]->(a)
                SET q.embedding = $vector
                """,
                q=q,
                a=a,
                vector=vector
            )

ingest_faq_csv("./lauki_qna.csv")

vector_store = Neo4jVector(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASS,
    index_name="faq_index",
    node_label="Question",
    text_node_property="text",
    embedding_node_property="embedding",
    embedding=embeddings
)

@tool
def search_faq(query: str) -> str:
    """
    Semantic FAQ search using Neo4j vector index.
    """
    results = vector_store.similarity_search(query, k=3)

    if not results:
        return "No relevant FAQ entries found."

    return "\n\n---\n\n".join(
        f"Q: {doc.page_content}" for doc in results
    )


@tool
def graph_faq_lookup(keyword: str) -> str:
    """
    Structured graph lookup using Cypher.
    Use when the query contains specific terms.
    """
    with driver.session() as session:
        result = session.run(
            """
            MATCH (q:Question)-[:HAS_ANSWER]->(a:Answer)
            WHERE q.text CONTAINS $keyword
            RETURN q.text AS question, a.text AS answer
            LIMIT 3
            """,
            keyword=keyword
        )

        records = result.data()
        if not records:
            return "No matching FAQs found."

        return "\n\n".join(
            f"Q: {r['question']}\nA: {r['answer']}"
            for r in records
        )

tools = [search_faq, graph_faq_lookup]

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

system_prompt = """
You are a GraphRAG-powered FAQ assistant.

Rules:
- Use semantic search for vague or general questions
- Use graph lookup for specific keywords or terms
- Combine answers if helpful
- Never hallucinate
- Answer only from retrieved data
"""

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt
)

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [("human", "Explain roaming activation.")]}
    )
    print(result["messages"][-1].content)
