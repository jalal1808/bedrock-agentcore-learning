import pandas as pd
import json
import time

from rag import rag_agent
from simple_rag import 

def run_benchmark(agent, name):
    with open("benchmark_questions.json") as f:
        questions = json.load(f)

    results = []

    for q in questions:
        start = time.time()
        response = agent.invoke(
            {"messages": [("human", q["question"])]}
        )
        latency = time.time() - start

        answer = response["messages"][-1].content.lower()

        score = sum(
            1 for kw in q["expected_keywords"]
            if kw.lower() in answer
        ) / len(q["expected_keywords"])

        results.append({
            "question": q["question"],
            "type": q["type"],
            "score": score,
            "latency": latency
        })

    return results

rag_results = run_benchmark(rag, "Simple RAG")
graphrag_results = run_benchmark(simple_rag, "GraphRAG")



df_rag = pd.DataFrame(rag_results)
df_graph = pd.DataFrame(graphrag_results)

print("Simple RAG Accuracy:", df_rag["score"].mean())
print("GraphRAG Accuracy:", df_graph["score"].mean())

print("\nGraph Queries Only:")
print(df_graph[df_graph["type"] == "graph"][["question", "score"]])
