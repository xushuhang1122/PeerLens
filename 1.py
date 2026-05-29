import sys
sys.path.insert(0, ".")
from src.paperradar.store.chroma import ChromaManager
from src.paperradar.store.bm25_index import BM25Index
from src.paperradar.schemas.paper import Paper

print("Loading papers from ChromaDB...")
chroma = ChromaManager()
col = chroma.content_collection
total = col.count()
print("Total papers:", total)

results = col.get(limit=total, include=["metadatas"])
papers = []
for i, meta in enumerate(results["metadatas"]):
    try:
        papers.append(Paper(
            id=results["ids"][i],
            title=meta.get("title", ""),
            abstract=meta.get("abstract", ""),
            keywords=meta.get("keywords", "").split(",") if meta.get("keywords") else [],
        ))
    except Exception:
        pass

print("Building BM25 from", len(papers), "papers...")
BM25Index().build(papers)
print("Done.")