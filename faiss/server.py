import os
import threading
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import faiss
from fastapi import FastAPI, HTTPException, Request
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from langchain_cohere import CohereRerank
from pydantic import BaseModel
import cohere

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")

RERANK_MODEL = os.getenv(key="RERANK_MODEL", default="")
RERANK_HOST = os.getenv(key="RERANK_HOST", default="")
RERANK_API_KEY = os.getenv(key="RERANK_API_KEY", default="")

RERANK_THRESHOLD: float = float(os.getenv(key="RERANK_THRESHOLD", default = 0.7))

# --- Request/Response models ---


class DocumentInput(BaseModel):
    content: str
    metadata: dict[str, Any] = {}


class CreateDocumentsRequest(BaseModel):
    documents: list[DocumentInput]


class CreateDocumentsResponse(BaseModel):
    ids: list[str]


class DocumentResponse(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any]


class ListDocumentsResponse(BaseModel):
    documents: list[DocumentResponse]


class UpdateDocumentRequest(BaseModel):
    content: str | None = None
    metadata: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    query: str
    k: int = 4
    rerank: bool = False
    filter: dict[str, Any] | None = None


class SearchResult(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any]
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class SyncDocumentInput(BaseModel):
    source_id: str
    content: str
    metadata: dict[str, Any] = {}


class SyncRequest(BaseModel):
    documents: list[SyncDocumentInput]
    delete_missing: bool = True


class SyncResponse(BaseModel):
    added: list[str]
    updated: list[str]
    deleted: list[str]
    unchanged: list[str]


# --- Vector store manager ---


class VectorStoreManager:
    def __init__(self, index_path: str, embedding_model: str, rerank_model: str = "", rerank_host: str = "", rerank_api_key: str = "", rerank_threshold: float = 0.7):
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        self.rerank_threshold = rerank_threshold

        if rerank_model and rerank_host:
            co_cli = cohere.ClientV2(base_url=rerank_host, api_key=rerank_api_key)
            self.reranker = CohereRerank(model = rerank_model, client=co_cli)
        else:
            self.reranker = None
        self.vector_store = self._load_or_create()
        self._lock = threading.Lock()

    def _load_or_create(self) -> FAISS:
        try:
            return FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            dimension = len(self.embeddings.embed_query("dimension probe"))
            index = faiss.IndexFlatL2(dimension)
            return FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

    def _save(self) -> None:
        self.vector_store.save_local(self.index_path)

    def _get_doc(self, doc_id: str) -> Document:
        doc = self.vector_store.docstore.search(doc_id)
        if not isinstance(doc, Document):
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        return doc

    def add_documents(self, inputs: list[DocumentInput]) -> list[str]:
        if not inputs:
            raise HTTPException(status_code=400, detail="No documents provided")
        docs = [
            Document(page_content=d.content, metadata=d.metadata) for d in inputs
        ]
        ids = [str(uuid4()) for _ in docs]
        with self._lock:
            self.vector_store.add_documents(docs, ids=ids)
            self._save()
        return ids

    def get_document(self, doc_id: str) -> DocumentResponse:
        doc = self._get_doc(doc_id)
        return DocumentResponse(
            id=doc_id, content=doc.page_content, metadata=doc.metadata
        )

    def list_documents(self) -> list[DocumentResponse]:
        results = []
        for doc_id in self.vector_store.index_to_docstore_id.values():
            doc = self.vector_store.docstore.search(doc_id)
            if isinstance(doc, Document):
                results.append(
                    DocumentResponse(
                        id=doc_id, content=doc.page_content, metadata=doc.metadata
                    )
                )
        return results

    def update_document(
        self, doc_id: str, content: str | None, metadata: dict[str, Any] | None
    ) -> DocumentResponse:
        if content is None and metadata is None:
            raise HTTPException(
                status_code=400, detail="Provide content or metadata to update"
            )
        existing = self._get_doc(doc_id)
        new_content = content if content is not None else existing.page_content
        new_metadata = metadata if metadata is not None else existing.metadata
        new_doc = Document(page_content=new_content, metadata=new_metadata)
        with self._lock:
            self.vector_store.delete(ids=[doc_id])
            self.vector_store.add_documents([new_doc], ids=[doc_id])
            self._save()
        return DocumentResponse(id=doc_id, content=new_content, metadata=new_metadata)

    def delete_document(self, doc_id: str) -> None:
        self._get_doc(doc_id)
        with self._lock:
            self.vector_store.delete(ids=[doc_id])
            self._save()

    def search(
        self, query: str, k: int, filter: dict[str, Any] | None, rerank: bool = False
    ) -> list[SearchResult]:
        if k < 1:
            raise HTTPException(status_code=400, detail="k must be at least 1")

        results = self.vector_store.similarity_search_with_score(
            query, k=k, filter=filter or None
        )

        if rerank:
            if not self.reranker:
                raise HTTPException(status_code=400, detail="Rerank requested, but server not initialized with reranking enabled")
            documents = [doc for doc, _ in results]
            ranked = self.reranker.rerank(documents=documents, query=query, max_tokens_per_doc = 10000)
            results = [(results[r["index"]][0], r["relevance_score"]) for r in ranked if r["relevance_score"] > self.rerank_threshold]


        doc_to_id = {}
        for doc_id in self.vector_store.index_to_docstore_id.values():
            stored = self.vector_store.docstore.search(doc_id)
            if isinstance(stored, Document):
                doc_to_id[id(stored)] = doc_id

        return [
            SearchResult(
                id=doc_to_id.get(id(doc), ""),
                content=doc.page_content,
                metadata=doc.metadata,
                score=float(score),
            )
            for doc, score in results
        ]

    def sync_documents(
        self, inputs: list[SyncDocumentInput], delete_missing: bool
    ) -> SyncResponse:
        existing: dict[str, tuple[str, str, dict[str, Any]]] = {}
        for doc_id in self.vector_store.index_to_docstore_id.values():
            doc = self.vector_store.docstore.search(doc_id)
            if isinstance(doc, Document) and "source_id" in doc.metadata:
                existing[doc.metadata["source_id"]] = (
                    doc_id,
                    doc.page_content,
                    doc.metadata,
                )

        incoming_source_ids = {d.source_id for d in inputs}
        added, updated, unchanged = [], [], []
        to_delete_ids = []
        to_add_docs: list[tuple[Document, str]] = []

        for inp in inputs:
            full_meta = {**inp.metadata, "source_id": inp.source_id}
            if inp.source_id not in existing:
                new_id = str(uuid4())
                to_add_docs.append(
                    (Document(page_content=inp.content, metadata=full_meta), new_id)
                )
                added.append(inp.source_id)
            else:
                old_id, old_content, old_meta = existing[inp.source_id]
                if old_content != inp.content or old_meta != full_meta:
                    to_delete_ids.append(old_id)
                    to_add_docs.append(
                        (
                            Document(page_content=inp.content, metadata=full_meta),
                            old_id,
                        )
                    )
                    updated.append(inp.source_id)
                else:
                    unchanged.append(inp.source_id)

        deleted = []
        if delete_missing:
            for source_id, (doc_id, _, _) in existing.items():
                if source_id not in incoming_source_ids:
                    to_delete_ids.append(doc_id)
                    deleted.append(source_id)

        with self._lock:
            if to_delete_ids:
                self.vector_store.delete(ids=to_delete_ids)
            if to_add_docs:
                docs = [d for d, _ in to_add_docs]
                ids = [i for _, i in to_add_docs]
                self.vector_store.add_documents(docs, ids=ids)
            if to_delete_ids or to_add_docs:
                self._save()

        return SyncResponse(
            added=added, updated=updated, deleted=deleted, unchanged=unchanged
        )


# --- FastAPI app ---


def get_store(request: Request) -> VectorStoreManager:
    return request.app.state.store


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = VectorStoreManager(
        index_path=FAISS_INDEX_PATH, embedding_model=EMBEDDING_MODEL,
        rerank_host=RERANK_HOST, rerank_model=RERANK_MODEL, rerank_api_key=RERANK_API_KEY,
        rerank_threshold=RERANK_THRESHOLD
    )
    yield


app = FastAPI(title="FAISS Vector Store API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", response_model=CreateDocumentsResponse)
def create_documents(body: CreateDocumentsRequest, request: Request):
    ids = get_store(request).add_documents(body.documents)
    return CreateDocumentsResponse(ids=ids)


@app.get("/documents", response_model=ListDocumentsResponse)
def list_documents(request: Request):
    docs = get_store(request).list_documents()
    return ListDocumentsResponse(documents=docs)


@app.get("/documents/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: str, request: Request):
    return get_store(request).get_document(doc_id)


@app.put("/documents/{doc_id}", response_model=DocumentResponse)
def update_document(doc_id: str, body: UpdateDocumentRequest, request: Request):
    return get_store(request).update_document(doc_id, body.content, body.metadata)


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str, request: Request):
    get_store(request).delete_document(doc_id)
    return {"deleted": True}


@app.post("/search", response_model=SearchResponse)
def search_documents(body: SearchRequest, request: Request):
    results = get_store(request).search(body.query, body.k, body.filter, rerank = body.rerank)
    return SearchResponse(results=results)


@app.post("/sync", response_model=SyncResponse)
def sync_documents(body: SyncRequest, request: Request):
    return get_store(request).sync_documents(body.documents, body.delete_missing)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
