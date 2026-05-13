
import json
import os

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import StreamWriter
from pydantic import BaseModel
from typing import Annotated, AsyncGenerator, List





class ChatRequest(BaseModel):
    message: str
    session_id: str
    retriever_mode: str


class ChatState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages]
    tool_mode: str = 'faiss'
          

class FAISSTool:

    def __init__(self, faiss_url: str, top_k: int = 4, rerank: bool = False):
        self.faiss_url = faiss_url
        self.top_k = top_k
        self.rerank = rerank

    def retrieve_cves(self, query: str) -> list[dict]:
        resp = httpx.post(
            f"{self.faiss_url}/search",
            json={"query": query, "k": self.top_k, "rerank": self.rerank},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["results"]
    
    def get_tool(self):
        instance = self

        @tool(description="Fetches CVEs related to the query")
        def search_cve(query: Annotated[str, "The query used to search for CVEs"]) -> str:
            results = instance.retrieve_cves(query)
            if not results:
                return "No relevant CVEs found."
            sections = []
            for r in results:
                meta = r.get("metadata", {})
                refs = meta.get("references", [])
                lines = [
                    f"[score: {r['score']:.4f}]",
                    f"CVE ID: {meta.get('cve_id', 'N/A')}",
                    f"Published: {meta.get('published', 'N/A')}",
                    f"Status: {meta.get('status', 'N/A')}",
                    f"Severity: {meta.get('severity', 'N/A')} (CVSS {meta.get('cvss_score', 'N/A')})",
                    f"Attack Vector: {meta.get('attack_vector', 'N/A')}",
                    f"CWE: {meta.get('cwe', 'N/A')}",
                    f"Description: {r['content']}",
                ]
                if refs:
                    lines.append(f"References: {', '.join(refs)}")
                sections.append("\n".join(lines))
            return "\n\n---\n\n".join(sections)

        return search_cve
        

class LightRAGTool:

    def __init__(self, lightrag_url: str, mode: str = "mix", top_k: int = 10, rerank: bool = True):
        self.lightrag_url = lightrag_url
        self.mode = mode
        self.top_k = top_k
        self.rerank = rerank

    def retrieve_cves(self, query: str) -> str:
        resp = httpx.post(
            f"{self.lightrag_url}/query",
            json={
                "query": query,
                "mode": self.mode,
                "top_k": self.top_k,
                "enable_rerank": self.rerank,
                "stream": False,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()["response"]
    def get_tool(self):
        instance = self

        @tool(description="Searches a knowledge graph of CVE vulnerabilities for entities, relationships, and context relevant to the query")
        def search_cve(query: Annotated[str, "The query used to search for CVEs"]) -> str:
            context = instance.retrieve_cves(query)
            if not context or context.strip() == "":
                return "No relevant CVE context found."
            return context

        return search_cve


class VulnAgent:
    def __init__(self, llm_base_url, llm_api_key, llm_model, faiss_url: str, lightrag_url: str, top_k: int, rerank: bool = False, use_responses: bool = False):

        self.system_prompt = """You are a vulnerability research assistant with access to a CVE knowledge base. \
When the user asks about vulnerabilities, CVEs, or security issues, ALWAYS use the search_cve tool to retrieve relevant information before answering. \
Ground your answers in the retrieved data. If the tool returns no results, say so explicitly."""
        self.prompt_template = ChatPromptTemplate.from_messages(messages=[("system", self.system_prompt),
                                                                          MessagesPlaceholder(variable_name="messages")])

        self.llm_base = ChatOpenAI(model=llm_model, base_url=llm_base_url, api_key = llm_api_key, use_responses_api=use_responses)

        self.faiss_tool = FAISSTool(faiss_url = faiss_url, top_k = top_k, rerank=rerank)
        self.lightrag_tool = LightRAGTool(lightrag_url=lightrag_url, top_k = top_k, rerank=rerank)
        self.checkpointer = InMemorySaver()

        self.graph = self._compile()

        


    async def _chat_agent(self, state: ChatState, config: RunnableConfig):

        writer: StreamWriter = get_stream_writer()

        last_msg = state.messages[-1] if state.messages else None
        if isinstance(last_msg, ToolMessage):
            tool_results = []
            for msg in reversed(state.messages):
                if isinstance(msg, ToolMessage):
                    tool_results.append(msg.content)
                else:
                    break
            tool_results.reverse()
            writer({"retrieved_docs": tool_results})

        if state.tool_mode == 'faiss':
            tool = self.faiss_tool.get_tool()
        else:
            tool = self.lightrag_tool.get_tool()

        agent = self.llm_base.bind_tools([tool])
        chain = self.prompt_template | agent
        
        accumulated: AIMessageChunk | None = None
        async for chunk in chain.astream({'messages': state.messages}):
            if chunk.content:
                writer({"chunk": chunk.content})
            accumulated = chunk if not accumulated else accumulated + chunk

        if not accumulated:
            fallback = AIMessage(content="I'm sorry, I couldn't process that request")
            writer({"chunk": fallback.content})
            return {"messages": [fallback]}
        

        full_response = AIMessage(
            content = accumulated.content or "",
            tool_calls = accumulated.tool_calls or []
        )

        print(state.messages)

        return {"messages": [full_response]}

    async def _should_continue(self, state: ChatState):
        writer: StreamWriter = get_stream_writer()

        #check last message for tool calls. If so, route to tool node and call tools. If not, stream.
        last_message = state.messages[-1]
        if last_message.tool_calls:
            if state.tool_mode == 'faiss':
                return "faiss_tool"
            else:
                return "lightrag_tool"
        else:
            return END


    async def invoke(self, message: str, session_id: str, tool_mode: str = "faiss") -> AsyncGenerator[dict, None]:
        input_state = {
            "messages": [HumanMessage(content=message)],
            "tool_mode": tool_mode,
        }
        config = {"configurable": {"thread_id": session_id}}
        async for event, metadata in self.graph.astream(
            input_state, config=config, stream_mode="custom"
        ):
            yield event

    async def invoke_sync(self, message: str, session_id: str, tool_mode: str = "faiss") -> dict:
        input_state = {
            "messages": [HumanMessage(content=message)],
            "tool_mode": tool_mode,
        }
        config = {"configurable": {"thread_id": session_id}}
        result = await self.graph.ainvoke(input_state, config=config)
        messages = result["messages"]

        retrieved_docs = []
        for msg in (messages):
            if isinstance(msg, ToolMessage):
                retrieved_docs.append(msg.content)
        retrieved_docs.reverse()

        return {"response": messages[-1].content, "retrieved_docs": retrieved_docs}

    def _compile(self):
        graph_builder = StateGraph(ChatState)

        graph_builder.add_node("chat_agent", self._chat_agent)
        graph_builder.add_node("faiss_tool", ToolNode(tools = [self.faiss_tool.get_tool()]))
        graph_builder.add_node("lightrag_tool", ToolNode(tools = [self.lightrag_tool.get_tool()]))

        graph_builder.add_edge(START, "chat_agent")

        graph_builder.add_conditional_edges("chat_agent", self._should_continue)

        graph_builder.add_edge("faiss_tool", "chat_agent")

        graph_builder.add_edge("lightrag_tool", "chat_agent")

        return graph_builder.compile(checkpointer=self.checkpointer)





app = FastAPI(title="VulnAgent", description="Vulnerability Agent")

agent = VulnAgent(
    llm_base_url=os.getenv("OPENAI_BASE_URL", ""),
    llm_api_key=os.getenv("OPENAI_API_KEY", ""),
    llm_model=os.getenv("CHAT_MODEL", ""),
    faiss_url=os.getenv("FAISS_URL", "http://faiss:8000"),
    lightrag_url=os.getenv("LIGHTRAG_URL", "http://lightrag:9621"),
    top_k=int(os.getenv("TOP_K", "4")),
    rerank=os.getenv("RERANK", "false").lower() == "true",
    use_responses=os.getenv("USE_RESPONSES", default="false").lower() == "true"
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest):
    async def sse_stream() -> AsyncGenerator[str, None]:
        async for event in agent.invoke(request.message, session_id=request.session_id, tool_mode=request.retriever_mode):
            if "chunk" in event:
                yield f"data: {json.dumps({'content': event['chunk']})}\n\n"
            elif "retrieved_docs" in event:
                yield f"data: {json.dumps({'retrieved_docs': event['retrieved_docs']})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_stream(), media_type="text/event-stream")


@app.post("/chat/sync")
async def chat_sync(request: ChatRequest):
    return await agent.invoke_sync(request.message, session_id=request.session_id, tool_mode=request.retriever_mode)