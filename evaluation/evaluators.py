from typing import Any, Dict, List

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field


#Quick tip, Structured Output order matters. 
# If you put reasoning before score, you can actually improve your consistancy and accuracy 
# by allowing the model to "think" a bit before deciding on a numerical score
# There's less of a need for this nowadays due to reasoing models, but it certainly doesn't hurt
# (And it's not garunteed you'd want to use a reasoning model, even for evals.)

class MetricResult(BaseModel):
    reasoning: str = Field(description="Brief explanation for the score")
    score: float = Field(description="Score from 0.0 to 1.0")

class EvalSample(BaseModel):
    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str
    source: str = ""


class SampleResult(BaseModel):
    sample: EvalSample
    faithfulness: MetricResult | None = None
    groundedness: MetricResult | None = None
    answer_relevancy: MetricResult | None = None
    context_precision: MetricResult | None = None
    context_recall: MetricResult | None = None


#This is not how I like to do it, but including the prompts in the class gets messy. 
# Typically, I'd have some layers of abstraction here (In what I call a "Generator") that tightly bind llm params, system prompt and tools/structured output together in a portable object)
faithfulness_prompt = """You are an evaluation judge assessing the faithfulness of a RAG pipeline's answer.

Faithfulness measures whether the answer contradicts the retrieved context. \
An answer is unfaithful if it states something that directly conflicts with or misrepresents \
information in the context. This metric does NOT penalize the answer for including information \
beyond the context — only for contradicting it.

Scoring:
- 1.0: The answer does not contradict anything in the retrieved context
- 0.7-0.9: The answer is mostly consistent, with one minor misstatement
- 0.4-0.6: The answer contains a clear contradiction alongside correct claims
- 0.1-0.3: The answer contradicts the context on multiple points
- 0.0: The answer directly contradicts the core facts in the context

You will be provided the question, retrieved context, and the answer to evaluate."""

groundedness_prompt = """You are an evaluation judge assessing the groundedness of a RAG pipeline's answer.

Groundedness measures whether the answer is derived only from the retrieved context. \
An ungrounded answer introduces facts, claims, or details not present in the context, \
even if those claims happen to be true. This metric penalizes any information that cannot \
be traced back to the provided context.

Scoring:
- 1.0: Every claim in the answer is directly traceable to the retrieved context
- 0.7-0.9: Most claims come from the context, with minor additions
- 0.4-0.6: A mix of grounded and ungrounded claims
- 0.1-0.3: Most claims are not found in the context
- 0.0: The answer is entirely based on outside information

You will be provided the question, retrieved context, and the answer to evaluate."""

answer_relevancy_prompt = """You are an evaluation judge assessing the relevancy of a RAG pipeline's answer.

Answer relevancy measures whether the answer directly addresses the question asked. \
An answer is irrelevant if it discusses unrelated topics, is evasive or overly generic, \
or provides information that doesn't answer what was asked.

Scoring:
- 1.0: The answer directly and completely addresses the question
- 0.7-0.9: The answer mostly addresses the question with minor tangents
- 0.4-0.6: The answer partially addresses the question
- 0.1-0.3: The answer barely relates to the question
- 0.0: The answer is completely off-topic

You will be provided the question and the answer to evaluate."""

context_precision_prompt = """You are an evaluation judge assessing the precision of a RAG pipeline's retrieval.

Context precision measures whether the retrieved context chunks are relevant to the question. \
Low precision means the retrieval system returned irrelevant documents alongside relevant ones.

Scoring:
- 1.0: All retrieved chunks contain information relevant to answering the question
- 0.7-0.9: Most chunks are relevant, with one or two irrelevant ones
- 0.4-0.6: Roughly half the chunks are relevant
- 0.1-0.3: Most chunks are irrelevant to the question
- 0.0: None of the retrieved chunks are relevant

You will be provided the question and the retrieved context chunks to evaluate."""

context_recall_prompt = """You are an evaluation judge assessing the recall of a RAG pipeline's retrieval.

Context recall measures whether the retrieved context contains all the information needed \
to produce the reference answer. Low recall means the retrieval system missed important documents.

Scoring:
- 1.0: The retrieved context contains all information from the reference answer
- 0.7-0.9: The context covers most of the reference, missing minor details
- 0.4-0.6: The context covers about half of what's needed
- 0.1-0.3: The context is missing most of the needed information
- 0.0: The context contains none of the information from the reference

You will be provided the question, retrieved context, and reference answer to evaluate."""


class Faithfulness:
    def __init__(self, llm):
        self.chain = (
            ChatPromptTemplate.from_messages(
                messages=[
                    ("system", faithfulness_prompt),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )| llm.with_structured_output(MetricResult)
        )

    def evaluate(self, sample: EvalSample) -> MetricResult:
        context = "\n\n---\n\n".join(sample.retrieved_contexts) or "(no context retrieved)"
        content = (
            f"Question: {sample.user_input}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            f"Answer: {sample.response}"
        )
        return self.chain.invoke({"messages": [HumanMessage(content=content)]})


class Groundedness:
    def __init__(self, llm):
        self.chain = (
            ChatPromptTemplate.from_messages(
                messages=[
                    ("system", groundedness_prompt),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )| llm.with_structured_output(MetricResult)
        )

    def evaluate(self, sample: EvalSample) -> MetricResult:
        context = "\n\n---\n\n".join(sample.retrieved_contexts) or "(no context retrieved)"
        content = (
            f"Question: {sample.user_input}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            f"Answer: {sample.response}"
        )
        return self.chain.invoke({"messages": [HumanMessage(content=content)]})


class AnswerRelevancy:
    def __init__(self, llm):
        self.chain = (
            ChatPromptTemplate.from_messages(
                messages=[
                    ("system", answer_relevancy_prompt),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )| llm.with_structured_output(MetricResult)
        )

    def evaluate(self, sample: EvalSample) -> MetricResult:
        content = (
            f"Question: {sample.user_input}\n\n"
            f"Answer: {sample.response}"
        )
        return self.chain.invoke({"messages": [HumanMessage(content=content)]})


class ContextPrecision:
    def __init__(self, llm):
        self.chain = (
            ChatPromptTemplate.from_messages(
                messages=[
                    ("system", context_precision_prompt),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )| llm.with_structured_output(MetricResult)
        )

    def evaluate(self, sample: EvalSample) -> MetricResult:
        context = "\n\n---\n\n".join(sample.retrieved_contexts) or "(no context retrieved)"
        content = (
            f"Question: {sample.user_input}\n\n"
            f"Retrieved Context:\n{context}"
        )
        return self.chain.invoke({"messages": [HumanMessage(content=content)]})


class ContextRecall:
    def __init__(self, llm):
        self.chain = (
            ChatPromptTemplate.from_messages(
                messages=[
                    ("system", context_recall_prompt),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )| llm.with_structured_output(MetricResult)
        )

    def evaluate(self, sample: EvalSample) -> MetricResult:
        context = "\n\n---\n\n".join(sample.retrieved_contexts) or "(no context retrieved)"
        content = (
            f"Question: {sample.user_input}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            f"Reference Answer: {sample.reference}"
        )
        return self.chain.invoke({"messages": [HumanMessage(content=content)]})

def evaluate(llm, samples: list[EvalSample], metrics: list[str] | None = None) -> List[SampleResult]:

    llm = llm

    metric_generators = {
    "faithfulness": Faithfulness(llm),
    "groundedness": Groundedness(llm),
    "answer_relevancy": AnswerRelevancy(llm),
    "context_precision": ContextPrecision(llm),
    "context_recall": ContextRecall(llm),
    }
    
    active_metrics: Dict[str, Any] = {}
    if not metrics:
        active_metrics = metric_generators
    else:
        for metric in metrics:
            metric_generator = metric_generators.get(metric
            )
            if metric_generator:
                active_metrics[metric] = metric_generator

    sample_results = []
    total = len(samples)
    for i, sample in enumerate(samples):
        sample_result = {'sample': sample}
        #Could make this faster by calling in parallel
        print(f"Evaluating sample {i+1}/{total}")
        for name, metric in active_metrics.items():
            result = metric.evaluate(sample)
            sample_result[name] = result
        sample_results.append(SampleResult(**sample_result))

    return sample_results
