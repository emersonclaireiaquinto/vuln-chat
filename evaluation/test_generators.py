from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field


class GeneratedTest(BaseModel):
    question: str = Field(description="Question that relates to the document(s)")
    answer: str = Field(
        description="Answer for the question created above. Must be grounded in the source documents"
    )


class Persona(BaseModel):
    name: str
    description: str


single_hop_system_prompt = """You are a test set generator for evaluating a vulnerability knowledge base. \
Your goal is to create a question/answer pair from a single CVE document.

You will be provided a persona. Tailor the question to what that persona would realistically ask — \
consider their technical depth, their responsibilities, and the terminology they would use.

Write the question the way a real person would type it into a chat interface: \
one short sentence, one focus, no compound sub-questions. \
Do not provide too many underlying details about a CVE \
Do not start the question with "As a ..." or reference the persona explicitly. \

Examples of good questions:
- "What is the CVSS score and attack vector for Log4Shell?"
- "What remediation steps are recommended for Spring4Shell?"
- "How does the runc container escape vulnerability work?"
- "What does Heartbleed expose?"
- "Is CVE-2022-22965 exploitable in Spring Boot jar deployments?"

Examples of bad questions:
- "For CVE-2018-15664 in Docker, what is the severity and attack vector, and what kind of access could an attacker gain through the vulnerable 'docker cp' API behavior?"
- "As a product manager, how urgent is CVE-2022-3786 to prioritize, and in plain language what kind of customer impact could it have?"
- "What kind of access could this vulnerability give an attacker?"
The answer must be 2-4 sentences, grounded entirely in the provided document. Do not include any outside knowledge."""

multi_hop_system_prompt = """You are a test set generator for evaluating a vulnerability knowledge base. \
Your goal is to create a question/answer pair that requires information from multiple CVE documents to answer.

You will be provided a persona. Tailor the question to what that persona would realistically ask.

The question must require reasoning across the provided documents — comparing severity, \
exploitability, shared weaknesses, or asking which to patch first. \
A good multi-hop question cannot be fully answered by reading only one of the documents.

Write the question the way a real person would type it into a chat interface: \
one short sentence, conversational tone, no compound sub-questions joined with "and". \
Do not leak specific technical details from the documents into the question. \
Do not start with "As a ..." or reference the persona explicitly.

Frame questions broadly so the retrieval system has to find the right documents — \
refer to the software ecosystem, vulnerability class, or affected component \
rather than listing specific CVE IDs. You may mention at most one CVE by name or nickname.

Examples of good questions:
- "Which OpenSSL vulnerabilities should I prioritize patching?"
- "What container escape CVEs should I worry about?"
- "Which is the bigger risk for my TLS stack, Heartbleed or the newer OpenSSL overflows?"
- "Are there any Log4j-related CVEs beyond the original Log4Shell?"

Examples of bad questions (do NOT generate these):
- "I'm triaging two incoming critical Apache issues, CVE-2021-45046 and CVE-2017-5638. Which one should I prioritize first for emergency patching if I'm looking for the higher-risk network-exploitable bug, and what key exploitation condition or trigger distinguishes each vulnerability?"
- "Between CVE-2014-0160 and CVE-2022-3786, which is more likely to expose secrets?"

The answer must be 2-4 sentences, grounded entirely in the provided documents. Do not include any outside knowledge."""

global_system_prompt = """You are a test set generator for evaluating a vulnerability knowledge base. \
Your goal is to create a question/answer pair that asks about a pattern or commonality across many CVE documents.

You will be provided a persona. Tailor the question to what that persona would realistically ask.

The question should ask about a pattern across the documents — shared attack vectors, \
CWE categories, severity trends, common software components, or aggregate counts.

Write the question the way a real person would type it into a chat interface: \
one short sentence, conversational tone. Do not reference the persona explicitly. \
Do not list CVE IDs in the question — ask about the pattern broadly.

Examples of good questions:
- "How many of the known CVEs have a network attack vector?"
- "What's the most common CWE across critical-severity vulnerabilities?"
- "Are most of these vulnerabilities remote code execution or something else?"
- "Which vulnerability category shows up the most in 2021-2022 CVEs?"

Examples of bad questions (do NOT generate these):
- "Looking across CVE-2021-45046, CVE-2021-34527, CVE-2021-26855, CVE-2023-23397, and CVE-2022-22963, what attack vector do they have in common, and how many are explicitly described as leading to remote code execution versus some other impact?"

The answer must be 2-4 sentences, grounded entirely in the provided documents. Do not include any outside knowledge."""


class TestGenerator:
    def __init__(self, llm):
        self._single_hop_chain = self._build_chain(llm, single_hop_system_prompt)
        self._multi_hop_chain = self._build_chain(llm, multi_hop_system_prompt)
        self._global_chain = self._build_chain(llm, global_system_prompt)

    def _build_chain(self, llm, system_prompt: str):
        return (
            ChatPromptTemplate.from_messages(
                messages=[
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            | llm.with_structured_output(GeneratedTest)
        )

    def _invoke(self, chain, documents: list[str], persona: Persona) -> GeneratedTest:
        parts = [f"Persona: {persona.name} — {persona.description}"]
        for i, doc in enumerate(documents, 1):
            parts.append(f"Document {i}:\n{doc}")
        message = HumanMessage(content="\n\n".join(parts))
        return chain.invoke({"messages": [message]})

    def generate_single_hop(self, document: str, persona: Persona) -> GeneratedTest:
        return self._invoke(self._single_hop_chain, [document], persona)

    def generate_multi_hop(self, documents: list[str], persona: Persona) -> GeneratedTest:
        return self._invoke(self._multi_hop_chain, documents, persona)

    def generate_global(self, documents: list[str], persona: Persona) -> GeneratedTest:
        return self._invoke(self._global_chain, documents, persona)
