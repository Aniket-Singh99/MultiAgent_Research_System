
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import research_tools

load_dotenv()


# --------------------------------
# 1. Initialize LLM
# --------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1024
)


# --------------------------------
# 2. Research Agent
# --------------------------------

research_agent = create_agent(
    model=llm,
    tools=research_tools,
    system_prompt="""
    You are a research agent.

    Your tasks:
    1. Search the web using Tavily.
    2. Select relevant URLs.
    3. Scrape useful webpages.
    4. Extract important facts, statistics and findings.
    5. Preserve source URLs.

    Return a concise research summary with:
    - Key findings
    - Important facts
    - Statistics
    - Different perspectives
    - Source URLs

    Avoid irrelevant information and unsupported claims.
    """
)


def run_research_agent(topic: str) -> str:

    response = research_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"Research the following topic: {topic}"
            }
        ]
    })

    return response["messages"][-1].content


# --------------------------------
# 3. Writer Chain
# --------------------------------

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a professional research report writer.

        Your task is to convert the provided research material
        into a detailed, structured, and easy-to-understand report.

        Follow this structure:

        1. Title
        2. Introduction
        3. Background
        4. Key Findings
        5. Detailed Analysis
        6. Challenges and Limitations
        7. Conclusion
        8. References

        Instructions:
        - Use only the provided research material.
        - Maintain a professional and academic tone.
        - Do not invent facts or statistics.
        - Include source URLs in the references.
        - Avoid unnecessary repetition.
        """
    ),
    (
        "human",
        """
        Research Topic:
        {topic}

        Research Material:
        {research_data}

        Write a comprehensive research report.
        """
    )
])

writer_chain = writer_prompt | llm | StrOutputParser()


def run_writer_chain(topic: str, research_data: str) -> str:

    response = writer_chain.invoke({
        "topic": topic,
        "research_data": research_data
    })

    return response


# --------------------------------
# 4. Critic Chain
# --------------------------------

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a strict research report reviewer.

        Your job is to critically evaluate the research report.

        Evaluate the report based on:

        1. Factual accuracy
        2. Research depth
        3. Completeness
        4. Logical structure
        5. Clarity and readability
        6. Relevance to the research topic
        7. References and supporting evidence

        Identify:
        - Missing information
        - Unsupported claims
        - Repeated content
        - Weak arguments
        - Areas requiring improvement

        At the end, provide:
        VERDICT: APPROVED or NEEDS REVISION

        Be strict and do not approve an incomplete report.
        """
    ),
    (
        "human",
        """
        Research Topic:
        {topic}

        Research Report:
        {report}

        Critically review this report and provide detailed feedback.
        """
    )
])

critic_chain = critic_prompt | llm | StrOutputParser()


def run_critic_chain(topic: str, report: str) -> str:

    response = critic_chain.invoke({
        "topic": topic,
        "report": report
    })

    return response
