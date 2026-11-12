
from agents import (
    run_research_agent,
    run_writer_chain,
    run_critic_chain
)


def run_pipeline(topic: str):

    print("\n" + "=" * 60)
    print("MULTI-AGENT RESEARCH SYSTEM")
    print("=" * 60)

    # --------------------------------
    # 1. Research Phase
    # --------------------------------

    print("\n[1] Research Agent is working...")

    research_data = run_research_agent(topic)

    print("\nResearch completed successfully!")


    # --------------------------------
    # 2. Writing Phase
    # --------------------------------

    print("\n[2] Writer Chain is generating the report...")

    report = run_writer_chain(
        topic,
        research_data
    )

    print("\nResearch report generated successfully!")


    # --------------------------------
    # 3. Critic Phase
    # --------------------------------

    print("\n[3] Critic Chain is reviewing the report...")

    feedback = run_critic_chain(
        topic,
        report
    )

    print("\nReview completed!")


    # --------------------------------
    # 4. Display Results
    # --------------------------------

    print("\n" + "=" * 60)
    print("FINAL RESEARCH REPORT")
    print("=" * 60)

    print(report)

    print("\n" + "=" * 60)
    print("CRITIC FEEDBACK")
    print("=" * 60)

    print(feedback)

    return {
        "topic": topic,
        "research_data": research_data,
        "report": report,
        "feedback": feedback
    }


