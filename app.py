
import streamlit as st
from pipeline import run_pipeline

# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔬",
    layout="wide"
)


# --------------------------------
# Application Header
# --------------------------------

st.title("🔬 Multi-Agent Research System")

st.markdown(
    """
    Research smarter with AI agents.

    Enter a research topic and let our multi-agent system
    search the web, collect information, write a report,
    and critically evaluate its quality.
    """
)

st.divider()


# --------------------------------
# Research Input
# --------------------------------

st.subheader("Enter Research Topic")

topic = st.text_input(
    "What would you like to research?",
    placeholder="e.g. Impact of Generative AI on Software Development"
)

start_research = st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True
)


# --------------------------------
# Research Pipeline
# --------------------------------

if start_research:

    if not topic.strip():

        st.warning("Please enter a research topic.")

    else:

        try:

            with st.spinner(
                "AI agents are researching your topic. Please wait..."
            ):

                result = run_pipeline(topic.strip())

            st.success("Research completed successfully!")

            # --------------------------------
            # Display Research Results
            # --------------------------------

            st.divider()

            st.header("📊 Research Results")

            tab1, tab2, tab3 = st.tabs(
                [
                    "Research Data",
                    "Final Report",
                    "Critic Feedback"
                ]
            )

            with tab1:

                st.subheader("Research Agent Output")

                st.write(result["research_data"])

            with tab2:

                st.subheader("Generated Research Report")

                st.markdown(result["report"])

                st.download_button(
                    label="📥 Download Research Report",
                    data=result["report"],
                    file_name="research_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            with tab3:

                st.subheader("Critic Review")

                st.markdown(result["feedback"])

        except Exception as e:

            st.error(f"An error occurred: {str(e)}")

            st.info(
                "Please check your API keys, model configuration, "
                "and terminal logs."
            )


# --------------------------------
# Footer
# --------------------------------

st.divider()

st.caption(
    "Built using LangChain, Groq, Tavily and Streamlit"
)
