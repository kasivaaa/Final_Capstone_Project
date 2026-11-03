from pathlib import Path

import joblib
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "baseline_model.joblib"

LABEL_NAMES = {
    0: "Neutral",
    1: "Offensive",
    2: "Hate Speech",
}

LABEL_DESCRIPTIONS = {
    0: (
        "The model identifies this post as neutral in tone. "
        "It may contain ordinary discussion, information or opinions."
    ),
    1: (
        "The model identifies language that may be insulting, "
        "hostile or offensive."
    ),
    2: (
        "The model identifies language that may contain hate speech "
        "or hostile language directed at a person or group."
    ),
}

st.set_page_config(
    page_title="Election Early Warning System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🛡️ Election Early Warning System")

st.markdown(
    """
    Identify language patterns in election-related social media
    posts using machine learning.

    **How it works:** Enter a post, analyse its language, and
    review the model's predicted category and confidence scores.
    """
)

st.divider()


# --------------------------------------------------
# User input
# --------------------------------------------------
st.subheader("Analyse a social media post")

st.write(
    "Paste the text of a public social media post below. "
    "You can enter text in English, Swahili or Sheng."
)

with st.form("post_analysis_form"):
    post_text = st.text_area(
        "Post text",
        placeholder=(
            "Example: Wananchi wameombwa kushiriki "
            "katika mkutano wa amani kwa utulivu..."
        ),
        height=150,
        max_chars=10000,
    )

    analyse = st.form_submit_button(
        "🔍 Analyse post",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# Analysis and results
# --------------------------------------------------
if analyse:
    if not post_text.strip():
        st.warning("Please enter some text before analysing.")

    else:
        try:
            with st.spinner("Analysing your post..."):
                model = load_model()

                predicted_label = int(
                    model.predict([post_text])[0]
                )

                probabilities = model.predict_proba(
                    [post_text]
                )[0]

                classes = model.named_steps[
                    "classifier"
                ].classes_

                class_scores = {
                    int(label): float(score)
                    for label, score in zip(
                        classes, probabilities
                    )
                }

            if predicted_label not in LABEL_NAMES:
                raise ValueError(
                    "The model returned an unknown class label."
                )

            predicted_name = LABEL_NAMES[predicted_label]
            predicted_score = class_scores[predicted_label]

            st.divider()
            st.header("Analysis results")

            st.success(
                f"Analysis complete. The strongest model match "
                f"is: {predicted_name}."
            )

            st.metric(
                label="Predicted category",
                value=predicted_name,
                help=(
                    "This category is selected from the model's "
                    "predicted output."
                ),
            )

            st.write(LABEL_DESCRIPTIONS[predicted_label])

            st.caption(
                f"Model score for this category: "
                f"{predicted_score:.1%}"
            )

            st.subheader("Category scores")

            st.write(
                "Compare the model's estimated scores for "
                "each category."
            )

            for label in sorted(LABEL_NAMES):
                score = class_scores.get(label, 0.0)

                st.write(
                    f"**{LABEL_NAMES[label]}**"
                )
                st.progress(score)
                st.caption(f"Estimated score: {score:.1%}")

            st.info(
                "These scores describe the model's estimates, "
                "not the probability that violence will occur. "
                "Review the original post and its context before "
                "making decisions. The model can make mistakes."
            )

        except Exception as exc:
            st.error(
                f"The post could not be analysed. "
                f"Please check the model files and try again. "
                f"Details: {exc}"
            )


# --------------------------------------------------
# User guidance
# --------------------------------------------------
st.divider()

with st.expander("How to use this system"):
    st.markdown(
        """
        1. Copy the text of a public social media post.
        2. Paste it into the text box.
        3. Click **Analyse post**.
        4. Read the predicted category and compare the scores.
        5. Consider the full context before interpreting the result.

        The model analyses text rather than images, videos or
        the full context of a social media conversation.
        """
    )

with st.expander("Understanding the categories"):
    st.markdown(
        """
        - **Neutral:** Language identified as neutral.
        - **Offensive:** Language that may be insulting or hostile.
        - **Hate Speech:** Language that may express hatred or
          hostility towards a person or group.

        These are the model's assigned categories. A prediction
        does not independently establish a person's intent.
        """
    )

with st.expander("Technical information and model evaluation"):
    st.markdown(
        """
        **Model:** TF-IDF with Logistic Regression

        **Training records:** 5,062

        **Held-out test records:** 1,266

        **Test accuracy:** 68.64%

        **Macro F1-score:** 0.44

        Accuracy measures the proportion of test predictions
        that were correct. Macro F1 summarises performance across
        classes while giving each class equal weight.

        The evaluation indicates that the model has difficulty
        identifying classes 1 and 2. Its predictions should
        therefore support further review, not automatic
        enforcement or decisions about individuals.
        """
    )

st.caption(
    "Election Early Warning System | "
    "Machine learning for social media analysis"
)
