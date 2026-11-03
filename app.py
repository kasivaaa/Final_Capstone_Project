
from pathlib import Path
from datetime import datetime
import json
import re
import urllib.error
import urllib.request

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "baseline_model.joblib"
HISTORY_PATH = ARTIFACTS_DIR / "analysis_history.csv"

LABEL_NAMES = {
    0: "Neutral",
    1: "Offensive",
    2: "Hate Speech",
}

CATEGORY_ORDER = ["Neutral", "Offensive", "Hate Speech"]

HISTORY_COLUMNS = [
    "Date analysed",
    "Source",
    "Post URL",
    "Post text",
    "Predicted category",
    "Neutral score",
    "Offensive score",
    "Hate Speech score",
    "Early warning signal",
    "Matched warning phrases",
    "Warning explanation",
]


# ============================================================
# 2. STREAMLIT PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Election Violence Early Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Election Violence Early Detection System")

st.markdown(
    """
    Classify social media text as Neutral, Offensive, or Hate Speech,
    and separately screen for configured violence-related phrases.

    Paste text manually. Fetching posts from X requires working API access.
    """
)

st.warning(
    "Research prototype only. Predictions are not verified threats "
    "or evidence of actual election violence."
)


# ============================================================
# 3. LOAD THE MACHINE-LEARNING MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# 4. LOAD SAVED ANALYSIS HISTORY
# ============================================================

def load_history():
    """Read the saved analysis history from CSV."""

    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    try:
        history = pd.read_csv(HISTORY_PATH)

        for column in HISTORY_COLUMNS:
            if column not in history.columns:
                history[column] = ""

        return history[HISTORY_COLUMNS]

    except (OSError, ValueError, pd.errors.ParserError):
        st.error(
            "The saved history file could not be read. "
            "Check artifacts/analysis_history.csv."
        )
        return pd.DataFrame(columns=HISTORY_COLUMNS)


# ============================================================
# 5. SAVE ANALYSIS HISTORY
# ============================================================

def save_history(record):
    """Append one analysis result while preserving older history columns."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    new_record = pd.DataFrame([record])
    if HISTORY_PATH.exists():
        try:
            old_history = pd.read_csv(HISTORY_PATH)
        except (OSError, ValueError, pd.errors.ParserError):
            old_history = pd.DataFrame()
        columns = list(dict.fromkeys(
            list(old_history.columns) + list(new_record.columns) + HISTORY_COLUMNS
        ))
        combined = pd.concat([
            old_history.reindex(columns=columns),
            new_record.reindex(columns=columns)
        ], ignore_index=True)
    else:
        combined = new_record.reindex(columns=HISTORY_COLUMNS)
    combined.to_csv(HISTORY_PATH, index=False, encoding="utf-8-sig")


# ============================================================
# 5A. RULE-BASED EARLY-WARNING LANGUAGE SCREEN
# ============================================================

# Transparent prototype indicators. These are NOT learned from labelled
# election-violence examples and must not be interpreted as a validated
# probability that violence will occur.
ELECTION_CONTEXT_TERMS = [
    "election", "elections", "vote", "voting", "ballot", "polling station",
    "campaign", "political rally", "rally", "mwanasiasa", "uchaguzi",
    "kupiga kura", "kura", "kampeni", "mkutano wa kisiasa",
]

DIRECT_VIOLENCE_PHRASES = [
    "kill them", "kill all", "we will kill", "go and kill",
    "burn their houses", "burn them", "attack them", "go attack",
    "let us attack", "let's attack", "take up arms", "murder them",
    "shoot them", "slaughter them", "waue wote", "tuwaue", "waueni",
    "tuwaachome", "tuwachome", "choma nyumba zao", "choma nyumba",
    "pigeni", "washambulie", "shambulia", "ua wote", "wateketeze",
]

MOBILIZATION_PHRASES = [
    "bring weapons", "carry weapons", "arm yourselves", "go after them",
    "drive them out", "chase them out", "attack the supporters",
    "shambulieni", "tuwafukuze", "wafukuzeni", "tuwavamie",
]

NEGATION_OR_REPORTING_CUES = [
    "do not attack", "don't attack", "stop the violence", "against violence",
    "condemn the attack", "reported that", "news about", "called for peace",
    "usiue", "msiue", "tusichochee", "amani",
]

def screen_violence_language(text):
    """Rule-based screening for election-violence indicators, not a trained predictor."""
    normalized = re.sub(r"\s+", " ", str(text).lower()).strip()

    def matches_phrases(phrases):
        found = []
        for phrase in phrases:
            pattern = r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)"
            if re.search(pattern, normalized):
                found.append(phrase)
        return sorted(set(found))

    election_matches = matches_phrases(ELECTION_CONTEXT_TERMS)
    direct_matches = matches_phrases(DIRECT_VIOLENCE_PHRASES)
    mobilization_matches = matches_phrases(MOBILIZATION_PHRASES)
    peace_cues = matches_phrases([
        "do not attack", "don't attack", "stop the violence", "against violence",
        "condemn the attack", "called for peace", "usiue", "msiue",
        "tusichochee", "amani", "reject violence", "peaceful elections",
        "hatutaki vurugu", "no violence", "maintain peace",
    ])
    reporting_cues = matches_phrases([
        "reported that", "news about", "according to reports",
        "condemned the attack", "the report said", "police reported",
    ])

    matched = sorted(set(election_matches + direct_matches + mobilization_matches + peace_cues + reporting_cues))

    # Avoid treating obvious anti-violence statements as endorsements.
    # This is a simple safeguard, not full contextual or negation understanding.
    if peace_cues and not (direct_matches or mobilization_matches):
        level = "NO CLEAR ELECTION-VIOLENCE SIGNAL — peace wording detected"
        reason = (
            "The statement contains configured wording opposing or discouraging violence. "
            "This simple screen can still miss sarcasm, mixed messages, quotations, and context."
        )
    elif reporting_cues and (direct_matches or mobilization_matches):
        level = "CONTEXT NEEDED — violence discussed in a reporting context"
        reason = (
            "Violence-related wording and a reporting cue were found. The statement may be "
            "reporting or condemning an event rather than encouraging it. The rules cannot reliably determine intent."
        )
    elif election_matches and (direct_matches or mobilization_matches):
        level = "POTENTIAL ELECTION-VIOLENCE RISK INDICATORS DETECTED"
        reason = (
            "Both election-related context and configured physical-violence or mobilization wording were detected. "
            "This warrants attention as a text-based warning signal, but it is not proof of intent and does not "
            "predict that violence will occur."
        )
    elif direct_matches or mobilization_matches:
        level = "POTENTIAL VIOLENCE WORDING — ELECTION LINK UNCLEAR"
        reason = (
            "Violence-related or mobilization wording was detected, but the configured election context was not found. "
            "The statement may be unrelated to elections; interpret it in context."
        )
    elif election_matches:
        level = "NO CLEAR ELECTION-VIOLENCE SIGNAL — election context only"
        reason = (
            "Election-related terms were detected, but no configured direct violence or mobilization wording was found. "
            "This does not guarantee the statement is safe; coded language and contextual signals may be missed."
        )
    else:
        level = "INSUFFICIENT INDICATORS — no configured signal detected"
        reason = (
            "No configured combination was matched. This does not prove the statement is safe: "
            "coded language, Sheng, spelling variants, sarcasm, and missing context may be missed."
        )

    return level, matched, reason


# ============================================================
# 6. FETCH A PUBLIC POST FROM X
# ============================================================

def fetch_x_post(post_url):
    """Fetch one public X post using its URL and Bearer Token."""

    match = re.search(
        r"(?:https?://)?(?:www\.)?"
        r"(?:x\.com|twitter\.com)/[^/?]+/status/(\d+)",
        post_url.strip(),
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            "Enter a valid X post URL containing "
            "/status/ followed by the post ID."
        )

    try:
        token = st.secrets["X_BEARER_TOKEN"].strip()
    except Exception:
        raise RuntimeError(
            "X_BEARER_TOKEN was not found. Check your "
            ".streamlit/secrets.toml file."
        ) from None

    if not token:
        raise RuntimeError(
            "The X Bearer Token is empty. Check your secrets file."
        )

    post_id = match.group(1)

    api_url = (
        f"https://api.x.com/2/tweets/{post_id}"
        "?tweet.fields=created_at,lang"
    )

    request = urllib.request.Request(
        api_url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "ElectionEarlyWarningResearchPrototype",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            message = "X rejected authentication. Check your token."
        elif exc.code == 402:
            message = (
                "X returned HTTP 402. Your account may need "
                "eligible API credits or a paid usage entitlement."
            )
        elif exc.code == 403:
            message = (
                "X denied access. Your API plan or app permissions "
                "may not allow this request."
            )
        elif exc.code == 404:
            message = (
                "The post was not found or is unavailable through "
                "your API access."
            )
        elif exc.code == 429:
            message = (
                "An X API usage or rate limit was reached. "
                "Check your developer dashboard."
            )
        else:
            message = f"X API returned HTTP {exc.code}."

        raise RuntimeError(message) from None

    except urllib.error.URLError:
        raise RuntimeError(
            "Could not connect to X. Check your internet connection."
        ) from None

    post = result.get("data")

    if not post or not post.get("text"):
        raise RuntimeError(
            "X did not return post text. The post may be unavailable."
        )

    return post


# ============================================================
# 7. POST INPUT METHOD
# ============================================================

source = st.radio(
    "Choose how to provide a post",
    ["Paste text", "Fetch from X post URL"],
    horizontal=True,
)

with st.form("analysis_form"):

    post_text_input = ""
    post_url_input = ""

    if source == "Paste text":

        post_text_input = st.text_area(
            "Paste the social media post",
            placeholder=(
                "Paste an English, Swahili or Sheng post here..."
            ),
            height=160,
            max_chars=10000,
        )

    else:

        post_url_input = st.text_input(
            "Public X post URL",
            placeholder="https://x.com/username/status/123456789",
        )

        st.caption(
            "X API access is required. Private or deleted posts "
            "may not be available."
        )

    analyse = st.form_submit_button(
        "Analyse post",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# 8. ANALYSE THE POST AND SAVE ITS RESULT
# ============================================================

if analyse:

    try:
        post_url = ""

        # Get the text from the selected source.
        if source == "Paste text":

            post_text = post_text_input.strip()

            if not post_text:
                st.error("Please paste a post before analysing.")
                st.stop()

        else:

            post_url = post_url_input.strip()

            if not post_url:
                st.error("Please enter an X post URL.")
                st.stop()

            with st.spinner("Retrieving the post from X..."):
                post_details = fetch_x_post(post_url)
                post_text = post_details["text"]

            st.success("Post retrieved successfully.")

            st.caption(
                "Post language: "
                f"{post_details.get('lang', 'Not provided')}"
            )

            st.text_area(
                "Retrieved post text",
                value=post_text,
                height=140,
                disabled=True,
            )

        # Load the trained model and make a prediction.
        model = load_model()

        predicted_label = int(model.predict([post_text])[0])
        probabilities = model.predict_proba([post_text])[0]
        classes = model.named_steps["classifier"].classes_

        category = LABEL_NAMES.get(
            predicted_label,
            f"Unknown class {predicted_label}",
        )
        warning_signal, matched_phrases, warning_explanation = screen_violence_language(post_text)

        # Prepare scores for each category.
        scores = {
            "Neutral": 0.0,
            "Offensive": 0.0,
            "Hate Speech": 0.0,
        }

        for class_label, score in zip(classes, probabilities):
            class_name = LABEL_NAMES.get(int(class_label))

            if class_name in scores:
                scores[class_name] = round(float(score), 4)

        # Create the record.
        record = {
            "Date analysed": datetime.now().astimezone().strftime(
                "%Y-%m-%d %H:%M:%S %z"
            ),
            "Source": source,
            "Post URL": post_url,
            "Post text": post_text,
            "Predicted category": category,
            "Neutral score": scores["Neutral"],
            "Offensive score": scores["Offensive"],
            "Hate Speech score": scores["Hate Speech"],
            "Early warning signal": warning_signal,
            "Matched warning phrases": "; ".join(matched_phrases),
            "Warning explanation": warning_explanation,
        }

        # Save the record to the local CSV file.
        save_history(record)

        st.success("Analysis completed and saved to history.")

        # Display the prediction.
        st.divider()
        st.subheader("Analysis Results")

        st.metric("Predicted category", category)

        st.subheader("Election-Violence Assessment (Prototype)")
        st.write(f"**Assessment:** {warning_signal}")
        st.write(warning_explanation)
        if matched_phrases:
            st.write("**Matched indicators:** " + ", ".join(matched_phrases))
        st.caption(
            "This is a transparent rule-based screening module, NOT a trained "
            "or validated predictor of whether election violence will occur. "
            "It does not calculate a violence probability. It can miss coded "
            "language, context, spelling variants, sarcasm, quotations, and Sheng. "
            "A phrase match is not proof of intent, and no match is not proof of safety."
        )

        if category == "Hate Speech":
            st.warning(
                "The model classified this text as Hate Speech. "
                "This is a preliminary prediction, not a verified finding."
            )

        elif category == "Offensive":
            st.info(
                "The model classified this text as Offensive. "
                "This does not automatically mean it promotes violence."
            )

        else:
            st.info(
                "The model classified this text as Neutral. "
                "It may still miss harmful content."
            )

        # Display the model scores.
        st.subheader("Model Scores")

        for class_label, score in zip(classes, probabilities):
            class_name = LABEL_NAMES.get(
                int(class_label),
                f"Class {class_label}",
            )

            st.write(f"**{class_name}**")
            st.progress(float(score))
            st.caption(f"Estimated model score: {score:.1%}")

        st.caption(
            "Scores may not be well calibrated. They are not proof "
            "that a post is harmful or safe."
        )

    except Exception as exc:
        st.error(str(exc))


# ============================================================
# 9. LOAD AND DISPLAY ANALYSIS HISTORY
# ============================================================

st.divider()
st.subheader("📋 Analysis History")

history_df = load_history()

if not history_df.empty:

    st.write(f"**Total posts analysed:** {len(history_df)}")
    if "Early warning signal" in history_df.columns:
        warning_count = int(
            history_df["Early warning signal"].fillna("")
            .str.contains("detected", case=False).sum()
        )
        metric_a, metric_b = st.columns(2)
        with metric_a:
            st.metric("Posts with configured warning signals", warning_count)
        with metric_b:
            st.metric(
                "Share of analysed posts",
                f"{warning_count / len(history_df) * 100:.1f}%"
                if len(history_df) else "0.0%",
            )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="📥 Download Analysis History as CSV",
        data=csv_data,
        file_name="election_violence_analysis_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:

    st.info(
        "No analysis history yet. Analyse a post to create your "
        "first record."
    )


# ============================================================
# 10. PREDICTION SUMMARY
# ============================================================

if not history_df.empty:

    st.divider()
    st.subheader("📊 Prediction Summary")

    # Count predictions for every category, including zero counts.
    category_counts = (
        history_df["Predicted category"]
        .value_counts()
        .reindex(CATEGORY_ORDER, fill_value=0)
    )

    total_posts = len(history_df)

    # Display summary metrics.
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Neutral",
            int(category_counts["Neutral"]),
        )

    with col2:
        st.metric(
            "Offensive",
            int(category_counts["Offensive"]),
        )

    with col3:
        st.metric(
            "Hate Speech",
            int(category_counts["Hate Speech"]),
        )

    # Display a bar chart.
    st.subheader("Distribution of Predictions")

    chart_df = pd.DataFrame({
        "Category": CATEGORY_ORDER,
        "Number of predictions": [
            int(category_counts[category])
            for category in CATEGORY_ORDER
        ],
    }).set_index("Category")

    st.bar_chart(
        chart_df,
        y="Number of predictions",
    )

    # Display percentages.
    st.subheader("Prediction Percentages")

    percentage_df = pd.DataFrame({
        "Category": CATEGORY_ORDER,
        "Number of predictions": [
            int(category_counts[category])
            for category in CATEGORY_ORDER
        ],
        "Percentage": [
            round(
                category_counts[category] / total_posts * 100,
                2,
            )
            for category in CATEGORY_ORDER
        ],
    })

    st.dataframe(
        percentage_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "These figures describe model predictions in your saved "
        "history. They are not confirmed cases of hate speech "
        "or election violence."
    )


# ============================================================
# 11. MODEL INFORMATION AND EVALUATION
# ============================================================

st.divider()
st.subheader("About the Model")

st.markdown(
    """
    - **Model:** TF-IDF with Logistic Regression.
    - **Training records:** 5,062.
    - **Held-out test records:** 1,266.
    - **Baseline test accuracy:** 68.64%.
    - **Macro F1-score:** 0.44.

    These evaluation results are preliminary. Performance is weak
    for the Offensive and Hate Speech classes. The separate phrase screen
    is a rule-based prototype, not a validated violence predictor. This
    system does not verify threats or establish that real-world election
    violence has occurred.
    """
)

st.caption(
    "Election Violence Early Detection System | Research prototype"
)
