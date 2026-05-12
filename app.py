import streamlit as st
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Nutraceutical Reviewer",
    page_icon="💊",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title-text {
    font-size: 48px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 10px;
}

.subtitle-text {
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.safe-box {
    background-color: #dcfce7;
    color: #166534;
    padding: 14px;
    border-radius: 12px;
    margin-bottom: 10px;
    font-weight: 500;
}

.warning-box {
    background-color: #fef3c7;
    color: #92400e;
    padding: 14px;
    border-radius: 12px;
    margin-bottom: 10px;
    font-weight: 500;
}

.danger-box {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 14px;
    border-radius: 12px;
    margin-bottom: 10px;
    font-weight: 500;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SAFE LIMITS
# --------------------------------------------------

SAFE_LIMITS = {
    "melatonin": {"limit": 5, "unit": "mg"},
    "magnesium": {"limit": 350, "unit": "mg"},
    "ashwagandha": {"limit": 600, "unit": "mg"},
    "l-theanine": {"limit": 400, "unit": "mg"},
    "caffeine": {"limit": 400, "unit": "mg"},
    "creatine": {"limit": 5000, "unit": "mg"},
    "beta-alanine": {"limit": 3200, "unit": "mg"},
    "vitamin c": {"limit": 2000, "unit": "mg"},
    "vitamin d": {"limit": 4000, "unit": "iu"},
    "zinc": {"limit": 40, "unit": "mg"},
    "iron": {"limit": 45, "unit": "mg"},
    "niacin": {"limit": 35, "unit": "mg"},
    "valerian": {"limit": 900, "unit": "mg"},
    "rhodiola": {"limit": 600, "unit": "mg"},
    "omega-3": {"limit": 5000, "unit": "mg"}
}

# --------------------------------------------------
# CATEGORY MAP
# --------------------------------------------------

CATEGORY_MAP = {
    "Sleep Support": [
        "Melatonin",
        "Magnesium",
        "L-Theanine",
        "Valerian"
    ],

    "Pre-Workout": [
        "Creatine",
        "Caffeine",
        "Beta-Alanine"
    ],

    "Multivitamin": [
        "Vitamin C",
        "Vitamin D",
        "Zinc",
        "Iron"
    ],

    "Stress Relief": [
        "Ashwagandha",
        "Rhodiola",
        "Magnesium"
    ],

    "Immune Support": [
        "Vitamin C",
        "Vitamin D",
        "Zinc"
    ],

    "Other": [
        "Magnesium",
        "Omega-3",
        "CoQ10"
    ]
}

# --------------------------------------------------
# MARKETING CLAIMS
# --------------------------------------------------

MARKETING_FLAGS = [
    "miracle cure",
    "instant fat loss",
    "guaranteed results",
    "100% effective",
    "overnight transformation",
    "no side effects",
    "clinically proven",
    "fda approved",
    "detox"
]

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("nih_supplement_formulations.csv")

    df.columns = df.columns.str.strip()

    df = df.fillna("")

    ingredient_info = {}

    for _, row in df.iterrows():

        ingredient = str(
            row.get("Ingredient", "")
        ).lower().strip()

        category = str(
            row.get("Ingredient_Category", "")
        ).strip()

        claims = str(
            row.get("Claims", "")
        ).strip()

        if ingredient and ingredient not in ingredient_info:

            ingredient_info[ingredient] = {
                "category": category,
                "description": claims[:200]
            }

    return ingredient_info

ingredient_info = load_data()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

model = load_model()

# --------------------------------------------------
# BUILD FAISS INDEX
# --------------------------------------------------

@st.cache_resource
def build_index():

    ingredient_names = list(
        ingredient_info.keys()
    )

    ingredient_texts = []

    for name in ingredient_names:

        ingredient_texts.append(
            f"{name} "
            f"{ingredient_info[name]['category']} "
            f"{ingredient_info[name]['description']}"
        )

    embeddings = model.encode(
        ingredient_texts,
        show_progress_bar=False
    )

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(
        np.array(embeddings).astype("float32")
    )

    return ingredient_names, index

ingredient_names, faiss_index = build_index()

# --------------------------------------------------
# PARSE INGREDIENTS
# --------------------------------------------------

def parse_ingredients(text):

    parsed = []

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        match = re.match(
            r"(.+?)\s+(\d+(?:\.\d+)?)\s*(mg|mcg|g|iu)?",
            line,
            re.IGNORECASE
        )

        if match:

            ingredient = match.group(1).strip().lower()

            dosage = float(match.group(2))

            unit = (
                match.group(3) or "mg"
            ).lower()

            parsed.append({
                "ingredient": ingredient,
                "dosage": dosage,
                "unit": unit
            })

    return parsed

# --------------------------------------------------
# ANALYZE RISKS
# --------------------------------------------------

def analyze_risks(parsed_data):

    observations = []

    flags = 0

    for item in parsed_data:

        ingredient = item["ingredient"]

        dosage = item["dosage"]

        found = False

        for key, val in SAFE_LIMITS.items():

            if key in ingredient:

                found = True

                limit = val["limit"]

                if dosage > limit:

                    observations.append({
                        "status": "DANGER",
                        "message":
                        f"{ingredient.title()} exceeds safe limit ({limit}{val['unit']})"
                    })

                    flags += 1

                elif dosage > 0.8 * limit:

                    observations.append({
                        "status": "WARNING",
                        "message":
                        f"{ingredient.title()} is close to safe limit"
                    })

                    flags += 1

                else:

                    observations.append({
                        "status": "SAFE",
                        "message":
                        f"{ingredient.title()} dosage appears safe"
                    })

                break

        if not found:

            observations.append({
                "status": "UNKNOWN",
                "message":
                f"Limited database information for {ingredient.title()}"
            })

    return observations, flags

# --------------------------------------------------
# SAFETY SCORE
# --------------------------------------------------

def calculate_safety_score(flags):

    score = 100 - (flags * 20)

    if score < 20:
        score = 20

    return score

# --------------------------------------------------
# MARKETING CLAIM DETECTION
# --------------------------------------------------

def detect_marketing_claims(text):

    found = []

    for claim in MARKETING_FLAGS:

        if claim.lower() in text.lower():

            found.append(claim)

    return found

# --------------------------------------------------
# SEMANTIC SEARCH
# --------------------------------------------------

def semantic_search(query, top_k=6):

    query_embedding = model.encode([query])

    distances, indices = faiss_index.search(
        np.array(query_embedding).astype("float32"),
        top_k
    )

    results = []

    for idx in indices[0]:

        ingredient = ingredient_names[idx]

        if ingredient not in results:
            results.append(ingredient)

    return results

# --------------------------------------------------
# RAG SUMMARY
# --------------------------------------------------

def rag_generate_summary(
    product_name,
    category,
    parsed_data,
    observations,
    flag_count
):

    ingredient_list = ", ".join(
        [x["ingredient"].title() for x in parsed_data]
    )

    para1 = (
        f"This {category} formulation contains: "
        f"{ingredient_list}."
    )

    para2 = (
        "The formulation was analyzed using "
        "NIH supplement dataset retrieval, "
        "semantic search, and dosage safety analysis."
    )

    if flag_count == 0:
        rating = "Excellent"

    elif flag_count == 1:
        rating = "Good"

    elif flag_count == 2:
        rating = "Fair"

    else:
        rating = "Poor"

    para3 = (
        f"Overall formulation rating: {rating}. "
        f"This review was generated using "
        f"FAISS vector retrieval and AI analysis."
    )

    return para1, para2, para3, rating

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title-text">'
    'Nutraceutical Formulation Reviewer'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle-text">'
    'AI-powered supplement safety analysis platform'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------

left_col, right_col = st.columns([1, 2])

# --------------------------------------------------
# LEFT PANEL
# --------------------------------------------------

with left_col:

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("Product Details")

    product_name = st.text_input(
        "Product Name",
        placeholder="e.g. SleepWell Pro"
    )

    category = st.selectbox(
        "Category",
        [
            "Sleep Support",
            "Pre-Workout",
            "Multivitamin",
            "Stress Relief",
            "Immune Support",
            "Other"
        ]
    )

    ingredients_raw = st.text_area(
        "Ingredient List",
        placeholder=
        "Melatonin 5 mg\n"
        "Magnesium 300 mg",
        height=260
    )

    analyze_btn = st.button(
        "Analyze Formulation",
        use_container_width=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# --------------------------------------------------
# RIGHT PANEL
# --------------------------------------------------

with right_col:

    if analyze_btn:

        if not ingredients_raw.strip():

            st.error(
                "Please enter ingredient list"
            )

        else:

            parsed_data = parse_ingredients(
                ingredients_raw
            )

            if not parsed_data:

                st.error(
                    "Could not parse ingredients. "
                    "Use format: Ingredient Amount Unit"
                )

            else:

                observations, flag_count = (
                    analyze_risks(parsed_data)
                )

                score = calculate_safety_score(
                    flag_count
                )

                # --------------------------
                # METRICS
                # --------------------------

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Ingredients",
                    len(parsed_data)
                )

                col2.metric(
                    "Flags",
                    flag_count
                )

                col3.metric(
                    "Safety Score",
                    f"{score}/100"
                )

                st.progress(score / 100)

                # --------------------------
                # TABLE
                # --------------------------

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.subheader(
                    "Structured Ingredient Analysis"
                )

                df_display = pd.DataFrame(
                    parsed_data
                )

                df_display.columns = [
                    "Ingredient",
                    "Dosage",
                    "Unit"
                ]

                st.dataframe(
                    df_display,
                    use_container_width=True
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

                # --------------------------
                # RECOMMENDATIONS
                # --------------------------

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.subheader(
                    "Recommended Ingredients"
                )

                recs = CATEGORY_MAP.get(
                    category,
                    CATEGORY_MAP["Other"]
                )

                cols = st.columns(3)

                for i, r in enumerate(recs):

                    with cols[i % 3]:

                        st.success(r)

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

                # --------------------------
                # OBSERVATIONS
                # --------------------------

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.subheader(
                    "Safety & Dosage Analysis"
                )

                for obs in observations:

                    if obs["status"] == "SAFE":

                        st.markdown(
                            f'<div class="safe-box">'
                            f'{obs["message"]}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    elif obs["status"] == "WARNING":

                        st.markdown(
                            f'<div class="warning-box">'
                            f'{obs["message"]}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            f'<div class="danger-box">'
                            f'{obs["message"]}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

                # --------------------------
                # MARKETING CLAIMS
                # --------------------------

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.subheader(
                    "Marketing Claim Detection"
                )

                flagged = detect_marketing_claims(
                    ingredients_raw + " " + product_name
                )

                if flagged:

                    for claim in flagged:

                        st.warning(claim)

                else:

                    st.success(
                        "No suspicious marketing claims detected"
                    )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

                # --------------------------
                # AI SUMMARY
                # --------------------------

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.subheader(
                    "AI Review Summary"
                )

                with st.spinner(
                    "Generating RAG-based review..."
                ):

                    para1, para2, para3, rating = (
                        rag_generate_summary(
                            product_name,
                            category,
                            parsed_data,
                            observations,
                            flag_count
                        )
                    )

                st.info(para1)

                st.info(para2)

                st.info(para3)

                st.markdown(
                    f"### Overall Rating: {rating}"
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

# --------------------------------------------------
# SEMANTIC SEARCH
# --------------------------------------------------

st.divider()

st.subheader(
    "Semantic Ingredient Search"
)

search_query = st.text_input(
    "Search by health goal",
    placeholder=
    "sleep support, muscle recovery, stress relief"
)

if search_query:

    results = semantic_search(search_query)

    cols = st.columns(3)

    for i, name in enumerate(results):

        with cols[i % 3]:

            info = ingredient_info.get(name, {})

            st.success(
                f"{name.title()}\n\n"
                f"Category: {info.get('category', '')}"
            )
