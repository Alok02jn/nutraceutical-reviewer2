import streamlit as st
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------

st.set_page_config(
    page_title="Nutraceutical Reviewer",
    page_icon="💊",
    layout="wide"
)

# -----------------------------------------
# SAFE UPPER LIMITS (NIH/FDA reference values)
# -----------------------------------------

SAFE_LIMITS = {
    "melatonin":      {"limit": 5,      "unit": "mg"},
    "magnesium":      {"limit": 350,    "unit": "mg"},
    "ashwagandha":    {"limit": 600,    "unit": "mg"},
    "l-theanine":     {"limit": 400,    "unit": "mg"},
    "caffeine":       {"limit": 400,    "unit": "mg"},
    "creatine":       {"limit": 5000,   "unit": "mg"},
    "beta-alanine":   {"limit": 3200,   "unit": "mg"},
    "vitamin c":      {"limit": 2000,   "unit": "mg"},
    "vitamin d":      {"limit": 4000,   "unit": "iu"},
    "vitamin d3":     {"limit": 4000,   "unit": "iu"},
    "zinc":           {"limit": 40,     "unit": "mg"},
    "iron":           {"limit": 45,     "unit": "mg"},
    "folic acid":     {"limit": 1000,   "unit": "mcg"},
    "vitamin b12":    {"limit": 3000,   "unit": "mcg"},
    "vitamin b6":     {"limit": 100,    "unit": "mg"},
    "niacin":         {"limit": 35,     "unit": "mg"},
    "calcium":        {"limit": 2500,   "unit": "mg"},
    "valerian root":  {"limit": 900,    "unit": "mg"},
    "valerian":       {"limit": 900,    "unit": "mg"},
    "rhodiola":       {"limit": 600,    "unit": "mg"},
    "chamomile":      {"limit": 1200,   "unit": "mg"},
    "5-htp":          {"limit": 300,    "unit": "mg"},
    "gaba":           {"limit": 750,    "unit": "mg"},
    "coq10":          {"limit": 1200,   "unit": "mg"},
    "vitamin a":      {"limit": 10000,  "unit": "iu"},
    "vitamin e":      {"limit": 1000,   "unit": "mg"},
    "thiamine":       {"limit": 100,    "unit": "mg"},
    "riboflavin":     {"limit": 400,    "unit": "mg"},
    "iodine":         {"limit": 1100,   "unit": "mcg"},
    "selenium":       {"limit": 400,    "unit": "mcg"},
    "passionflower":  {"limit": 800,    "unit": "mg"},
    "turmeric":       {"limit": 1500,   "unit": "mg"},
    "curcumin":       {"limit": 1000,   "unit": "mg"},
    "biotin":         {"limit": 10000,  "unit": "mcg"},
    "l-carnitine":    {"limit": 2000,   "unit": "mg"},
    "glutamine":      {"limit": 14000,  "unit": "mg"},
    "arginine":       {"limit": 9000,   "unit": "mg"},
    "citrulline":     {"limit": 6000,   "unit": "mg"},
    "omega-3":        {"limit": 5000,   "unit": "mg"},
}

# -----------------------------------------
# MARKETING CLAIM PATTERNS
# -----------------------------------------

MARKETING_FLAGS = [
    (r"miracle\s*cure",                       "Miracle cure claim"),
    (r"instant\s*fat\s*loss",                "Instant fat loss claim"),
    (r"guaranteed\s*results",                "Guaranteed results claim"),
    (r"100\s*%\s*(cure|effective)",          "100% effectiveness claim"),
    (r"overnight\s*transformation",          "Overnight transformation claim"),
    (r"no\s*side\s*effects",                "'No side effects' claim"),
    (r"fda[\s\-]?approved",                 "FDA-approved claim (supplements are NOT FDA-approved)"),
    (r"clinically\s*proven",                "Clinically proven claim (requires citation)"),
    (r"reverses?\s*(aging|diabetes|cancer)", "Disease reversal claim"),
    (r"detox(ify|ification)?",              "Detox/cleanse claim"),
]

# -----------------------------------------
# CATEGORY RECOMMENDATIONS
# -----------------------------------------

CATEGORY_MAP = {
    "Sleep Support":  ["melatonin", "magnesium", "valerian", "l-theanine", "gaba", "passionflower"],
    "Pre-Workout":    ["creatine", "caffeine", "beta-alanine", "citrulline", "arginine"],
    "Multivitamin":   ["vitamin c", "vitamin d", "zinc", "iron", "folic acid", "biotin"],
    "Stress Relief":  ["ashwagandha", "rhodiola", "l-theanine", "magnesium"],
    "Immune Support": ["vitamin c", "vitamin d", "zinc", "selenium"],
    "Other":          ["magnesium", "omega-3", "coq10"],
}

# -----------------------------------------
# LOAD DATASET + BUILD KNOWLEDGE CORPUS
#
# ingredient_info — dict for dosage lookups
# nih_corpus      — list of NIH claim chunks
#                   used by the RAG pipeline
# -----------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("nih_supplement_formulations.csv")
    df.columns = df.columns.str.strip()
    df = df.fillna("")

    bad = {
        "nickel", "copper", "chloride", "water",
        "calories", "fat", "sodium", "sugars", "carbohydrate"
    }

    ingredient_info = {}
    nih_corpus = []   # every usable NIH claim row becomes a retrievable chunk

    for _, row in df.iterrows():
        raw = str(row.get("Ingredient", ""))
        key = re.sub(r"[^a-zA-Z0-9\s]", "", raw).strip().lower()
        if not key or key in bad:
            continue

        raw_desc = str(row.get("Claims", ""))
        desc = re.sub(
            r"\[P\d+\]|STRUCTURE/FUNCTION CLAIM|OTHER INGREDIENT.*?CLAIM OR USE"
            r"|NUTRITION-RELATED CLAIM OR USE|Directions:.*",
            "", raw_desc
        ).strip()

        # Build ingredient lookup (first occurrence per ingredient)
        if key not in ingredient_info:
            matched_limit = None
            matched_unit = "mg"
            for limit_key, limit_data in SAFE_LIMITS.items():
                if limit_key in key or key in limit_key:
                    matched_limit = limit_data["limit"]
                    matched_unit = limit_data["unit"]
                    break

            ingredient_info[key] = {
                "category":    str(row.get("Ingredient_Category", "")),
                "description": desc[:200],
                "safe_limit":  matched_limit if matched_limit else 1000,
                "unit":        matched_unit,
            }

        # Add every row with a meaningful claim to RAG corpus
        if desc and len(desc) > 20:
            nih_corpus.append({
                "ingredient": key,
                "category":   str(row.get("Ingredient_Category", "")),
                "claim":      desc[:300],
            })

    return ingredient_info, nih_corpus

ingredient_info, nih_corpus = load_data()

# -----------------------------------------
# LOAD SENTENCE TRANSFORMER (CACHED)
# Core AI model: produces 384-dim embeddings
# -----------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------------------
# BUILD TWO FAISS INDEXES (CACHED)
#
# ingredient_index — for the semantic search feature
# corpus_index     — for RAG summary retrieval
# -----------------------------------------

@st.cache_data
def build_indexes(info_keys, corpus_size):
    # Index 1: ingredient-level (semantic search tab)
    ing_names = list(info_keys)
    ing_texts = [
        f"{n} {ingredient_info[n]['category']} {ingredient_info[n]['description']}"
        for n in ing_names
    ]
    ing_emb = model.encode(ing_texts, show_progress_bar=False)
    ing_idx = faiss.IndexFlatL2(ing_emb.shape[1])
    ing_idx.add(np.array(ing_emb).astype("float32"))

    # Index 2: corpus-level (RAG summary)
    corpus_texts = [
        f"{c['ingredient']} {c['category']} {c['claim']}"
        for c in nih_corpus
    ]
    corp_emb = model.encode(corpus_texts, show_progress_bar=False)
    corp_idx = faiss.IndexFlatL2(corp_emb.shape[1])
    corp_idx.add(np.array(corp_emb).astype("float32"))

    return ing_names, ing_idx, corp_idx

ingredient_names, ingredient_index, corpus_index = build_indexes(
    tuple(ingredient_info.keys()), len(nih_corpus)
)

# -----------------------------------------
# INGREDIENT PARSER
# -----------------------------------------

def parse_ingredients(text):
    parsed = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(
            r"(.+?)\s+(\d+(?:\.\d+)?)\s*(mg|g|mcg|iu|cfu)?",
            line, re.IGNORECASE
        )
        if m:
            name = re.sub(r"[^a-zA-Z0-9\s\-]", "", m.group(1)).strip().lower()
            dosage = float(m.group(2))
            unit = (m.group(3) or "mg").lower()
            parsed.append({"ingredient": name, "dosage": dosage, "unit": unit})
    return parsed

# -----------------------------------------
# RISK ANALYSIS
# -----------------------------------------

def analyze_risks(parsed_data):
    observations = []
    flag_count = 0

    for item in parsed_data:
        ingredient = item["ingredient"]
        dosage = item["dosage"]

        matched = None
        if ingredient in ingredient_info:
            matched = ingredient
        else:
            for key in ingredient_info:
                if ingredient in key or key in ingredient:
                    matched = key
                    break

        if matched:
            safe_limit = ingredient_info[matched]["safe_limit"]
            ratio = dosage / safe_limit if safe_limit else 0

            if ratio > 1.5:
                observations.append({
                    "status": "DANGER",
                    "message": (
                        f"HIGH DOSE ⚠️: {ingredient.title()} at {dosage}{item['unit']} "
                        f"exceeds safe upper limit of "
                        f"{safe_limit}{ingredient_info[matched]['unit']}."
                    ),
                })
                flag_count += 1
            elif ratio > 0.8:
                observations.append({
                    "status": "WARNING",
                    "message": (
                        f"NEAR LIMIT ⚡: {ingredient.title()} at {dosage}{item['unit']} "
                        f"is close to the safe upper limit "
                        f"({safe_limit}{ingredient_info[matched]['unit']})."
                    ),
                })
                flag_count += 1
            else:
                observations.append({
                    "status": "SAFE",
                    "message": (
                        f"ACCEPTABLE ✅: {ingredient.title()} at {dosage}{item['unit']} "
                        f"is within safe range "
                        f"(UL: {safe_limit}{ingredient_info[matched]['unit']})."
                    ),
                })
        else:
            observations.append({
                "status": "UNKNOWN",
                "message": (
                    f"❓ {ingredient.title()}: not in NIH database — "
                    f"manual review recommended."
                ),
            })

    return observations, flag_count

# -----------------------------------------
# SAFETY SCORE
# -----------------------------------------

def calculate_safety_score(flag_count, total):
    if total == 0:
        return 100
    penalty = min(flag_count * 20, 80)
    return max(100 - penalty, 10)

# -----------------------------------------
# MARKETING CLAIM DETECTION
# -----------------------------------------

def detect_marketing_claims(text):
    found = []
    for pattern, label in MARKETING_FLAGS:
        if re.search(pattern, text, re.IGNORECASE):
            found.append(label)
    return found

# -----------------------------------------
# RAG-BASED AI SUMMARY  ← core AI pipeline
#
# Steps:
# 1. Embed a query per ingredient using the
#    sentence-transformer model
# 2. Search corpus_index (FAISS) to retrieve
#    the top-5 most semantically similar NIH
#    claim chunks for each ingredient
# 3. Filter retrieved chunks for relevance
# 4. Assemble retrieved facts into a structured
#    3-paragraph review with quality rating
#
# This is a full Retrieval-Augmented Generation
# (RAG) pipeline — 100% local, zero cost,
# grounded in real NIH dataset knowledge.
# -----------------------------------------

def rag_generate_summary(product_name, category, parsed_data, observations, flag_count):

    # Step 1 & 2: Retrieve top NIH facts per ingredient via FAISS
    retrieved_facts = {}

    for item in parsed_data:
        ing = item["ingredient"]

        # Embed a natural-language query for this ingredient
        query = f"{ing} supplement health benefits effects {category}"
        query_embedding = model.encode([query])

        distances, indices = corpus_index.search(
            np.array(query_embedding).astype("float32"),
            k=5
        )

        facts = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(nih_corpus):
                chunk = nih_corpus[idx]
                claim_text = chunk["claim"].strip()

                # Step 3: Relevance filter
                # Keep if ingredient name overlaps with claim text
                # OR embedding distance is very close (< 0.8)
                ing_words = set(ing.split())
                claim_words = set(claim_text.lower().split())
                if (ing_words & claim_words) or dist < 0.8:
                    if claim_text not in facts:
                        facts.append(claim_text)

        if facts:
            retrieved_facts[ing] = facts[:3]

    # Step 4: Determine quality rating from dosage observations
    danger_list = [o["message"] for o in observations if o["status"] == "DANGER"]
    warning_list = [o["message"] for o in observations if o["status"] == "WARNING"]

    if flag_count == 0:
        rating = "Excellent"
        rating_reason = (
            "All ingredients are within established safe upper limits "
            "with no dosage concerns identified."
        )
    elif flag_count == 1 and not danger_list:
        rating = "Good"
        rating_reason = (
            "One ingredient is near its safe upper limit "
            "but no critical violations were detected."
        )
    elif len(danger_list) == 1:
        rating = "Fair"
        rating_reason = (
            "One ingredient exceeds the recommended safe upper limit "
            "and requires attention before use."
        )
    else:
        rating = "Poor"
        rating_reason = (
            f"{len(danger_list)} ingredient(s) exceed safe upper limits. "
            "This formulation requires significant revision."
        )

    # Step 4a: Paragraph 1 — Formulation purpose
    # Built from retrieved NIH facts, not hardcoded text
    purpose_lines = []
    for item in parsed_data:
        ing = item["ingredient"]
        if ing in retrieved_facts and retrieved_facts[ing]:
            sentence = retrieved_facts[ing][0].split(".")[0].strip()
            if sentence:
                purpose_lines.append(f"{ing.title()} ({sentence.lower()})")

    if purpose_lines:
        para1 = (
            f"This {category} formulation \"{product_name or 'Unnamed'}\" "
            f"contains {len(parsed_data)} active ingredient(s). "
            f"Based on NIH supplement label data retrieved via vector search, "
            f"the key components include: {'; '.join(purpose_lines)}."
        )
    else:
        para1 = (
            f"This {category} formulation contains {len(parsed_data)} ingredient(s): "
            f"{', '.join(i['ingredient'].title() for i in parsed_data)}. "
            f"These were cross-referenced against the NIH supplement label database."
        )

    # Step 4b: Paragraph 2 — Synergy analysis
    # Checks category combinations present in the formulation
    categories_present = set(
        ingredient_info[i["ingredient"]]["category"]
        for i in parsed_data
        if i["ingredient"] in ingredient_info
    )

    synergy_notes = []
    if "botanical" in categories_present and "mineral" in categories_present:
        synergy_notes.append(
            "botanical adaptogens and minerals may offer complementary "
            "stress-modulating and physiological support"
        )
    if "amino acid" in categories_present and "non-nutrient/non-botanical" in categories_present:
        synergy_notes.append(
            "amino acid and non-botanical components may synergize "
            "for enhanced bioavailability"
        )
    if "vitamin" in categories_present and "mineral" in categories_present:
        synergy_notes.append(
            "vitamins and minerals together are known to enhance mutual "
            "absorption and metabolic cofactor activity"
        )

    extra_facts = []
    for item in parsed_data[:3]:
        ing = item["ingredient"]
        if ing in retrieved_facts and len(retrieved_facts[ing]) > 1:
            extra_facts.append(
                retrieved_facts[ing][1].split(".")[0].strip().lower()
            )

    if synergy_notes or extra_facts:
        parts = []
        if synergy_notes:
            parts.append(f"Regarding ingredient synergy: {'; '.join(synergy_notes)}.")
        if extra_facts:
            parts.append(
                f"NIH records further indicate: {'; '.join(extra_facts)}."
            )
        para2 = " ".join(parts)
    else:
        para2 = (
            "The ingredients span multiple functional categories. "
            "No specific synergistic interactions were identified in "
            "the NIH corpus for this combination — effects should be "
            "evaluated per ingredient."
        )

    # Step 4c: Paragraph 3 — Dosage concerns + rating
    concern_parts = []
    if danger_list:
        concern_parts.append(f"Critical issues: {'; '.join(danger_list)}.")
    if warning_list:
        concern_parts.append(f"Warnings: {'; '.join(warning_list)}.")
    if not concern_parts:
        concern_parts.append(
            "No dosage concerns detected — all ingredients are within "
            "established safe upper limits."
        )

    para3 = (
        f"Dosage analysis: {' '.join(concern_parts)} "
        f"Overall rating: {rating}. {rating_reason} "
        f"This review was generated by a RAG pipeline using the NIH "
        f"Dietary Supplement Label Database and is not a substitute "
        f"for professional medical advice."
    )

    return para1, para2, para3, rating, retrieved_facts

# -----------------------------------------
# SEMANTIC SEARCH
# -----------------------------------------

def semantic_search(query, top_k=6):

    q_emb = model.encode([query])

    distances, indices = ingredient_index.search(

        np.array(q_emb).astype("float32"),

        top_k * 3
    )

    results = []

    bad_terms = {

        "water",
        "nickel",
        "copper",
        "fat",
        "sugars",
        "carbohydrate",
        "calories",
        "chloride"
    }

    for i in indices[0]:

        ingredient = ingredient_names[i]

        if ingredient.lower() not in bad_terms:

            if ingredient not in results:

                results.append(ingredient)

        if len(results) >= top_k:

            break

    return results
# -----------------------------------------

# Updated Nutraceutical Reviewer App (Modern UI Version)

Use this updated UI section to replace your current `# UI` section in the app code you uploaded. fileciteturn11file0

```python
# -----------------------------------------
# MODERN UI
# -----------------------------------------

# Custom CSS
st.markdown(
    """
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
        padding: 14px;
        border-radius: 12px;
        color: #166534;
        font-weight: 500;
        margin-bottom: 10px;
    }

    .warning-box {
        background-color: #fef3c7;
        padding: 14px;
        border-radius: 12px;
        color: #92400e;
        font-weight: 500;
        margin-bottom: 10px;
    }

    .danger-box {
        background-color: #fee2e2;
        padding: 14px;
        border-radius: 12px;
        color: #991b1b;
        font-weight: 500;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------
# HEADER
# -----------------------------------------

st.markdown(
    '<div class="title-text">Nutraceutical Formulation Reviewer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle-text">AI-powered supplement safety and formulation analysis platform using RAG + NIH dataset retrieval</div>',
    unsafe_allow_html=True
)

# -----------------------------------------
# LAYOUT
# -----------------------------------------

left_col, right_col = st.columns([1, 2])

# -----------------------------------------
# LEFT PANEL
# -----------------------------------------

with left_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

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
        placeholder="Melatonin 5 mg\nMagnesium 300 mg",
        height=260
    )

    analyze_btn = st.button(
        "Analyze Formulation",
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------
# RIGHT PANEL
# -----------------------------------------

with right_col:

    if analyze_btn:

        if not ingredients_raw.strip():

            st.error("Please enter ingredient list")

        else:

            parsed_data = parse_ingredients(ingredients_raw)

            if not parsed_data:

                st.error(
                    "Could not parse ingredients. Use format: Ingredient Amount Unit"
                )

            else:

                observations, flag_count = analyze_risks(parsed_data)

                score = calculate_safety_score(
                    flag_count,
                    len(parsed_data)
                )

                # -----------------------------
                # METRICS
                # -----------------------------

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

                # -----------------------------
                # STRUCTURED TABLE
                # -----------------------------

                st.markdown('<div class="card">', unsafe_allow_html=True)

                st.subheader("Structured Ingredient Analysis")

                df_display = pd.DataFrame(parsed_data)
                df_display.columns = [
                    "Ingredient",
                    "Dosage",
                    "Unit"
                ]

                st.dataframe(
                    df_display,
                    use_container_width=True
                )

                st.markdown('</div>', unsafe_allow_html=True)

                # -----------------------------
                # RECOMMENDATIONS
                # -----------------------------

                st.markdown('<div class="card">', unsafe_allow_html=True)

                st.subheader("Recommended Ingredients")

                recs = CATEGORY_MAP.get(
                    category,
                    CATEGORY_MAP["Other"]
                )

                rec_cols = st.columns(3)

                for i, r in enumerate(recs):

                    with rec_cols[i % 3]:

                        st.success(r.title())

                st.markdown('</div>', unsafe_allow_html=True)

                # -----------------------------
                # RISK ANALYSIS
                # -----------------------------

                st.markdown('<div class="card">', unsafe_allow_html=True)

                st.subheader("Safety & Dosage Analysis")

                for obs in observations:

                    if obs["status"] == "SAFE":

                        st.markdown(
                            f'<div class="safe-box">{obs["message"]}</div>',
                            unsafe_allow_html=True
                        )

                    elif obs["status"] == "WARNING":

                        st.markdown(
                            f'<div class="warning-box">{obs["message"]}</div>',
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            f'<div class="danger-box">{obs["message"]}</div>',
                            unsafe_allow_html=True
                        )

                st.markdown('</div>', unsafe_allow_html=True)

                # -----------------------------
                # MARKETING CLAIMS
                # -----------------------------

                st.markdown('<div class="card">', unsafe_allow_html=True)

                st.subheader("Marketing Claim Detection")

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

                st.markdown('</div>', unsafe_allow_html=True)

                # -----------------------------
                # AI SUMMARY
                # -----------------------------

                st.markdown('<div class="card">', unsafe_allow_html=True)

                st.subheader("AI Review Summary")

                with st.spinner("Generating RAG-based review..."):

                    para1, para2, para3, rating, retrieved_facts = (
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

                st.markdown('</div>', unsafe_allow_html=True)

                # -----------------------------
                # NIH FACTS
                # -----------------------------

                with st.expander(
                    "Retrieved NIH Supporting Facts"
                ):

                    if retrieved_facts:

                        for ing, facts in retrieved_facts.items():

                            st.markdown(
                                f"#### {ing.title()}"
                            )

                            for f in facts:

                                st.caption(f)

                    else:

                        st.write(
                            "No NIH retrieval facts available"
                        )

# -----------------------------------------
# SEMANTIC SEARCH
# -----------------------------------------

st.divider()

st.subheader("Semantic Ingredient Search")

search_query = st.text_input(
    "Search by health goal",
    placeholder="sleep support, muscle recovery, stress relief"
)

if search_query:

    results = semantic_search(search_query)

    search_cols = st.columns(3)

    for i, name in enumerate(results):

        with search_cols[i % 3]:

            info = ingredient_info.get(name, {})

            st.success(
                f"{name.title()}\n\n"
                f"Category: {info.get('category', '')}\n\n"
                f"Safe Limit: {info.get('safe_limit', '?')} {info.get('unit', '')}"
            )
```

# IMPORTANT

Replace ONLY your current `# UI` section with this upgraded version.

Do NOT remove:

* parse_ingredients()
* analyze_risks()
* semantic_search()
* rag_generate_summary()
* FAISS code
* dataset loading code
* SAFE_LIMITS
