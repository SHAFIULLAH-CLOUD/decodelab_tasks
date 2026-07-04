"""
================================================================================
 PROJECT 3 CAPSTONE — TECH STACK RECOMMENDER
 DecodeLabs | Artificial Intelligence Industrial Training Kit
================================================================================

WHAT THIS SCRIPT DOES (matches the brief exactly):
    Maps a user's raw skills + career interests -> a ranked list of the
    Top-3 most relevant job roles, using pure Content-Based Filtering.

    No collaborative filtering, no historical user data, no neural nets.
    Just: Vector Mapping -> TF-IDF Weighting -> Cosine Similarity -> Ranking.

THE IPO ARCHITECTURE (as taught in the deck):
    INPUT   (User State)      -> user types 3+ skills
    PROCESS (Similarity Logic)-> TF-IDF vectors + Cosine Similarity
    OUTPUT  (Top-N List)      -> Top 3 ranked job roles

THE 4-STEP PIPELINE:
    1. Ingestion -> read raw_skills.csv, capture user skills
    2. Scoring   -> TF-IDF weight everything, compute cosine similarity
    3. Sorting   -> sort job roles by descending similarity score
    4. Filtering -> cut the list down to Top-3 (prevents choice overload)

Why TF-IDF instead of plain binary vectors (1s and 0s)?
    Binary vectors treat "Python" (common) and "Kubernetes" (specific) as
    EQUALLY important. That's the flaw shown on slide 10 (3 != 3 problem).
    TF-IDF fixes this: it rewards specific/rare skills and downweights
    generic ones that appear in almost every job role.

Why Cosine Similarity instead of Euclidean Distance?
    Euclidean distance cares about magnitude (how many total skills a role
    lists). Cosine similarity only cares about ORIENTATION/DIRECTION —
    i.e. whether the user's skill profile *points the same way* as a job
    role's skill profile, regardless of how many skills each one lists.
    That's why it's the industry standard for this kind of matching.

NOTE ON IMPLEMENTATION:
    TF-IDF and Cosine Similarity are built here from raw math (no sklearn),
    on purpose — the whole point of this milestone is proving you
    understand the "logic skeleton" underneath the libraries, not just
    calling .fit_transform() and hoping for the best.
================================================================================
"""

import csv
import math
from collections import Counter


# --------------------------------------------------------------------------
# STEP 1a: INGESTION — Load the job-role dataset (our "items")
# --------------------------------------------------------------------------
def load_job_roles(csv_path):
    """
    Reads raw_skills.csv and returns a dict:
        { "Data Scientist": ["python", "sql", "machine learning", ...], ... }
    All skills are lowercased and stripped so "Python" and "python" match.
    """
    job_roles = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = row["job_role"].strip()
            skills = [s.strip().lower() for s in row["skills"].split(",")]
            job_roles[role] = skills
    return job_roles


# --------------------------------------------------------------------------
# STEP 1b: INGESTION — Capture the user's raw skill input
# --------------------------------------------------------------------------
def get_user_skills():
    """
    Per the spec: "your script must accept a minimum of three user inputs
    to ensure sufficient data density for accurate matching."
    """
    print("=" * 60)
    print(" TECH STACK RECOMMENDER — DecodeLabs Matchmaker Engine")
    print("=" * 60)
    print("Enter at least 3 skills or interests (comma-separated).")
    print('Example: Python, Cloud Computing, Automation\n')

    while True:
        raw = input("Your skills: ").strip()
        skills = [s.strip().lower() for s in raw.split(",") if s.strip()]
        if len(skills) >= 3:
            return skills
        print(f"⚠ You entered {len(skills)} skill(s). Minimum is 3. Try again.\n")


# --------------------------------------------------------------------------
# STEP 2a: PROCESS — Build the shared vocabulary space
# --------------------------------------------------------------------------
def build_vocabulary(job_roles, user_skills):
    """
    Every skill mentioned anywhere (across all job roles AND the user)
    becomes one dimension in our vector space. This is the "shared
    vocabulary" from slide 9 — item features and user features MUST map
    to the exact same vocabulary or the similarity math breaks.
    """
    vocab = set(user_skills)
    for skills in job_roles.values():
        vocab.update(skills)
    return sorted(vocab)  # sorted for a stable, repeatable ordering


# --------------------------------------------------------------------------
# STEP 2b: PROCESS — Term Frequency (TF)
# --------------------------------------------------------------------------
def compute_tf(document_terms):
    """
    TF = (count of term t in document) / (total terms in document)
    "document" here is either a job role's skill list OR the user's skill list.
    """
    total_terms = len(document_terms)
    counts = Counter(document_terms)
    return {term: count / total_terms for term, count in counts.items()}


# --------------------------------------------------------------------------
# STEP 2c: PROCESS — Inverse Document Frequency (IDF)
# --------------------------------------------------------------------------
def compute_idf(all_documents, vocab):
    """
    IDF = log( total_documents / documents_containing_term )
    Penalizes skills that appear in almost every job role (e.g. "python")
    and rewards skills that are rare/specific (e.g. "solidity", "cryptography").
    The log applies the dampening effect described on slide 12.
    """
    total_docs = len(all_documents)
    idf = {}
    for term in vocab:
        docs_with_term = sum(1 for doc in all_documents if term in doc)
        # +1 smoothing so we never divide by zero for a term that exists in vocab
        idf[term] = math.log(total_docs / (docs_with_term if docs_with_term else 1))
    return idf


# --------------------------------------------------------------------------
# STEP 2d: PROCESS — Convert a document into a weighted TF-IDF vector
# --------------------------------------------------------------------------
def tfidf_vector(document_terms, vocab, idf):
    """
    Combines TF and IDF: weight(term) = TF(term) * IDF(term)
    Returns a full-length vector aligned to the shared vocabulary, so every
    vector (job role or user) has the exact same dimensions in the exact
    same order — required for cosine similarity to work correctly.
    """
    tf = compute_tf(document_terms)
    return [tf.get(term, 0.0) * idf.get(term, 0.0) for term in vocab]


# --------------------------------------------------------------------------
# STEP 2e: PROCESS — Cosine Similarity (the "Similarity Engine")
# --------------------------------------------------------------------------
def cosine_similarity(vec_a, vec_b):
    """
    cos(theta) = (A . B) / (||A|| * ||B||)

    Measures the ANGLE between two vectors, not their magnitude — so a
    user with 3 skills can still score highly against a job role with 10
    listed skills, as long as they point in the same "direction."

    Score interpretation:
        1  -> perfectly aligned (great match)
        0  -> no shared characteristics
       -1  -> impossible here since TF-IDF values are non-negative,
              so our real range is naturally 0 to 1.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        # Cold Start Problem (slide 20): a zero vector can't be compared.
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# --------------------------------------------------------------------------
# STEP 2 (full) + STEP 3 + STEP 4: SCORING -> SORTING -> FILTERING
# --------------------------------------------------------------------------
def recommend_top_n(user_skills, job_roles, top_n=3):
    """
    Runs the full 4-step pipeline:
        1. Ingestion already happened (job_roles, user_skills passed in)
        2. Scoring   -> TF-IDF + cosine similarity for every job role
        3. Sorting   -> descending order by score
        4. Filtering -> truncate to top_n
    """
    all_documents = list(job_roles.values()) + [user_skills]
    vocab = build_vocabulary(job_roles, user_skills)
    idf = compute_idf(all_documents, vocab)

    user_vector = tfidf_vector(user_skills, vocab, idf)

    # --- Step 2: Scoring ---
    scores = []
    for role, skills in job_roles.items():
        role_vector = tfidf_vector(skills, vocab, idf)
        score = cosine_similarity(user_vector, role_vector)
        scores.append((role, score))

    # --- Step 3: Sorting ---
    scores.sort(key=lambda pair: pair[1], reverse=True)

    # --- Step 4: Filtering ---
    return scores[:top_n]


# --------------------------------------------------------------------------
# MAIN — ties the whole IPO pipeline together
# --------------------------------------------------------------------------
def main():
    job_roles = load_job_roles("raw_skills.csv")
    user_skills = get_user_skills()

    top_matches = recommend_top_n(user_skills, job_roles, top_n=3)

    print("\n" + "=" * 60)
    print(" TOP 3 RECOMMENDED CAREER PATHS")
    print("=" * 60)

    if all(score == 0 for _, score in top_matches):
        # Cold start fallback (slide 21): no overlap at all with any role
        print("No strong matches found for those skills.")
        print("Falling back to trending / most in-demand roles:")
        fallback = ["Data Scientist", "DevOps Engineer", "Full Stack Developer"]
        for i, role in enumerate(fallback, start=1):
            print(f"  {i}. {role}  (trending fallback)")
    else:
        for i, (role, score) in enumerate(top_matches, start=1):
            match_pct = round(score * 100, 1)
            print(f"  {i}. {role:<28} — {match_pct}% match")
            print(f"     Core skills: {', '.join(job_roles[role])}")

    print("=" * 60)


if __name__ == "__main__":
    main()
