"""Test ALL 40 diseases from the user's symptom table to find failures."""
import os, json, joblib, numpy as np, importlib.util, warnings
warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
classifier    = joblib.load(os.path.join(MODEL_DIR, "disease_classifier.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

with open(os.path.join(MODEL_DIR, "disease_symptoms.json")) as f:
    disease_symptoms = json.load(f)

spec = importlib.util.spec_from_file_location(
    "symptom_mapper",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "services", "symptom_mapper.py")
)
mapper_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapper_mod)
map_symptoms = mapper_mod.map_symptoms

def hybrid(symptoms, top_n=3):
    matched = map_symptoms(symptoms, feature_names)
    # Overlap scoring
    ol_scores = {}
    for disease, syms in disease_symptoms.items():
        ds = set(syms)
        overlap = matched & ds
        if not overlap: continue
        prec = len(overlap) / len(matched)
        rec  = len(overlap) / len(ds)
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        ol_scores[disease] = f1
    # ML scoring
    vec = np.array([1 if f in matched else 0 for f in feature_names], dtype=int).reshape(1,-1)
    proba = classifier.predict_proba(vec)[0]
    ml_scores = {label_encoder.classes_[i]: float(proba[i]) for i in range(len(proba))}
    # Combine
    combined = {}
    for d in set(list(ol_scores.keys()) + list(ml_scores.keys())):
        combined[d] = ol_scores.get(d, 0) * 0.7 + ml_scores.get(d, 0) * 0.3
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"disease": d, "confidence": round(s, 4)} for d, s in ranked], matched

# All 40 diseases from the user's table
tests = [
    ("Vertigo", ["dizziness", "loss of balance", "spinning sensation", "nausea"]),
    ("AIDS", ["weight loss", "fatigue", "fever", "night sweats", "swollen lymph nodes"]),
    ("Acne", ["pimples", "skin rash", "blackheads", "oily skin"]),
    ("Alcoholic Hepatitis", ["jaundice", "abdominal pain", "nausea", "fatigue"]),
    ("Allergy", ["sneezing", "itching", "skin rash", "watery eyes"]),
    ("Arthritis", ["joint pain", "swelling joints", "stiffness", "movement pain"]),
    ("Bronchial Asthma", ["breathlessness", "wheezing", "chest tightness", "cough"]),
    ("Cervical Spondylosis", ["neck pain", "dizziness", "headache", "shoulder pain"]),
    ("Chicken Pox", ["skin rash", "itching", "fever", "fatigue"]),
    ("Chronic Cholestasis", ["itching", "jaundice", "dark urine", "abdominal pain"]),
    ("Common Cold", ["cough", "sneezing", "runny nose", "sore throat"]),
    ("Dengue", ["high fever", "headache", "joint pain", "skin rash"]),
    ("Diabetes", ["excessive hunger", "excessive thirst", "frequent urination", "fatigue"]),
    ("Hemorrhoids (Piles)", ["rectal pain", "bleeding during bowel movements", "itching"]),
    ("Drug Reaction", ["skin rash", "itching", "swelling", "fever"]),
    ("Fungal Infection", ["itching", "skin rash", "redness", "skin peeling"]),
    ("GERD", ["acidity", "heartburn", "chest pain", "indigestion"]),
    ("Gastroenteritis", ["diarrhea", "vomiting", "abdominal pain", "dehydration"]),
    ("Heart Attack", ["chest pain", "breathlessness", "sweating", "nausea"]),
    ("Hepatitis B", ["jaundice", "fatigue", "abdominal pain", "dark urine"]),
    ("Hepatitis A", ["jaundice", "nausea", "vomiting", "fatigue"]),
    ("Hepatitis D", ["jaundice", "abdominal pain", "fatigue", "dark urine"]),
    ("Hepatitis E", ["jaundice", "fever", "nausea", "fatigue"]),
    ("Hypertension", ["headache", "dizziness", "blurred vision"]),
    ("Hyperthyroidism", ["weight loss", "sweating", "rapid heartbeat", "anxiety"]),
    ("Hypoglycemia", ["sweating", "dizziness", "confusion", "shakiness"]),
    ("Hypothyroidism", ["fatigue", "weight gain", "dry skin", "depression"]),
    ("Impetigo", ["skin rash", "blisters", "itching", "redness"]),
    ("Jaundice", ["yellow skin", "yellow eyes", "dark urine", "fatigue"]),
    ("Malaria", ["fever", "chills", "sweating", "headache"]),
    ("Migraine", ["severe headache", "nausea", "sensitivity to light"]),
    ("Osteoarthritis", ["joint pain", "stiffness", "swelling joints"]),
    ("Paralysis (Brain Hemorrhage)", ["weakness", "loss of balance", "speech problems", "headache"]),
    ("Peptic Ulcer Disease", ["abdominal pain", "indigestion", "nausea", "bloating"]),
    ("Pneumonia", ["cough", "fever", "chest pain", "breathlessness"]),
    ("Psoriasis", ["skin rash", "itching", "dry skin", "scaly patches"]),
    ("Tuberculosis", ["persistent cough", "weight loss", "fever", "night sweats"]),
    ("Typhoid", ["high fever", "abdominal pain", "weakness", "headache"]),
    ("Urinary Tract Infection", ["burning urination", "frequent urination", "pelvic pain"]),
    ("Varicose Veins", ["swollen veins", "leg pain", "heaviness in legs"]),
]

print(f"{'EXPECTED':<40} {'PREDICTED #1':<40} {'CONF':>6}  MATCH")
print("=" * 100)

passed = 0
failed = 0
fail_list = []

for expected, symptoms in tests:
    results, matched = hybrid(symptoms, top_n=3)
    top = results[0] if results else {"disease": "NONE", "confidence": 0}
    
    # Flexible matching (dataset has slight name variations)
    exp_lower = expected.lower().strip()
    pred_lower = top["disease"].lower().strip()
    is_match = (
        exp_lower in pred_lower or 
        pred_lower in exp_lower or 
        exp_lower.split()[0] in pred_lower or 
        pred_lower.split()[0] in exp_lower or
        ("hemorrhoids" in exp_lower and "hemmorhoids" in pred_lower) or
        ("hemmorhoids" in exp_lower and "hemorrhoids" in pred_lower)
    )
    
    status = "PASS" if is_match else "FAIL"
    if is_match:
        passed += 1
    else:
        failed += 1
        top3 = [f"{r['disease']}({r['confidence']*100:.0f}%)" for r in results]
        fail_list.append((expected, symptoms, top3, sorted(matched)))
    
    print(f"{expected:<40} {top['disease']:<40} {top['confidence']*100:5.1f}%  {status}")

print(f"\n{'='*100}")
print(f"PASSED: {passed}/40  |  FAILED: {failed}/40")

if fail_list:
    print(f"\n{'='*100}")
    print("FAILED CASES DETAIL:")
    for exp, syms, top3, matched in fail_list:
        print(f"\n  Expected: {exp}")
        print(f"  Symptoms: {syms}")
        print(f"  Matched features: {matched}")
        print(f"  Top 3: {top3}")
