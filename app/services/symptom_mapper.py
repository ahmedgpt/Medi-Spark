"""
Symptom Mapper — translates natural-language symptom input into exact
feature names that the trained ML model recognises.

The Kaggle Symptom-Disease dataset uses 132 very specific column names
(e.g. 'high_fever' not 'fever', 'muscle_pain' not 'body aches').
Users type everyday descriptions, so this module bridges the gap.

Flow:
  1. Normalise user text  →  lowercase, strip, underscores
  2. Try exact match against feature list
  3. Try synonym / alias lookup
  4. Fuzzy-match fallback (difflib)
"""

import difflib
from typing import List, Set

# ── Synonym dictionary ──────────────────────────────────────────────────────
# Keys   = things a user might type (normalised to lowercase + underscores)
# Values = list of model feature names to activate
#
# IMPORTANT: every value MUST be a valid column from the Kaggle dataset's
# 132 symptom features stored in  models/feature_names.pkl
SYMPTOM_SYNONYMS = {
    # ── Fever variants ────────────────────────────────────────────────────
    "fever":                    ["high_fever"],
    "high_temperature":         ["high_fever"],
    "temperature":              ["high_fever", "mild_fever"],
    "low_grade_fever":          ["mild_fever"],
    "slight_fever":             ["mild_fever"],
    "burning_up":               ["high_fever"],

    # ── Pain ──────────────────────────────────────────────────────────────
    "body_aches":               ["muscle_pain", "joint_pain"],
    "body_pain":                ["muscle_pain", "joint_pain"],
    "bodyache":                 ["muscle_pain", "joint_pain"],
    "aches":                    ["muscle_pain", "joint_pain"],
    "pain":                     ["muscle_pain"],
    "muscle_ache":              ["muscle_pain"],
    "muscle_aches":             ["muscle_pain"],
    "sore_muscles":             ["muscle_pain"],
    "sore_body":                ["muscle_pain", "joint_pain"],
    "stomach_ache":             ["stomach_pain", "abdominal_pain"],
    "tummy_pain":               ["stomach_pain", "abdominal_pain"],
    "tummy_ache":               ["stomach_pain", "abdominal_pain"],
    "abdominal_cramps":         ["abdominal_pain", "cramps"],
    "stomach_cramps":           ["stomach_pain", "cramps"],
    "lower_back_pain":          ["back_pain"],
    "shoulder_pain":            ["neck_pain"],
    "movement_pain":            ["movement_stiffness", "joint_pain"],
    "rectal_pain":              ["pain_in_anal_region"],
    "pelvic_pain":              ["abdominal_pain", "bladder_discomfort"],

    # ── Head / Neuro ──────────────────────────────────────────────────────
    "head_pain":                ["headache"],
    "severe_headache":          ["headache"],
    "migraine":                 ["headache"],
    "sensitivity_to_light":     ["visual_disturbances"],
    "light_sensitivity":        ["visual_disturbances"],
    "photophobia":              ["visual_disturbances"],
    "confusion":                ["altered_sensorium", "slurred_speech"],
    "disoriented":              ["altered_sensorium"],

    # ── Respiratory ───────────────────────────────────────────────────────
    "sore_throat":              ["throat_irritation", "patches_in_throat"],
    "throat_pain":              ["throat_irritation"],
    "scratchy_throat":          ["throat_irritation"],
    "shortness_of_breath":      ["breathlessness"],
    "difficulty_breathing":     ["breathlessness"],
    "breathing_difficulty":     ["breathlessness"],
    "trouble_breathing":        ["breathlessness"],
    "wheezing":                 ["breathlessness"],
    "chest_tightness":          ["breathlessness"],
    "tight_chest":              ["breathlessness"],
    "persistent_cough":         ["cough", "phlegm"],
    "nasal_congestion":         ["congestion"],
    "blocked_nose":             ["congestion"],
    "stuffy_nose":              ["congestion"],
    "sneezing":                 ["continuous_sneezing"],
    "continuous_cough":         ["cough", "phlegm"],
    "dry_cough":                ["cough", "phlegm"],
    "wet_cough":                ["cough", "phlegm"],
    "persistent_cough":         ["cough", "phlegm", "blood_in_sputum"],
    "cough":                    ["cough", "phlegm", "rusty_sputum"],
    "mucus":                    ["phlegm", "mucoid_sputum"],
    "sputum":                   ["phlegm", "mucoid_sputum"],
    "cold":                     ["continuous_sneezing", "runny_nose", "congestion"],
    "common_cold":              ["continuous_sneezing", "runny_nose", "congestion"],

    # ── Skin ──────────────────────────────────────────────────────────────
    "rash":                     ["skin_rash", "red_spots_over_body"],
    "skin_rash":                ["skin_rash", "red_spots_over_body"],
    "skin_eruptions":           ["nodal_skin_eruptions"],
    "hives":                    ["skin_rash"],
    "pimples":                  ["pus_filled_pimples", "blackheads"],
    "acne_breakout":            ["pus_filled_pimples", "blackheads"],
    "blisters":                 ["blister"],
    "peeling_skin":             ["skin_peeling", "dischromic__patches"],
    "skin_peeling":             ["skin_peeling", "dischromic__patches"],
    "yellow_skin":              ["yellowish_skin"],
    "red_spots":                ["red_spots_over_body"],
    "redness":                  ["red_sore_around_nose", "skin_rash"],
    "dry_skin":                 ["skin_peeling", "brittle_nails", "cold_hands_and_feets"],
    "oily_skin":                ["pus_filled_pimples", "blackheads"],
    "scaly_patches":            ["silver_like_dusting", "skin_peeling"],
    "scales":                   ["silver_like_dusting"],
    "swelling":                 ["swelling_joints", "swelled_lymph_nodes"],

    # ── GI ────────────────────────────────────────────────────────────────
    "diarrhea":                 ["diarrhoea"],
    "diarrhoea_loose_stools":   ["diarrhoea"],
    "loose_motions":            ["diarrhoea"],
    "loose_stools":             ["diarrhoea"],
    "throwing_up":              ["vomiting"],
    "puking":                   ["vomiting"],
    "feeling_sick":             ["nausea"],
    "nauseous":                 ["nausea"],
    "gas":                      ["passage_of_gases", "acidity"],
    "bloating":                 ["passage_of_gases", "swelling_of_stomach"],
    "acid_reflux":              ["acidity", "indigestion"],
    "heartburn":                ["acidity"],
    "indigestion_problems":     ["indigestion", "acidity"],
    "no_appetite":              ["loss_of_appetite"],
    "loss_of_hunger":           ["loss_of_appetite"],
    "not_hungry":               ["loss_of_appetite"],
    "blood_in_stool":           ["bloody_stool"],
    "bleeding_during_bowel_movements": ["bloody_stool", "irritation_in_anus"],

    # ── Eye ───────────────────────────────────────────────────────────────
    "blurred_vision":           ["blurred_and_distorted_vision"],
    "blurry_vision":            ["blurred_and_distorted_vision"],
    "vision_problems":          ["blurred_and_distorted_vision", "visual_disturbances"],
    "red_eyes":                 ["redness_of_eyes"],
    "watery_eyes":              ["watering_from_eyes"],
    "yellow_eyes":              ["yellowish_skin"],
    "eye_pain":                 ["pain_behind_the_eyes"],

    # ── General / Systemic ────────────────────────────────────────────────
    "tiredness":                ["fatigue", "lethargy"],
    "exhaustion":               ["fatigue", "lethargy"],
    "weakness":                 ["fatigue", "muscle_weakness"],
    "feeling_weak":             ["fatigue", "muscle_weakness"],
    "malaise_general":          ["malaise"],
    "feeling_unwell":           ["malaise", "fatigue"],
    "sweats":                   ["sweating"],
    "night_sweats":             ["sweating", "high_fever"],
    "cold_sweat":               ["sweating", "chills"],
    "trembling":                ["shivering"],
    "shaking":                  ["shivering"],
    "shakiness":                ["shivering", "palpitations", "anxiety"],
    "weight_decrease":          ["weight_loss", "muscle_wasting"],
    "losing_weight":            ["weight_loss", "muscle_wasting"],
    "weight_loss":              ["weight_loss", "muscle_wasting"],
    "weight_increase":          ["weight_gain"],
    "gaining_weight":           ["weight_gain"],
    "thirsty":                  ["irregular_sugar_level"],
    "excessive_thirst":         ["irregular_sugar_level", "excessive_hunger"],
    "dehydrated":               ["dehydration"],
    "anxious":                  ["anxiety", "restlessness"],
    "nervousness":              ["anxiety", "restlessness"],
    "anxiety":                  ["anxiety", "restlessness", "irritability"],
    "feeling_depressed":        ["depression"],
    "sadness":                  ["depression"],
    "irritable":                ["irritability"],
    "mood_changes":             ["mood_swings"],
    "restless":                 ["restlessness"],
    "sleepless":                ["restlessness"],
    "cant_concentrate":         ["lack_of_concentration"],
    "brain_fog":                ["lack_of_concentration"],

    # ── Urinary ───────────────────────────────────────────────────────────
    "frequent_urination":       ["polyuria"],
    "burning_urination":        ["burning_micturition"],
    "painful_urination":        ["burning_micturition"],
    "blood_in_urine":           ["spotting__urination"],
    "smelly_urine":             ["foul_smell_of_urine"],
    "dark_urine_color":         ["dark_urine"],
    "yellow_pee":               ["yellow_urine"],
    "bladder_pain":             ["bladder_discomfort"],

    # ── Musculoskeletal ───────────────────────────────────────────────────
    "stiff_joints":             ["swelling_joints", "movement_stiffness"],
    "stiffness":                ["movement_stiffness", "stiff_neck"],
    "swollen_joints":           ["swelling_joints"],
    "difficulty_walking":       ["painful_walking", "weakness_in_limbs"],
    "leg_pain":                 ["knee_pain", "painful_walking"],
    "hip_pain":                 ["hip_joint_pain"],
    "weak_limbs":               ["weakness_in_limbs"],
    "arm_weakness":             ["weakness_in_limbs"],
    "leg_weakness":             ["weakness_in_limbs"],
    "stiff_body":               ["movement_stiffness"],
    "neck_stiffness":           ["stiff_neck"],
    "heaviness_in_legs":        ["swollen_legs", "prominent_veins_on_calf"],
    "swollen_veins":            ["swollen_blood_vessels", "prominent_veins_on_calf"],

    # ── Cardiac ───────────────────────────────────────────────────────────
    "rapid_heartbeat":          ["fast_heart_rate"],
    "heart_racing":             ["fast_heart_rate", "palpitations"],
    "heart_pounding":           ["palpitations"],
    "irregular_heartbeat":      ["palpitations"],

    # ── Liver / Jaundice ──────────────────────────────────────────────────
    "jaundice":                 ["yellowish_skin", "dark_urine"],
    "jaundice_symptoms":        ["yellowish_skin", "dark_urine"],
    "liver_failure":            ["acute_liver_failure"],
    "liver_problems":           ["acute_liver_failure", "yellowish_skin"],

    # ── Misc ──────────────────────────────────────────────────────────────
    "swollen_lymph_nodes":      ["swelled_lymph_nodes", "patches_in_throat"],
    "lymph_nodes":              ["swelled_lymph_nodes"],
    "glands_swollen":           ["swelled_lymph_nodes"],
    "sinus":                    ["sinus_pressure", "congestion"],
    "sinus_pain":               ["sinus_pressure"],
    "nose_bleed":               ["runny_nose"],
    "vertigo":                  ["dizziness", "spinning_movements", "loss_of_balance"],
    "spinning_sensation":       ["spinning_movements", "dizziness"],
    "spinning":                 ["spinning_movements"],
    "balance_issues":           ["loss_of_balance", "unsteadiness"],
    "slurred_words":            ["slurred_speech"],
    "speech_problems":          ["slurred_speech", "altered_sensorium"],
    "one_sided_weakness":       ["weakness_of_one_body_side"],
    "paralysis":                ["weakness_of_one_body_side"],
    "numbness":                 ["weakness_of_one_body_side"],
    "loss_of_taste":            ["loss_of_smell"],
    "loss_of_taste/smell":      ["loss_of_smell"],
    "loss_of_smell_taste":      ["loss_of_smell"],
    "anosmia":                  ["loss_of_smell"],
    "cant_smell":               ["loss_of_smell"],
    "cant_taste":               ["loss_of_smell"],
    "swollen_feet":             ["swollen_legs", "swollen_extremeties"],
    "swollen_ankles":           ["swollen_legs", "swollen_extremeties"],
    "puffy_face":               ["puffy_face_and_eyes"],
    "facial_swelling":          ["puffy_face_and_eyes"],
    "cold_feet":                ["cold_hands_and_feets"],
    "cold_hands":               ["cold_hands_and_feets"],
    "sugar_problems":           ["irregular_sugar_level"],
    "high_blood_sugar":         ["irregular_sugar_level"],
    "low_blood_sugar":          ["irregular_sugar_level"],
    "thyroid_problems":         ["enlarged_thyroid"],
    "swollen_thyroid":          ["enlarged_thyroid"],
    "hungry_all_the_time":      ["excessive_hunger", "increased_appetite"],
    "overeating":               ["excessive_hunger", "increased_appetite"],
    "varicose":                 ["swollen_blood_vessels", "prominent_veins_on_calf"],
    "veins_showing":            ["prominent_veins_on_calf", "swollen_blood_vessels"],
    "bruises":                  ["bruising"],
    "nail_problems":            ["brittle_nails", "small_dents_in_nails"],
    "muscle_cramps":            ["cramps", "muscle_pain"],
    "spasms":                   ["cramps"],
    "confusion":                ["altered_sensorium"],
    "disoriented":              ["altered_sensorium"],
    "unconscious":              ["coma"],
    "fainting":                 ["coma", "dizziness"],
    "internal_bleeding":        ["stomach_bleeding"],
    "blood_vomit":              ["stomach_bleeding"],
    "swollen_belly":            ["distention_of_abdomen", "swelling_of_stomach"],
    "alcohol_history":          ["history_of_alcohol_consumption"],
    "drinking_history":         ["history_of_alcohol_consumption"],
    "coughing_blood":           ["blood_in_sputum"],
    "anal_pain":                ["pain_in_anal_region"],
    "piles":                    ["pain_in_anal_region", "bloody_stool", "irritation_in_anus"],
    "hemorrhoids":              ["pain_in_anal_region", "bloody_stool", "irritation_in_anus"],
    "constipated":              ["constipation"],
    "irregular_periods":        ["abnormal_menstruation"],
    "period_problems":          ["abnormal_menstruation"],
    "menstrual_problems":       ["abnormal_menstruation"],
    "skin_patches":             ["dischromic__patches"],
    "discolored_skin":          ["dischromic__patches"],
    "blood_transfusion":        ["receiving_blood_transfusion"],
    "unsterile_injection":      ["receiving_unsterile_injections"],
    "red_nose":                 ["red_sore_around_nose"],
    "nose_sore":                ["red_sore_around_nose"],
    "yellow_scabs":             ["yellow_crust_ooze"],
    "obesity_overweight":       ["obesity"],
    "overweight":               ["obesity"],
    "obese":                    ["obesity"],
    "dry_lips":                 ["drying_and_tingling_lips"],
    "tingling_lips":            ["drying_and_tingling_lips"],
    "hot_flashes":              ["cold_hands_and_feets"],  # dataset quirk
    "fluid_retention":          ["fluid_overload"],
    "water_retention":          ["fluid_overload"],
    "belly_ache":               ["belly_pain", "stomach_pain"],
    "mouth_ulcers":             ["ulcers_on_tongue"],
    "tongue_sores":             ["ulcers_on_tongue"],
    "itchy":                    ["itching"],
    "itchiness":                ["itching"],
    "skin_itching":             ["itching"],
    "internal_itch":            ["internal_itching"],
    "toxic_appearance":         ["toxic_look_(typhos)"],
    "sunken_eyes_dehydration":  ["sunken_eyes", "dehydration"],
    "hollow_eyes":              ["sunken_eyes"],
    "rusty_colored_sputum":     ["rusty_sputum"],
    "slimy_sputum":             ["mucoid_sputum"],
    "silver_scales":            ["silver_like_dusting"],
    "nail_dents":               ["small_dents_in_nails"],
    "inflamed_nails":           ["inflammatory_nails"],
    "nail_inflammation":        ["inflammatory_nails"],
    "scarring":                 ["scurring"],
}


# Minimum similarity score for fuzzy matching (0.0 - 1.0)
_FUZZY_CUTOFF = 0.55

# ── Build a lookup of ALL known symptom phrases (features + synonyms) ────
# Used by _extract_from_text() to find symptoms in non-delimited input.
# Sorted longest-first for greedy matching.
_ALL_KNOWN_PHRASES: list = []   # populated lazily


def _get_known_phrases(feature_names: List[str]) -> list:
    """Build and cache the list of all recognisable symptom phrases."""
    global _ALL_KNOWN_PHRASES
    if _ALL_KNOWN_PHRASES:
        return _ALL_KNOWN_PHRASES

    phrases = set()
    # Add all feature names (with underscores replaced by spaces for matching)
    for f in feature_names:
        phrases.add(f.replace("_", " "))
    # Add all synonym keys
    for key in SYMPTOM_SYNONYMS:
        phrases.add(key.replace("_", " "))

    # Sort longest-first so greedy matching prefers "high fever" over "fever"
    _ALL_KNOWN_PHRASES = sorted(phrases, key=len, reverse=True)
    return _ALL_KNOWN_PHRASES


def _extract_from_text(text: str, feature_names: List[str]) -> List[str]:
    """
    Extract individual symptom phrases from a blob of text that may lack
    delimiters.  E.g. 'high fever severe headache joint pain skin rash'
    -> ['high fever', 'severe headache', 'joint pain', 'skin rash']

    Uses greedy longest-match against all known symptom names and synonyms.
    """
    text = text.strip().lower()
    if not text:
        return []

    known = _get_known_phrases(feature_names)
    found: List[str] = []
    remaining = text

    # Greedy extraction: find longest known phrase, remove it, repeat
    changed = True
    while remaining and changed:
        changed = False
        for phrase in known:
            if phrase in remaining:
                found.append(phrase)
                # Remove the matched phrase from remaining text
                remaining = remaining.replace(phrase, " ", 1)
                remaining = " ".join(remaining.split())  # normalise whitespace
                changed = True
                break  # restart from longest phrases

    return found


def map_symptoms(raw_symptoms: List[str],
                 feature_names: List[str]) -> Set[str]:
    """
    Map a list of user-typed symptom strings to the set of model feature
    names they correspond to.

    Parameters
    ----------
    raw_symptoms  : list of raw strings, e.g. ['fever', 'body aches', 'cough']
    feature_names : the 132 feature column names from feature_names.pkl

    Returns
    -------
    set of matched feature names (subset of *feature_names*)
    """
    feature_set = set(feature_names)
    matched: Set[str] = set()
    unmatched: list = []

    for raw in raw_symptoms:
        # Normalise: lowercase, strip, spaces -> underscores
        norm = raw.strip().lower().replace(" ", "_")
        if not norm:
            continue

        # 1. Direct match against feature names
        if norm in feature_set:
            matched.add(norm)
            continue

        # 2. Synonym lookup
        if norm in SYMPTOM_SYNONYMS:
            for feat in SYMPTOM_SYNONYMS[norm]:
                if feat in feature_set:
                    matched.add(feat)
            continue

        # 3. Try removing trailing 's' (simple plural handling)
        singular = norm.rstrip("s") if norm.endswith("s") and len(norm) > 3 else norm
        if singular != norm and singular in feature_set:
            matched.add(singular)
            continue
        if singular != norm and singular in SYMPTOM_SYNONYMS:
            for feat in SYMPTOM_SYNONYMS[singular]:
                if feat in feature_set:
                    matched.add(feat)
            continue

        # 4. Fuzzy match fallback
        close = difflib.get_close_matches(
            norm, feature_names, n=1, cutoff=_FUZZY_CUTOFF
        )
        if close:
            matched.add(close[0])
            print(f"[MAPPER] Fuzzy-matched '{raw}' -> '{close[0]}'")
            continue

        # 5. Text extraction — the input may contain multiple symptoms
        #    without delimiters (e.g. "high fever headache joint pain")
        extracted = _extract_from_text(raw, feature_names)
        if extracted:
            print(f"[MAPPER] Extracted from '{raw}': {extracted}")
            # Recursively map the extracted symptoms
            sub_matched = map_symptoms(extracted, feature_names)
            matched.update(sub_matched)
        else:
            unmatched.append(raw)

    if unmatched:
        print(f"[MAPPER] WARNING: Could not match: {unmatched}")
    if matched:
        print(f"[MAPPER] OK Activated features ({len(matched)}): {sorted(matched)}")

    return matched

