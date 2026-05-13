"""
Day 10: Severity Scorer
Calculates a 0-100 risk score based on symptoms, duration, and age.
Exactly as defined in the project proposal.
"""

SYMPTOM_WEIGHTS = {
    "chest_pain": 30,
    "difficulty_breathing": 28,
    "vomiting_blood": 35,
    "high_fever": 20,
    "severe_headache": 15,
    "fatigue": 8,
    "cough": 6,
    "skin_rash": 5,
    "fever": 15,
    "headache": 10,
    "nausea": 7,
    "vomiting": 10,
    "diarrhoea": 8,
    "loss_of_appetite": 6,
    "weight_loss": 10,
    "sweating": 5,
    "chills": 8,
    "joint_pain": 7,
    "stomach_pain": 8,
    "acidity": 5,
    "ulcers_on_tongue": 4,
    "yellowish_skin": 12,
    "dark_urine": 10,
    "itching": 4,
    "breathlessness": 20,
    "palpitations": 15,
    "dizziness": 10,
    "obesity": 8,
    "swollen_legs": 10,
    "puffy_face_and_eyes": 8,
    "enlarged_thyroid": 10,
    "brittle_nails": 4,
    "excessive_hunger": 6,
    "extra_marital_contacts": 5,
    "muscle_weakness": 8,
    "stiff_neck": 12,
    "swelling_joints": 8,
    "movement_stiffness": 8,
    "spinning_movements": 10,
    "unsteadiness": 10,
    "weakness_of_one_body_side": 20,
    "loss_of_smell": 5,
    "bladder_discomfort": 6,
    "foul_smell_of_urine": 5,
    "continuous_feel_of_urine": 6,
    "passage_of_gases": 4,
    "internal_itching": 5,
    "toxic_look_(typhos)": 15,
    "depression": 10,
    "irritability": 6,
    "muscle_pain": 7,
    "altered_sensorium": 18,
    "red_spots_over_body": 8,
    "belly_pain": 8,
    "abnormal_menstruation": 7,
    "watering_from_eyes": 4,
    "increased_appetite": 5,
    "polyuria": 8,
    "family_history": 5,
    "mucoid_sputum": 6,
    "rusty_sputum": 8,
    "lack_of_concentration": 5,
    "visual_disturbances": 10,
    "receiving_blood_transfusion": 10,
    "receiving_unsterile_injections": 10,
    "coma": 35,
    "stomach_bleeding": 25,
    "distention_of_abdomen": 10,
    "history_of_alcohol_consumption": 8,
    "blood_in_sputum": 15,
    "prominent_veins_on_calf": 8,
    "pain_behind_the_eyes": 10,
    "constipation": 5,
    "pain_during_bowel_movements": 6,
    "pain_in_anal_region": 6,
    "bloody_stool": 12,
    "irritation_in_anus": 5,
    "neck_pain": 8,
    "weakness_in_limbs": 10,
    "back_pain": 7,
    "mild_fever": 10,
    "yellow_urine": 8,
    "yellowing_of_eyes": 12,
    "acute_liver_failure": 30,
    "fluid_overload": 15,
    "swelling_of_stomach": 10,
    "swelled_lymph_nodes": 10,
    "malaise": 6,
    "blurred_and_distorted_vision": 10,
    "phlegm": 5,
    "throat_irritation": 4,
    "redness_of_eyes": 5,
    "sinus_pressure": 6,
    "runny_nose": 4,
    "congestion": 5,
    "loss_of_balance": 10,
    "unsteadiness": 10,
    "weakness_of_one_body_side": 20,
    "cold_hands_and_feets": 6,
    "hot_flushes": 5,
    "skin_peeling": 5,
    "silver_like_dusting": 4,
    "small_dents_in_nails": 4,
    "inflammatory_nails": 5,
    "blister": 6,
    "red_sore_around_nose": 4,
    "yellow_crust_ooze": 5,
}


def calculate_severity(symptoms: list, duration_days: int = 1, age: int = 30) -> int:
    """
    Calculate severity score 0-100.
    Args:
        symptoms     : list of symptom strings
        duration_days: how many days symptoms have persisted
        age          : patient age
    Returns:
        int score between 0 and 100
    """
    # Normalise symptom names to match weight keys
    normalised = [s.strip().lower().replace(" ", "_") for s in symptoms]

    # Sum base weights
    base = sum(SYMPTOM_WEIGHTS.get(s, 5) for s in normalised)

    # Duration multiplier (capped at 2x)
    duration_factor = min(1 + (duration_days * 0.1), 2.0)

    # Age multiplier
    age_factor = 1.3 if age > 60 or age < 5 else 1.0

    score = int(base * duration_factor * age_factor)
    return min(score, 100)