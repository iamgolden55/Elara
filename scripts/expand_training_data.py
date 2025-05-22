#!/usr/bin/env python3
"""
🔥 ELARA AI TRAINING DATA EXPANDER 🔥
Adds diverse medical specialty training examples for professional LoRA training
"""

import json
import os
from datetime import datetime

# Additional training examples across medical specialties
ADDITIONAL_EXAMPLES = [
    # Pediatrics
    {
        "instruction": "A 6-month-old infant presents with fever, poor feeding, and irritability. Temperature is 38.5°C. Parents report decreased wet diapers. What is your assessment and management approach?",
        "response": "Clinical presentation in a 6-month-old with fever and decreased oral intake suggests systemic illness requiring urgent evaluation.\n\nImmediate assessment priorities:\n1. Full set of vital signs including weight for dehydration assessment\n2. Complete physical examination focusing on signs of serious bacterial infection (SBI)\n3. Evaluation for focal infection sources (ear, throat, skin, GU)\n\nDiagnostic workup for febrile infant 3-36 months:\n- Complete blood count with differential\n- Blood culture\n- Urinalysis and urine culture (catheterized sample)\n- Consider lumbar puncture if ill-appearing or high-risk factors\n\nManagement:\n- Antipyretics for comfort (acetaminophen or ibuprofen if >6mo)\n- Assess hydration status and provide fluids as needed\n- Empirical antibiotics if ill-appearing or meeting high-risk criteria\n- Close monitoring and parental education on warning signs\n- Arrange follow-up within 24 hours\n\nRed flags requiring immediate intervention: lethargy, poor perfusion, respiratory distress, petechial rash, bulging fontanelle."
    },
    
    # Surgery/Trauma
    {
        "instruction": "A 32-year-old male involved in motor vehicle collision presents with abdominal pain and hypotension. FAST exam shows free fluid in pelvis. Hemoglobin drops from 12.8 to 9.1 g/dL over 2 hours. How do you proceed?",
        "response": "This scenario suggests traumatic hemoperitoneum with ongoing blood loss requiring immediate intervention.\n\nPrimary assessment (ATLS approach):\n1. Airway: Secure if compromised\n2. Breathing: Assessment for pneumo/hemothorax\n3. Circulation: Large bore IV access, type and crossmatch\n4. Disability: Neurologic assessment\n5. Exposure: Full examination for injuries\n\nImmediate management:\n- Activate trauma team and OR if not already done\n- Massive transfusion protocol: Type O blood until crossmatched\n- Target systolic BP 90-100 mmHg (permissive hypotension)\n- Consider TXA if within 3 hours of injury\n\nDiagnostic considerations:\n- CT abdomen/pelvis if hemodynamically stable\n- If unstable → immediate exploratory laparotomy\n- Consider diagnostic peritoneal lavage if FAST inconclusive\n\nSurgical indications:\n- Hemodynamic instability despite resuscitation\n- Positive FAST with clinical deterioration\n- Peritoneal signs\n- Ongoing transfusion requirements >2 units\n\nAnticipate: Splenic laceration, hepatic injury, mesenteric tear, or major vascular injury. Early surgical consultation mandatory."
    },
    
    # Emergency Medicine
    {
        "instruction": "A 45-year-old female presents with acute shortness of breath, chest pain, and hemoptysis. Recently returned from overseas flight. Vitals: HR 110, BP 100/60, O2 sat 89% on room air. What is your diagnostic approach?",
        "response": "Clinical presentation strongly suggests pulmonary embolism (PE) in the setting of recent travel (prolonged immobilization).\n\nRisk stratification using Wells Score:\n- Clinical features of DVT: +3\n- Alternative diagnosis less likely: +3\n- Heart rate >100: +1.5\n- Immobilization/surgery in past 4 weeks: +1.5\n- Previous PE/DVT: +1.5\n- Hemoptysis: +1\n- Malignancy: +1\n\nImmediate stabilization:\n1. Supplemental oxygen to maintain >90%\n2. IV access and monitoring\n3. Consider non-invasive ventilation if severe hypoxemia\n\nDiagnostic approach:\n- If Wells score <4: D-dimer → if elevated, proceed to imaging\n- If Wells score ≥4 or high clinical suspicion: CT pulmonary angiogram (CTPA)\n- Consider bedside echo to assess RV strain\n- Baseline labs: CBC, PT/PTT, troponin, BNP\n\nAnticoagulation:\n- Start therapeutic anticoagulation unless contraindicated\n- Options: Heparin, LMWH, or DOAC\n- If massive PE with shock: Consider thrombolytics or embolectomy\n\nRisk stratification post-diagnosis: Assess for submassive vs massive PE to guide further management and monitoring level."
    },
    
    # Radiology Interpretation
    {
        "instruction": "Chest X-ray shows bilateral lower lobe infiltrates with air bronchograms. Patient is a 70-year-old with fever, productive cough, and confusion. What is your radiological interpretation and clinical correlation?",
        "response": "Radiological interpretation:\nBilateral lower lobe infiltrates with air bronchograms are consistent with alveolar filling process. The presence of air bronchograms indicates consolidated airspaces with patent bronchi.\n\nDifferential diagnosis:\n1. Pneumonia (bacterial, atypical, or viral)\n2. Pulmonary edema (cardiogenic or non-cardiogenic)\n3. Aspiration pneumonitis\n4. Hemorrhage or contusion\n5. Malignancy (less likely given bilateral nature)\n\nClinical correlation:\nIn a 70-year-old with fever, productive cough, and confusion, community-acquired pneumonia (CAP) is most likely. The confusion may represent sepsis-associated delirium or evolving pneumonia severity.\n\nSeverity assessment (CURB-65):\n- Confusion: +1\n- Age ≥65: +1\n- Additional factors: check BUN, respiratory rate, blood pressure\n\nRecommended management:\n1. Assess pneumonia severity (PSI score or CURB-65)\n2. Obtain sputum and blood cultures\n3. Empirical antibiotic therapy based on severity and risk factors\n4. Consider atypical coverage if indicated\n5. Supportive care and monitoring\n\nFollow-up imaging in 6-8 weeks if clinical resolution to ensure complete radiographic clearance."
    },
    
    # Psychiatry
    {
        "instruction": "A 28-year-old male brought by police after threatening suicide. He appears agitated, reports hearing voices telling him to harm himself, and has not slept in 3 days. How do you assess and manage this patient?",
        "response": "This presentation requires immediate psychiatric evaluation and risk assessment for both suicide and psychosis.\n\nImmediate safety assessment:\n1. Ensure safe environment (remove potential weapons, continuous observation)\n2. Assess immediate suicide risk using standardized tools\n3. Consider involuntary hold if imminent danger to self\n\nMental status examination:\n- Appearance and behavior: agitation, psychomotor activity\n- Speech: rate, volume, coherence\n- Mood and affect: dysphoric, labile\n- Thought process: organized vs disorganized\n- Thought content: suicidal ideation, homicidal ideation\n- Perceptual disturbances: auditory hallucinations (command type)\n- Cognitive function: orientation, attention, memory\n- Insight and judgment: impaired\n\nRisk assessment factors:\n- Acute psychosis with command hallucinations\n- Sleep deprivation (possible mania or drug-induced)\n- Explicit suicidal threats\n- Agitation and poor reality testing\n\nManagement:\n1. Medical clearance: basic metabolics, toxicology screen, vital signs\n2. Psychiatric hold if meeting criteria\n3. Consider chemical sedation if severely agitated\n4. Antipsychotic medication (haloperidol, olanzapine, or risperidone)\n5. Close monitoring in secure psychiatric unit\n6. Social work evaluation for safety planning and discharge planning\n\nDifferential: First break psychosis, bipolar disorder with psychotic features, substance-induced psychosis, brief psychotic disorder."
    },
    
    # Dermatology
    {
        "instruction": "A 35-year-old female presents with a 6mm asymmetric, irregularly bordered, dark brown lesion on her back that has changed in size over 2 months. What is your assessment and next steps?",
        "response": "Clinical presentation raises significant concern for melanoma based on ABCDE criteria:\n- A: Asymmetry ✓\n- B: Border irregularity ✓\n- C: Color variation (dark brown)\n- D: Diameter >6mm ✓\n- E: Evolution/Evolving ✓\n\nImmediate assessment:\n1. Dermatoscopic examination if available\n2. Complete skin examination for additional concerning lesions\n3. Lymph node examination (especially regional drainage)\n4. Photography for documentation\n\nManagement:\n1. **Urgent dermoscopy and biopsy** - Should be performed within 2 weeks\n2. Full excisional biopsy with 1-2mm margins (not shave biopsy)\n3. Send specimen for histopathologic examination\n4. Patient education about sun protection and self-examination\n\nIf melanoma confirmed:\n- Staging based on Breslow thickness and other features\n- Wide local excision with appropriate margins\n- Sentinel lymph node biopsy if indicated (thickness >0.8mm or other high-risk features)\n- Refer to dermatology/surgical oncology\n- Surveillance plan based on stage\n\nRed flags requiring immediate referral:\n- Rapid growth\n- Bleeding or ulceration\n- Multiple concerning lesions\n- Family history of melanoma\n\nEmphasize importance of regular skin checks and sun protective measures."
    },
    
    # Orthopedics
    {
        "instruction": "A 65-year-old female fell from standing height and cannot bear weight on left leg. X-ray shows displaced left femoral neck fracture. PMH includes osteoporosis. What are your management considerations?",
        "response": "Displaced femoral neck fracture in elderly patient requires urgent surgical intervention due to high risk of avascular necrosis and nonunion.\n\nImmediate considerations:\n1. Pain management and hemodynamic stability\n2. DVT prophylaxis\n3. Pre-operative medical optimization\n4. NPO status for surgery\n\nSurgical options based on patient factors:\n\n**Total Hip Arthroplasty (preferred for displaced fractures in this age group):**\n- Advantages: Immediate weight bearing, lower revision rate\n- Considerations: Bone quality, activity level, life expectancy\n\n**Hemiarthroplasty:**\n- Consider if less active or significant comorbidities\n- Faster surgery but may require future revision\n\n**Internal fixation (ORIF):**\n- Generally not recommended for displaced fractures in elderly\n- High failure rate due to poor bone quality\n\nPre-operative assessment:\n- Cardiology clearance if indicated\n- Labs: CBC, CMP, PT/PTT, Type & Screen\n- ECG, CXR\n- Optimize medical comorbidities\n\nPost-operative considerations:\n- Early mobilization and physical therapy\n- Fall prevention assessment\n- Bone health optimization (calcium, vitamin D, bisphosphonates)\n- Nutritional assessment\n- Discharge planning (home vs rehabilitation)\n\nDelay in surgery >48 hours associated with increased morbidity and mortality."
    },
    
    # Obstetrics & Gynecology
    {
        "instruction": "A 26-year-old G2P1 at 34 weeks gestation presents with severe headache, epigastric pain, and blurred vision. BP 170/110, proteinuria 3+. What is your diagnosis and immediate management?",
        "response": "Clinical presentation strongly suggests severe preeclampsia/preeclampsia with severe features at 34 weeks gestation.\n\nDiagnostic criteria met:\n- Hypertension ≥160/110 mmHg ✓\n- Severe features: CNS symptoms (headache, visual changes), epigastric pain ✓\n- Proteinuria ✓\n\nImmediate assessment and stabilization:\n1. **Magnesium sulfate** for seizure prophylaxis (4-6g IV bolus, then 2g/hr)\n2. **Antihypertensive therapy** - goal SBP <160, DBP <110\n   - Labetalol 20mg IV or Hydralazine 5-10mg IV\n3. **Corticosteroids** for fetal lung maturity (betamethasone 12mg IM q24h x2)\n4. Continuous fetal monitoring\n5. Strict I/O monitoring\n\nLaboratory evaluation:\n- CBC with platelets (check for HELLP syndrome)\n- Comprehensive metabolic panel (creatinine, liver enzymes)\n- Coagulation studies\n- Uric acid, LDH\n- 24-hour urine protein (if time permits)\n\nCritical monitoring:\n- Neurologic status (reflexes, mental status)\n- Visual changes\n- RUQ/epigastric pain\n- Urine output >30ml/hr\n- Signs of magnesium toxicity\n\nDelivery planning:\n- **Immediate delivery indicated** for preeclampsia with severe features at ≥34 weeks\n- Mode depends on cervical status and fetal presentation\n- NICU consultation for premature infant\n- Anesthesia consultation for delivery planning\n\nComplications to monitor: Eclampsia, HELLP syndrome, placental abruption, acute renal failure, cerebral hemorrhage."
    }
]

def expand_training_data():
    """Add new diverse medical training examples to existing dataset"""
    
    # Path to existing training data
    training_file = "/Users/new/elara_main/training_data/medical_professional_training.jsonl"
    
    # Read existing data
    existing_examples = []
    if os.path.exists(training_file):
        with open(training_file, 'r') as f:
            for line in f:
                if line.strip():
                    existing_examples.append(json.loads(line))
    
    print(f"📊 Found {len(existing_examples)} existing training examples")
    
    # Append new examples
    with open(training_file, 'a') as f:
        for i, example in enumerate(ADDITIONAL_EXAMPLES):
            f.write(json.dumps(example) + '\n')
            print(f"✅ Added example {i+1}: {example['instruction'][:60]}...")
    
    print(f"\n🎉 SUCCESS! Total training examples: {len(existing_examples) + len(ADDITIONAL_EXAMPLES)}")
    print(f"📁 File: {training_file}")
    
    # Create backup
    backup_file = f"/Users/new/elara_main/training_data/medical_professional_training_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    os.rename(training_file, backup_file)
    
    # Write all examples to new file
    with open(training_file, 'w') as f:
        for example in existing_examples + ADDITIONAL_EXAMPLES:
            f.write(json.dumps(example) + '\n')
    
    print(f"💾 Backup created: {backup_file}")
    
    # Show coverage summary
    specialties = [
        "Cardiology (STEMI)",
        "Internal Medicine (UTI/AKI)", 
        "Infectious Disease (Meningitis)",
        "Hematology (Anticoagulation)",
        "Neurology (Stroke)",
        "Pediatrics (Febrile infant)",
        "Surgery/Trauma (MVA)",
        "Emergency Medicine (PE)",
        "Radiology (CXR interpretation)",
        "Psychiatry (Acute psychosis)",
        "Dermatology (Melanoma)",
        "Orthopedics (Hip fracture)",
        "OB/GYN (Preeclampsia)"
    ]
    
    print(f"\n🩺 MEDICAL SPECIALTIES COVERED ({len(specialties)}):")
    for i, specialty in enumerate(specialties, 1):
        print(f"{i:2d}. {specialty}")
    
    return len(existing_examples) + len(ADDITIONAL_EXAMPLES)

if __name__ == "__main__":
    print("🔥 EXPANDING MEDICAL PROFESSIONAL TRAINING DATA 🔥\n")
    total_examples = expand_training_data()
    print(f"\n🚀 Ready for LoRA training with {total_examples} diverse medical examples!")
