#!/usr/bin/env python3
"""
Setup script for medical_professional_lora LoRA adapter
Creates the training configuration and prepares for LoRA fine-tuning
"""

import json
import os
from pathlib import Path

def create_lora_config():
    """Create LoRA configuration for medical professional training"""
    
    # LoRA configuration for medical professional fine-tuning
    lora_config = {
        "model_type": "mistral",
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "local_model_path": "/Users/new/elara_main/models_files/mistral",
        "lora_config": {
            "r": 16,  # Low-rank dimension
            "lora_alpha": 32,  # LoRA scaling parameter
            "target_modules": [
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "training_config": {
            "output_dir": "/Users/new/elara_main/lora_adapters/medical_professional",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "save_total_limit": 3,
            "fp16": True,
            "optim": "paged_adamw_8bit",
            "weight_decay": 0.01,
            "max_grad_norm": 1.0
        },
        "data_config": {
            "dataset_path": "/Users/new/elara_main/data/medical_professional_training.jsonl",
            "max_seq_length": 1024,
            "train_split": 0.9,
            "eval_split": 0.1
        },
        "prompt_template": {
            "instruction": "### Human: {instruction}\n\n### Assistant: {response}",
            "system_message": "You are a medical AI assistant designed to help healthcare professionals. Provide detailed, accurate clinical information using appropriate medical terminology. Always include safety considerations and encourage consulting with colleagues when appropriate."
        }
    }
    
    return lora_config

def create_medical_training_data():
    """Create sample medical professional training data"""
    
    # Enhanced medical professional training examples
    training_examples = [
        {
            "instruction": "A 55-year-old male presents with acute chest pain radiating to the left arm. EKG shows ST elevation in leads II, III, aVF. What is your immediate assessment and management plan?",
            "response": "Based on the clinical presentation and EKG findings showing ST elevation in the inferior leads (II, III, aVF), this suggests an acute ST-elevation myocardial infarction (STEMI) involving the inferior wall, likely due to right coronary artery occlusion.\n\nImmediate management priorities:\n1. Activate STEMI protocol and cardiology consultation\n2. Administer dual antiplatelet therapy (aspirin 325mg + P2Y12 inhibitor)\n3. Anticoagulation with heparin\n4. Primary PCI within 90 minutes if available, or thrombolytic therapy if PCI not accessible within 120 minutes\n5. Continuous cardiac monitoring and vital sign assessment\n6. Pain management with nitroglycerin (ensuring no contraindications)\n7. Consider right-sided EKG leads to assess for RV involvement\n\nKey contraindications to assess: Recent surgery, bleeding history, hypertension >180/110 mmHg for thrombolytics."
        },
        {
            "instruction": "Interpret these lab values: WBC 15,000, Hgb 8.5, platelets 450,000, BUN 45, Creatinine 2.1, eGFR 35. Patient has 3-day history of dysuria and flank pain.",
            "response": "Laboratory interpretation:\n- Leukocytosis (WBC 15,000) suggests acute infection\n- Anemia (Hgb 8.5) - may be chronic or acute\n- Thrombocytosis (platelets 450,000) - reactive, likely due to inflammation\n- Elevated BUN (45) and creatinine (2.1) with reduced eGFR (35) indicate acute kidney injury (AKI)\n\nClinical correlation with dysuria and flank pain suggests acute pyelonephritis with secondary AKI.\n\nRecommended workup:\n1. Urinalysis with microscopy and urine culture\n2. Blood cultures\n3. Renal ultrasound to rule out obstruction\n4. Consider CT urogram if concern for complicated infection or stones\n\nManagement considerations:\n- Broad-spectrum antibiotics adjusted for renal function\n- Aggressive fluid resuscitation if not contraindicated\n- Monitor renal function closely\n- Consider nephrology consultation if AKI persists"
        },
        {
            "instruction": "A 28-year-old female healthcare worker presents with fever, headache, and neck stiffness. CSF analysis shows: protein 120 mg/dL, glucose 35 mg/dL, WBC 850 (90% neutrophils). Serum glucose is 95 mg/dL.",
            "response": "CSF analysis interpretation:\n- Elevated protein (120 mg/dL, normal <45)\n- Low glucose (35 mg/dL with CSF:serum ratio of 0.37, concerning for bacterial etiology)\n- Elevated WBC count with neutrophilic predominance (850 cells/μL, >90% neutrophils)\n\nThis CSF profile is highly suggestive of acute bacterial meningitis.\n\nImmediate management (should be initiated before LP if delayed):\n1. Blood cultures × 2\n2. Empirical antibiotic therapy: Vancomycin + Ceftriaxone\n3. Add ampicillin if >50 years old (Listeria coverage)\n4. Dexamethasone 0.15 mg/kg q6h × 4 days (start with or before antibiotics)\n5. Close monitoring in ICU setting\n\nAdditional workup:\n- CSF culture, bacterial antigens, PCR panel\n- Procalcitonin\n- Imaging if concern for mass effect or complications\n\nHealthcare worker exposure consideration: Ensure respiratory precautions until N. meningitidis ruled out. Consider prophylaxis for close contacts if confirmed."
        },
        {
            "instruction": "Patient on warfarin presents with INR of 8.5 and no active bleeding. How do you manage anticoagulation reversal?",
            "response": "For supratherapeutic INR (8.5) without active bleeding:\n\nRisk stratification:\n- High bleeding risk: INR >9.0 with risk factors\n- No active bleeding but significantly elevated INR\n\nManagement approach:\n1. Hold warfarin immediately\n2. Give oral vitamin K 2.5-5 mg (faster reversal than IV, less risk of resistance)\n3. Recheck INR in 24 hours\n4. Monitor closely for bleeding signs\n\nIf bleeding develops or urgent procedure needed:\n- 4-factor prothrombin complex concentrate (PCC) 25-50 units/kg, OR\n- Fresh frozen plasma 15-30 mL/kg if PCC unavailable\n- Vitamin K 10 mg IV slow push\n- Consider recombinant factor VIIa if refractory\n\nRestarting anticoagulation:\n- Resume warfarin at reduced dose when INR <3.0\n- Investigate cause of elevation (drug interactions, illness, diet)\n- Consider alternative anticoagulant if recurrent instability\n\nMonitoring: Daily INR until therapeutic range achieved."
        },
        {
            "instruction": "What are the indications and contraindications for thrombolytic therapy in acute ischemic stroke?",
            "response": "Thrombolytic therapy (alteplase) indications for acute ischemic stroke:\n\n**Time windows:**\n- IV tPA: ≤4.5 hours from symptom onset\n- Endovascular therapy: ≤6-24 hours (selected patients with large vessel occlusion)\n\n**Inclusion criteria:**\n- Acute ischemic stroke with measurable neurologic deficit\n- Age ≥18 years\n- NIHSS ≥4 (or disabling deficit)\n\n**Absolute contraindications:**\n- ICH on imaging\n- Hemorrhagic transformation of stroke\n- Active internal bleeding\n- Suspected/known bleeding disorder\n- Recent surgery/trauma within 2 weeks\n- Recent GI/GU bleeding within 21 days\n- Arterial puncture at non-compressible site within 7 days\n- Previous stroke within 3 months\n- Platelet count <100,000\n- Heparin use within 48h with elevated PTT\n- Oral anticoagulants with INR >1.7\n\n**Relative contraindications (3-4.5h window):**\n- Age >80 years\n- NIHSS >25\n- History of stroke + diabetes\n- Oral anticoagulant use regardless of INR\n\n**Blood pressure criteria:**\n- Must be <185/110 mmHg before treatment\n- Maintain <180/105 mmHg post-treatment\n\nConsider endovascular therapy for large vessel occlusions even if tPA contraindicated."
        }
    ]
    
    return training_examples

def setup_directories():
    """Create necessary directories for LoRA training"""
    
    directories = [
        "/Users/new/elara_main/lora_adapters",
        "/Users/new/elara_main/lora_adapters/medical_professional",
        "/Users/new/elara_main/training_configs",
        "/Users/new/elara_main/training_data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def save_training_config(config):
    """Save LoRA training configuration"""
    
    config_path = "/Users/new/elara_main/training_configs/medical_professional_lora.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Saved LoRA config to: {config_path}")

def save_training_data(training_examples):
    """Save training data in JSONL format"""
    
    data_path = "/Users/new/elara_main/training_data/medical_professional_training.jsonl"
    
    with open(data_path, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✅ Saved training data to: {data_path}")
    print(f"📊 Training examples: {len(training_examples)}")

def create_requirements_file():
    """Create requirements file for LoRA training"""
    
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "peft>=0.7.0",
        "accelerate>=0.25.0",
        "datasets>=2.14.0",
        "bitsandbytes>=0.41.0",
        "trl>=0.7.0",
        "wandb>=0.15.0",
        "protobuf>=3.19.0",
        "sentencepiece>=0.1.99"
    ]
    
    req_path = "/Users/new/elara_main/training_requirements.txt"
    
    with open(req_path, 'w') as f:
        f.write('\n'.join(requirements))
    
    print(f"✅ Created requirements file: {req_path}")

def main():
    print("🔥 MEDICAL PROFESSIONAL LORA SETUP 🔥")
    print("=" * 50)
    
    # Create directory structure
    print("\n📁 Setting up directories...")
    setup_directories()
    
    # Create LoRA configuration
    print("\n⚙️ Creating LoRA configuration...")
    lora_config = create_lora_config()
    save_training_config(lora_config)
    
    # Create training data
    print("\n📚 Creating training data...")
    training_examples = create_medical_training_data()
    save_training_data(training_examples)
    
    # Create requirements file
    print("\n📋 Creating requirements file...")
    create_requirements_file()
    
    print("\n🎉 SETUP COMPLETE! 🎉")
    print("-" * 50)
    print("✨ Ready for medical_professional_lora training!")
    print("\n📋 Next steps:")
    print("1. Install training requirements: pip install -r training_requirements.txt")
    print("2. Run the LoRA trainer script")
    print("3. Evaluate and integrate the adapter")
    print("\n🎯 Files created:")
    print("  • LoRA config: /Users/new/elara_main/training_configs/medical_professional_lora.json")
    print("  • Training data: /Users/new/elara_main/training_data/medical_professional_training.jsonl")
    print("  • Requirements: /Users/new/elara_main/training_requirements.txt")

if __name__ == "__main__":
    main()
