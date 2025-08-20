import streamlit as st 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import GRU, Dense, Embedding, Dropout 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from sklearn.model_selection import train_test_split 
import pickle 
import re 
import requests 
import json 
from datetime import datetime, timedelta 
import plotly.express as px 
import plotly.graph_objects as go 
from collections import Counter 
import warnings 
warnings.filterwarnings('ignore') 
 
# Configure Streamlit page 
st.set_page_config( 
    page_title="Medical Health GRU Chatbot", 
    page_icon=" CHAT", 
    layout="wide" 
) 
 
class MedicalGRUChatbot: 
    def __init__(self): 
        self.tokenizer = None 
        self.model = None 
        self.max_sequence_length = 150 
        self.vocab_size = 15000 
        self.medical_keywords = self.load_medical_keywords() 
        self.symptoms_db = self.load_symptoms_database() 
        self.drug_db = self.load_drug_database() 
         
    def load_medical_keywords(self): 
        """Load medical terminology and keywords""" 
        return { 
            'symptoms': ['headache', 'fever', 'cough', 'fatigue', 'nausea', 'dizziness', 'pain', 'ache',  'shortness of breath', 'chest pain', 'abdominal pain', 'back pain', 'sore throat', 'runny nose', 'congestion', 'vomiting', 'diarrhea', 'constipation', 'rash', 'swelling', 'joint pain', 'muscle pain', 'insomnia', 'anxiety', 'depression'], 
            'body_parts': ['head', 'chest', 'abdomen', 'back', 'arm', 'leg', 'throat', 'stomach', 'heart', 'lungs', 'kidney', 'liver', 'brain', 'eye', 'ear', 'nose', 'mouth'], 
            'conditions': ['diabetes', 'hypertension', 'asthma', 'pneumonia', 'covid-19', 'flu',  'cold', 'migraine', 'arthritis', 'depression', 'anxiety', 'allergies'], 
            'specialties': ['cardiology', 'neurology', 'gastroenterology', 'pulmonology', 'orthopedics', 'dermatology', 'psychiatry', 'emergency medicine', 'family medicine'] 
        } 
     
    def load_symptoms_database(self): 
        """Load symptom-condition database""" 
        return { 
            'fever + cough + fatigue': { 
                'conditions': ['Common Cold', 'Flu', 'COVID-19', 'Pneumonia'], 
                'recommendations': 'Rest, hydration, monitor temperature. Consult doctor if symptoms worsen.', 
                'urgency': 'moderate' 
            }, 
            'chest pain + shortness of breath': { 
                'conditions': ['Heart Attack', 'Angina', 'Pulmonary Embolism', 'Asthma'], 
                'recommendations': 'SEEK IMMEDIATE MEDICAL ATTENTION', 
                'urgency': 'high' 
            }, 
            'headache + fever + stiff neck': { 
                'conditions': ['Meningitis', 'Encephalitis'], 
                'recommendations': 'EMERGENCY - Go to hospital immediately', 
                'urgency': 'critical' 
            }, 
            'abdominal pain + nausea + vomiting': { 
                'conditions': ['Gastroenteritis', 'Appendicitis', 'Food Poisoning'], 
                'recommendations': 'Monitor symptoms, stay hydrated. See doctor if severe or persistent.', 
                'urgency': 'moderate' 
            }, 
            'persistent cough + weight loss + night sweats': { 
                'conditions': ['Tuberculosis', 'Lung Cancer', 'Chronic Infection'], 
                'recommendations': 'Consult pulmonologist for thorough evaluation', 
                'urgency': 'high' 
            } 
        } 
     
    def load_drug_database(self): 
        """Load drug information database""" 
        return { 
            'paracetamol': { 
                'uses': 'Pain relief, fever reduction', 
                'dosage': '500-1000mg every 4-6 hours, max 4g/day', 
                'side_effects': 'Liver damage with overdose', 
                'contraindications': 'Severe liver disease' 
            }, 
            'ibuprofen': { 
                'uses': 'Pain relief, anti-inflammatory, fever reduction', 
                'dosage': '200-400mg every 4-6 hours, max 1.2g/day', 
                'side_effects': 'Stomach upset, kidney problems', 
                'contraindications': 'Kidney disease, stomach ulcers' 
            }, 
            'aspirin': { 
                'uses': 'Pain relief, blood thinner, heart protection', 
                'dosage': '75-325mg daily for prevention, 500-1000mg for pain', 
                'side_effects': 'Bleeding risk, stomach irritation', 
                'contraindications': 'Bleeding disorders, children with viral infections' 
            }, 
            'metformin': { 
                'uses': 'Type 2 diabetes management', 
                'dosage': '500-850mg twice daily with meals', 
                'side_effects': 'Gastrointestinal upset, lactic acidosis (rare)', 
                'contraindications': 'Kidney disease, liver disease' 
            }, 
            'lisinopril': { 
                'uses': 'High blood pressure, heart failure', 
                'dosage': '2.5-40mg once daily', 
                'side_effects': 'Dry cough, hyperkalemia, angioedema', 
                'contraindications': 'Pregnancy, bilateral renal artery stenosis' 
            } 
        } 
     
    def preprocess_medical_text(self, text): 
        """Clean and preprocess medical text data""" 
        text = text.lower() 
        # Keep medical terminology intact 
        text = re.sub(r'[^a-zA-Z0-9\s\-/]', '', text) 
        text = re.sub(r'\s+', ' ', text).strip() 
        return text 
     
    def create_medical_model(self, vocab_size, embedding_dim=150, gru_units=300): 
        """Create medical-specific GRU model""" 
        model = Sequential([ 
            Embedding(vocab_size, embedding_dim, input_length=self.max_sequence_length), 
            GRU(gru_units, return_sequences=True, dropout=0.3, recurrent_dropout=0.2), 
            GRU(gru_units//2, return_sequences=True, dropout=0.3, recurrent_dropout=0.2), 
            GRU(gru_units//4, dropout=0.3), 
            Dense(512, activation='relu'), 
            Dropout(0.5), 
            Dense(256, activation='relu'), 
            Dropout(0.3), 
            Dense(vocab_size, activation='softmax') 
        ]) 
         
        model.compile( 
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy'] 
        ) 
        return model 
     
    def prepare_medical_training_data(self, conversations): 
        """Prepare medical training data""" 
        input_texts = [] 
        target_texts = [] 
         
        for conversation in conversations: 
            if len(conversation) >= 2: 
                for i in range(len(conversation) - 1): 
                    input_texts.append(self.preprocess_medical_text(conversation[i])) 
                    target_texts.append(self.preprocess_medical_text(conversation[i + 1])) 
         
        return input_texts, target_texts 
     
    def train_model(self, input_texts, target_texts): 
        """Train the medical GRU model""" 
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>") 
        all_texts = input_texts + target_texts 
        self.tokenizer.fit_on_texts(all_texts) 
         
        input_sequences = self.tokenizer.texts_to_sequences(input_texts) 
        target_sequences = self.tokenizer.texts_to_sequences(target_texts) 
         
        X = pad_sequences(input_sequences, maxlen=self.max_sequence_length, padding='post') 
        y = pad_sequences(target_sequences, maxlen=self.max_sequence_length, padding='post') 
         
        y = y[:, 0] if y.shape[1] > 0 else np.zeros(len(y)) 
         
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
         
        self.model = self.create_medical_model(len(self.tokenizer.word_index) + 1) 
         
        history = self.model.fit( 
            X_train, y_train, 
            validation_data=(X_test, y_test), 
            epochs=15, 
            batch_size=32, 
            verbose=0 
        ) 
         
        return history 
     
    def analyze_symptoms(self, user_input): 
        """Analyze symptoms mentioned by user""" 
        symptoms_found = [] 
        input_lower = user_input.lower() 
         
        for symptom in self.medical_keywords['symptoms']: 
            if symptom in input_lower: 
                symptoms_found.append(symptom) 
         
        # Check symptom combinations 
        for symptom_combo, info in self.symptoms_db.items(): 
            combo_symptoms = symptom_combo.split(' + ') 
            if all(symptom in input_lower for symptom in combo_symptoms): 
                return { 
                    'symptoms': combo_symptoms, 
                    'possible_conditions': info['conditions'], 
                    'recommendations': info['recommendations'], 
                    'urgency': info['urgency'] 
                } 
         
        return {'symptoms': symptoms_found, 'analysis': 'partial'} 
     
    def get_drug_info(self, drug_name): 
        """Get drug information""" 
        drug_lower = drug_name.lower() 
        for drug, info in self.drug_db.items(): 
            if drug in drug_lower: 
                return info 
        return None 
     
    def generate_medical_response(self, input_text, max_length=60): 
        """Generate medical response using trained model and rule-based system""" 
        if not self.model or not self.tokenizer: 
            return "Medical system not initialized. Please train the model first." 
         
        # Check for emergency keywords 
        emergency_keywords = ['chest pain', 'difficulty breathing', 'severe pain', 'unconscious',  'bleeding', 'poisoning', 'heart attack', 'stroke'] 
         
        input_lower = input_text.lower() 
         
        for emergency in emergency_keywords: 
            if emergency in input_lower: 
                return "EMERGENCY: Please call emergency services (911) or go to the nearest hospital immediately. This chatbot cannot replace emergency medical care." 
         
        # Analyze symptoms 
        symptom_analysis = self.analyze_symptoms(input_text) 
        if 'possible_conditions' in symptom_analysis: 
            response = f"Based on your symptoms ({', '.join(symptom_analysis['symptoms'])}), possible conditions include: {', '.join(symptom_analysis['possible_conditions'])}.\n\n" 
            response += f"Recommendation: {symptom_analysis['recommendations']}\n\n" 
            if symptom_analysis['urgency'] == 'critical': 
                response = " " + response 
            elif symptom_analysis['urgency'] == 'high': 
                response = " " + response 
            response += "Please consult with a healthcare professional for proper diagnosis and treatment." 
            return response 
         
        # Check for drug information requests 
        drug_keywords = ['medicine', 'medication', 'drug', 'pill', 'tablet'] 
        if any(keyword in input_lower for keyword in drug_keywords): 
            for drug in self.drug_db.keys(): 
                if drug in input_lower: 
                    drug_info = self.get_drug_info(drug) 
                    if drug_info: 
                        return(
                            f"{drug.title()} Information:**\n\n"
                            f"Uses: {drug_info['uses']}\n" 
                            f"Dosage:{drug_info['dosage']}\n"  
                            f"Side Effects: {drug_info['side_effects']}\n" 
                            f"Contraindications: {drug_info['contraindications']}\n\n" 
                           "Always consult your doctor before taking any medication." 
                        )
         
        # Generate response using GRU model 
        processed_input = self.preprocess_medical_text(input_text) 
        sequence = self.tokenizer.texts_to_sequences([processed_input]) 
        padded_sequence = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post') 
         
        response_words = [] 
        current_sequence = padded_sequence[0].tolist() 
         
        for _ in range(max_length): 
            prediction = self.model.predict(np.array([current_sequence]), verbose=0) 
            next_word_idx = np.argmax(prediction[0]) 
             
            next_word = None 
            for word, idx in self.tokenizer.word_index.items(): 
                if idx == next_word_idx: 
                    next_word = word 
                    break 
             
            if next_word and next_word != "<OOV>": 
                response_words.append(next_word) 
                current_sequence = current_sequence[1:] + [next_word_idx] 
                 
                # Stop at sentence boundaries 
                if next_word in ['.', '!', '?']: 
                    break 
            else: 
                break 
         
        model_response = " ".join(response_words) if response_words else "I understand your concern. Could you provide more details about your symptoms?" 
         
        # Add medical disclaimer 
        model_response += "\nMedical Disclaimer: This chatbot provides general health information only. Always consult qualified healthcare professionals for medical advice, diagnosis, and treatment." 
         
        return model_response 
 
def get_health_data(): 
    """Fetch real-time health-related data""" 
    data = {} 
     
    try: 
        # COVID-19 data (example with disease.sh API) 
        covid_url = "https://disease.sh/v3/covid-19/all" 
        covid_response = requests.get(covid_url, timeout=5) 
        if covid_response.status_code == 200: 
            covid_data = covid_response.json() 
            data['covid_cases'] = f"{covid_data['cases']:,}" 
            data['covid_deaths'] = f"{covid_data['deaths']:,}" 
            data['covid_recovered'] = f"{covid_data['recovered']:,}" 
         
        # Health tips API (mock data) 
        health_tips = [ 
            "Drink at least 8 glasses of water daily", 
            "Get 7-9 hours of sleep each night", 
            "Exercise for at least 30 minutes daily", 
            "Eat 5 servings of fruits and vegetables daily", 
            "Practice stress management techniques", 
            "Get regular health check-ups", 
            "Maintain good hand hygiene", 
            "Limit processed foods and sugar intake" 
        ] 
        data['daily_health_tip'] = np.random.choice(health_tips) 
         
    except Exception as e: 
        st.error(f"Error fetching health data: {e}") 
     
    return data 
 
def create_medical_conversations(): 
    """Create medical conversation training data""" 
    conversations = [ 
        # Symptom inquiries 
        ["I have a headache", "I understand you're experiencing a headache. Can you describe the pain - is it throbbing, sharp, or dull? How long have you had it?"], 
        ["I feel nauseous", "Nausea can be uncomfortable. Are you experiencing any other symptoms like vomiting, fever, or abdominal pain?"], 
        ["I have chest pain", "Chest pain requires immediate attention. Please describe the pain and consider seeking emergency medical care if it's severe."], 
        ["I can't sleep", "Insomnia can affect your health. Have you tried establishing a regular sleep routine? Consider speaking with a healthcare provider."], 
        ["I have a fever", "A fever indicates your body is fighting something. Monitor your temperature and stay hydrated. Seek medical care if it persists or is very high."],          
        # General health questions 
        ["How much water should I drink?", "Generally, adults should drink about 8 glasses (64 ounces) of water daily, but individual needs may vary based on activity and climate."], 
        ["What foods are good for heart health?", "Heart-healthy foods include leafy greens, whole grains, fatty fish, berries, nuts, and olive oil. Limit processed foods and saturated fats."], 
        ["How often should I exercise?", "The CDC recommends at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity weekly, plus strength training twice a week."],          
        # Medication questions 
        ["Can I take ibuprofen with food?", "Yes, taking ibuprofen with food can help reduce stomach irritation. Always follow the dosage instructions on the package."], 
        ["What is paracetamol used for?", "Paracetamol is used for pain relief and reducing fever. The usual adult dose is 500-1000mg every 4-6 hours, not exceeding 4g daily."], 
         
        # Preventive care 
        ["How can I prevent getting sick?", "Practice good hygiene, get adequate sleep, eat nutritious foods, exercise regularly, manage stress, and get recommended vaccinations."], 
        ["When should I see a doctor?", "See a doctor for persistent symptoms, severe pain, high fever, difficulty breathing, or any concerning changes in your health."], 
         
        # Mental health 
        ["I feel anxious", "Anxiety is common and treatable. Consider relaxation techniques, regular exercise, and speaking with a mental health professional if it persists."], 
        ["I feel depressed", "Depression is a serious condition that's treatable. Please consider reaching out to a mental health professional or your primary care doctor."], 
         
        # Emergency scenarios 
        ["I'm having trouble breathing", "Difficulty breathing is serious. Please call emergency services or go to the nearest emergency room immediately."], ["I think I'm having a heart attack", "This is a medical emergency. Call 911 immediately or have someone take you to the emergency room right now."] ] 
     
    return conversations 
 
def main(): 
    st.title("Medical Health GRU Chatbot") 
    st.markdown("**AI-Powered Health Assistant for Medical Information and Guidance**") 
    st.markdown("---") 
     
    # Medical disclaimer 
    st.warning("Important Medical Disclaimer: This chatbot is for informational purposes only and does not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical concerns.") 
     
    # Initialize chatbot 
    if 'medical_chatbot' not in st.session_state: 
        st.session_state.medical_chatbot = MedicalGRUChatbot() 
        st.session_state.trained = False 
        st.session_state.chat_history = [] 
        st.session_state.patient_profile = {} 
     
    # Sidebar for medical configuration 
    with st.sidebar: 
        st.header("Medical Dashboard") 
         
        # Patient Profile 
        st.subheader("Patient Information") 
        age = st.number_input("Age", min_value=0, max_value=120, value=30) 
        gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"]) 
        existing_conditions = st.multiselect("Existing Conditions",["None", "Diabetes", "Hypertension", "Asthma", "Heart Disease", "Arthritis", "Other"]) 

        if st.button("Save Profile"): 
            st.session_state.patient_profile = { 
                'age': age, 'gender': gender, 'conditions': existing_conditions 
            } 
            st.success("Profile saved!") 
         
        # Model training section 
        st.subheader("AI Model Training") 
        if st.button("Train Medical Model"): 
            with st.spinner("Training medical AI model..."): 
                conversations = create_medical_conversations() 
                input_texts, target_texts = st.session_state.medical_chatbot.prepare_medical_training_data(conversations) 
                history = st.session_state.medical_chatbot.train_model(input_texts, target_texts) 
                st.session_state.trained = True 
                st.success("Medical AI model trained successfully!") 
         
        # Health data section 
        st.subheader("Health Data") 
        if st.button("Fetch Health Updates"): 
            with st.spinner("Fetching health data..."): 
                health_data = get_health_data() 
                st.session_state.health_data = health_data 
                for key, value in health_data.items(): 
                    if 'covid' in key: 
                        st.metric(key.replace('_', ' ').title(), value) 
                    else: 
                        st.info(f" {value}") 
         
        # Emergency contacts 
        st.subheader("Emergency Contacts") 
        st.error("**Emergency:** 911") 
        st.warning("**Poison Control:** 1-800-222-1222") 
        st.info("**Mental Health Crisis:** 988") 
         
        # Clear chat 
        if st.button("Clear Consultation History"): 
            st.session_state.chat_history = [] 
            st.rerun() 
     
    # Main interface 
    col1, col2 = st.columns([2, 1]) 
     
    with col1: 
        st.subheader("Medical Consultation Chat") 
         
        # Training status 
        if st.session_state.trained: 
            st.success("Medical AI Model is ready for consultation!") 
        else: 
            st.warning("Please train the medical model first using the sidebar.") 
         
        # Quick symptom buttons 
        st.subheader("Quick Symptom Check") 
        symptom_cols = st.columns(4) 
        quick_symptoms = ["Headache", "Fever", "Cough", "Nausea", "Chest Pain", "Back Pain", "Fatigue", "Dizziness"] 
         
        for i, symptom in enumerate(quick_symptoms): 
            with symptom_cols[i % 4]: 
                if st.button(symptom, key=f"symptom_{i}"): 
                    if st.session_state.trained: 
                        response = st.session_state.medical_chatbot.generate_medical_response(f"I have {symptom.lower()}") 
                        st.session_state.chat_history.append((f"I have {symptom.lower()}", response)) 
                        st.rerun() 
         
        # Chat history display 
        st.subheader("Consultation History") 
        chat_container = st.container() 
        with chat_container: 
            for i, (patient_msg, doctor_msg) in enumerate(st.session_state.chat_history): 
                st.write(f"**Patient:** {patient_msg}") 
                st.write(f"**AI Doctor:** {doctor_msg}") 
                st.markdown("---") 
         
        # Chat input 
        patient_input = st.text_area("Describe your symptoms or ask a health question:", placeholder="e.g., I have a persistent cough and fever for 3 days...") 
         
        if st.button("Get Medical Advice") and patient_input: 
            if st.session_state.trained: 
                # Generate medical response 
                doctor_response = st.session_state.medical_chatbot.generate_medical_response(patient_input) 
                 
                # Add to consultation history 
                st.session_state.chat_history.append((patient_input, doctor_response)) 
                st.rerun() 
            else: 
                st.error("Please train the medical model first!") 
     
    with col2: 
        st.subheader("Health Analytics") 
         
        if st.session_state.chat_history: 
            # Consultation statistics 
            total_consultations = len(st.session_state.chat_history) 
            st.metric("Total Consultations", total_consultations) 
             
            # Symptom frequency analysis 
            all_messages = [msg[0] for msg in st.session_state.chat_history] 
            symptoms_mentioned = [] 
             
            for msg in all_messages: 
                for symptom in st.session_state.medical_chatbot.medical_keywords['symptoms']: 
                    if symptom in msg.lower(): 
                        symptoms_mentioned.append(symptom) 
             
            if symptoms_mentioned: 
                symptom_counts = Counter(symptoms_mentioned) 
                top_symptoms = dict(symptom_counts.most_common(5)) 
                 
                fig = px.bar( 
                    x=list(top_symptoms.keys()), 
                    y=list(top_symptoms.values()), 
                    title="Most Reported Symptoms", 
                    labels={'x': 'Symptoms', 'y': 'Frequency'} 
                ) 
                st.plotly_chart(fig, use_container_width=True) 
         
        # Health reminders 
        st.subheader("Health Reminders") 
        current_hour = datetime.now().hour 
         
        if 6 <= current_hour < 12: 
            st.info("Good morning! Don't forget to take your morning medications.") 
        elif 12 <= current_hour < 18: 
            st.info("Afternoon reminder: Stay hydrated and take a walking break.") 
        else: 
            st.info("Evening: Wind down and prepare for good sleep hygiene.") 
         
        # Medical resources 
        st.subheader("Medical Resources") 
        st.markdown(""" 
        **Trusted Health Sources:** 
        - [CDC](https://cdc.gov) - Disease Control & Prevention 
        - [WHO](https://who.int) - World Health Organization 
        - [NIH](https://nih.gov) - National Institutes of Health 
        - [WebMD](https://webmd.com) - Medical Information 
        - [Mayo Clinic](https://mayoclinic.org) - Medical Reference 
        """) 
         
        # Model information 
        st.subheader("AI Model Info") 
        st.write("**Type:** Medical GRU Neural Network") 
        st.write("**Specialization:** Symptom Analysis") 
        st.write("**Features:** Drug Information, Emergency Detection") 
        st.write("**Vocabulary:** 15,000 medical terms") 
        st.write("**Sequence Length:** 150 tokens") 
     
    # Footer with medical information 
    st.markdown("---") 
    st.markdown(""" 
    **About This Medical Chatbot:** 
    - Provides general health information and guidance 
    - Uses GRU neural networks trained on medical conversations 
    - Includes symptom analysis and drug information 
    - Detects potential emergency situations 
    - **NOT a substitute for professional medical care** 
     
    **Always consult healthcare professionals for:** 
    - Serious symptoms or emergencies 
    - Medication changes or prescriptions   
    - Diagnosis and treatment decisions 
    - Chronic condition management 
    """) 
 
if __name__ == "__main__": 
    main() 
 