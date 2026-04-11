import os
import gradio as gr
from medical_agents import run_medical_consultation
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts
from dotenv import load_dotenv

load_dotenv()

def process_inputs(audio_filepath, image_filepath):
    # Handle missing inputs
    if not audio_filepath or not image_filepath:
        return "Missing data", "Please record your voice AND upload an image.", "", "", None

    try:
        # Step 1: Transcribe the patient's voice
        speech_to_text_output = transcribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3"
        )
    except Exception as e:
        return f"STT Error: {str(e)}", "Could not transcribe audio.", "", "", None

    # Step 2: Run the Multi-Agent Medical Team via LangGraph
    try:
        results = run_medical_consultation(
            transcription=speech_to_text_output, 
            image_path=image_filepath
        )
        
        severity = results["severity"]
        doctor_res = results.get("doctor_analysis", "Not required for this mild case.")
        nurse_res = results.get("nurse_advice", "Case escalated to doctor for specialist analysis.")
        summary_res = results["final_consultation"]
        
        # Add severity badge to doctor response for clarity
        if severity == "SEVERE":
            doctor_res = "🚩 SEVERE CASE DETECTED\n" + doctor_res
        else:
            nurse_res = "✅ MILD CASE\n" + nurse_res
        
    except Exception as e:
        return speech_to_text_output, f"Agent Error: {str(e)}", "Error", "Error", None

    # Step 3: Convert the Summary to speech
    try:
        voice_of_doctor = text_to_speech_with_gtts(
            input_text=summary_res, 
            output_filepath="final.mp3"
        )
    except Exception as e:
        voice_of_doctor = None

    return speech_to_text_output, doctor_res, nurse_res, summary_res, voice_of_doctor

# --- Premium UI Design ---

custom_css = """
.gradio-container {
    background-color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}
.header-box {
    text-align: center;
    padding: 2.5rem;
    background: white;
    border-radius: 12px;
    margin-bottom: 2rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.header-box h1 {
    color: #0f172a;
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.header-box p {
    color: #64748b;
    font-size: 1.1rem;
}
.input-col, .output-col {
    background: white;
    padding: 2rem;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
}
.submit-btn {
    background: #2563eb !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.8rem !important;
    border-radius: 8px !important;
}
.submit-btn:hover {
    background: #1d4ed8 !important;
}
footer {display: none !important}
.footer-name {
    position: fixed;
    bottom: 30px;
    right: 20px;
    font-size: 0.9rem;
    color: #64748b;
    font-weight: 500;
    z-index: 100;
}
#doctor-markdown, #nurse-markdown, #summary-markdown, #stt-markdown {
    background: white;
    padding: 1.5rem;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    margin-top: 10px;
    color: #1e293b;
}
"""

with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", neutral_hue="slate"), css=custom_css) as demo:
    gr.Markdown("Lisha Vilvanathan", elem_classes="footer-name")
    with gr.Column(elem_classes="header-box"):
        gr.Markdown("# ⚕️ AI Doctor ")
        gr.Markdown("Vision & Voice Enabled Diagnostic Intelligent AI Assistant")

    with gr.Row():
        with gr.Column(elem_classes="input-col"):
            gr.Markdown("### 🎤 Step 1: Tell the Doctor")
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record your question")
            
            gr.Markdown("### 📸 Step 2: Upload Image")
            image_input = gr.Image(type="filepath", label="Scan/Upload affected area")
            
            submit_btn = gr.Button("ASK THE AI DOCTOR", elem_classes="submit-btn")
            
        with gr.Column(elem_classes="output-col"):
            gr.Markdown("### 📝 Patient's Transcription")
            stt_output = gr.Markdown("*Your words will appear here...*", elem_id="stt-markdown")
            
            with gr.Tabs():
                with gr.Tab("🩺 Doctor's Analysis"):
                    doctor_text_output = gr.Markdown("Waiting for diagnosis...", elem_id="doctor-markdown")
                with gr.Tab("🩹 Nurse's Home Care"):
                    nurse_text_output = gr.Markdown("Waiting for care instructions...", elem_id="nurse-markdown")
                with gr.Tab("📋 Consultation Summary"):
                    summary_text_output = gr.Markdown("Waiting for final summary...", elem_id="summary-markdown")
            
            gr.Markdown("### 🔊 Doctor's Voice")
            audio_output = gr.Audio(label="Listen to the summary", autoplay=True)

    submit_btn.click(
        fn=process_inputs,
        inputs=[audio_input, image_input],
        outputs=[stt_output, doctor_text_output, nurse_text_output, summary_text_output, audio_output]
    )
    
    gr.Markdown(
        """
        <div style='text-align: center; color: #64748b; margin-top: 2rem; font-size: 0.9rem;'>
            Disclaimer: This is an AI tool for educational purposes. Always consult a real medical professional for health issues.
        </div>
        """,
        sanitize_html=False
    )

if __name__ == "__main__":
    demo.launch(debug=True, show_api=False)