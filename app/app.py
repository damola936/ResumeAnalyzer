import streamlit as st
import fitz
import warnings
from huggingface_hub import login
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from streamlit.runtime.uploaded_file_manager import UploadedFile
import pandas as pd
import spacy
import ollama
import matplotlib.pyplot as plt
import seaborn as sns
import math
import collections # Import collections for counting tags
from torch import Tensor
from wordcloud import WordCloud # Import WordCloud for word cloud generation
from streamlit_pdf_viewer import pdf_viewer # Import the PDF viewer component
import re

warnings.filterwarnings("ignore")



# ---- MARKDOWNS/CSS/APP-CONFIG-----------------
st.set_page_config(page_title="Resume Analyzer", layout="centered")

# Initialize session state for modal and analysis results
if 'show_pdf_modal' not in st.session_state:
    st.session_state.show_pdf_modal = False
if 'selected_profile_for_modal' not in st.session_state:
    st.session_state.selected_profile_for_modal = None
if 'profiles' not in st.session_state:
    st.session_state.profiles = []
if 'sorted_profiles' not in st.session_state:
    st.session_state.sorted_profiles = []
if 'analysis_performed' not in st.session_state:
    st.session_state.analysis_performed = False


# Custom CSS for modern styling
st.markdown("""
<style>
    /* General body styling */
    body {
        font-family: 'Inter', sans-serif;
        background-color: #1E1E2F;
        color: #E0E0E0;
    }

    .block-container {
        max-width: 85% !important;
        margin: auto !important;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #C77DFF;
    }

    hr {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        border: 0;
        border-top: 1px solid #444;
    }

    .stAlert.stInfo {
        background-color: #2A2A40;
        color: #C77DFF;
        border-color: #5E3AAE;
    }

    .stFileUploader label,
    .stTextArea label,
    .stSlider label {
        font-weight: bold;
        color: #C77DFF;
    }

    .stButton>button {
        background-color: #C77DFF;
        color: #1E1E2F;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #9D4EDD;
        color: #000000
    }

    .stSpinner > div > div {
        border-top-color: #C77DFF;
    }

    .badge {
        display: inline-block;
        padding: 6px 10px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #fff;
        background-color: #7B2CBF;
        border-radius: 15px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 120px;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }

    .badge.match {
        background-color: #C77DFF;
        font-weight: 700;
    }

    .tags-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        justify-content: center;
        margin-top: 10px;
        margin-bottom: 10px;
        padding: 10px;
        border: 1px solid #3A3A4F;
        border-radius: 10px;
        background-color: #2A2A40;
        min-height: 40px;
        align-items: center;
        box-shadow: inset 0 1px 3px rgba(255,255,255,0.05);
    }

    .no-tags {
        font-style: italic;
        color: #AAAAAA;
        font-size: 0.85rem;
        text-align: center;
        grid-column: 1 / -1;
    }

    .card {
        padding: 25px;
        border: 1px solid #3F3F5A;
        border-radius: 18px;
        box-shadow: 4px 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 25px;
        text-align: center;
        background-color: #2A2A40;
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .card:hover {
        transform: translateY(-7px);
        box-shadow: 6px 6px 20px rgba(0,0,0,0.6);
    }

    .card h3 {
        margin-top: 0;
        color: #F15BB5;
        margin-bottom: 10px;
    }

    .card p strong {
        color: #C77DFF;
    }

    .card p {
        margin-bottom: 6px;
        font-size: 1rem;
    }

    .button-container {
        text-align: center;
        margin-top: auto;
        padding-top: 10px;
    }

    .stPlotlyChart, .stDeckGlChart, .stPyDeckChart, .stMap {
        margin-top: 20px;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.4);
    }
</style>
""", unsafe_allow_html=True)



#------CLASSES---------------------------------
class Profile:
    def __init__(self, resume: UploadedFile, job_description: str | UploadedFile):
        self.resume = resume
        self.job_description = job_description
        self.job_description_tags = None
        self.resume_text = None
        self.experience = None
        # Handle potential file name issues and remove extension
        try:
            self.name = os.path.splitext(self.resume.name)[0]
        except Exception as e:
            st.warning(f"A resume has no name with error: {e}")
            self.name = "Unnamed Resume" # Default name if extraction fails
        self.score = None
        self.tags = None
        self.get_text(self.resume)
        # Ensure resume_text is not None before processing
        if self.resume_text:
            self.get_score(self.resume_text, self.job_description)
            self.get_tags()
            self.get_experience_val(self.resume_text)
        else:
            self.score = 0 # Assign a default score if text extraction fails
            self.tags = []
            self.job_description_tags = []

    def get_text(self, resume_file) -> None:
        try:
            # Read the file bytes
            file_bytes = resume_file.read()
            if not file_bytes:
                resume_file.seek(0)
                file_bytes = resume_file.read()
            # Open with PyMuPDF from bytes
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text = ''
                for page in doc:
                    text += page.get_text() + '\n'
                self.resume_text = text.strip()
        except Exception as e:
            st.error(f"Error reading PDF file {resume_file.name}: {e}")
            self.resume_text = None  # Set to None if extraction fails

    def get_score(self, resume_text, job_description) -> None:
        try:
            job_text = read_job_description(job_description)
            if not job_text:
                 self.score = 0
                 st.warning("Job description is empty, cannot calculate score.")
                 return

            jd_embed = generate_embedding(job_text)
            res_embed = generate_embedding(resume_text)
            self.score = round(cosine_similarity(jd_embed, res_embed))
        except Exception as e:
            st.error(f"Error calculating score for {self.name}: {e}")
            self.score = 0 # Assign a default score on error


    def get_tags(self) -> None:
        try:
            # Load the small English model
            nlp = spacy.load('en_core_web_sm')
            # Process resume text and extract entities
            resume_doc = nlp(self.resume_text)
            # Extract entities and their labels
            resume_tags = [(ent.text, ent.label_) for ent in resume_doc.ents]

            # Process job description text and extract entities
            desc_text = read_job_description(self.job_description)
            desc_doc = nlp(desc_text)
            # Extract entities and their labels
            desc_tags = [(ent.text, ent.label_) for ent in desc_doc.ents]

            # Filter tags based on entity labels and length
            # Consider common entity types relevant to resumes/job descriptions
            relevant_labels = ['GPE','ORG', 'PERSON', 'PRODUCT', 'EVENT', 'LANGUAGE', 'SKILL', 'JOB_TITLE', 'EDUCATION'] # Added more relevant labels
            self.tags = [t for t,l in resume_tags if l in relevant_labels and 2 < len(t) < 30] # Increased max tag length slightly
            self.job_description_tags = [t for t,l in desc_tags if l in relevant_labels and 2 < len(t) < 30] # Increased max tag length slightly

        except Exception as e:
            st.error(f"Error extracting tags for {self.name}: {e}")
            self.tags = []
            self.job_description_tags = []
    

    def get_experience_val(self, resume_text) -> None:
        system_message = f"""
                          ## Instructions
                          # You are an assistant that analyzes and returns the years of experience of a person when given resume content
                          # Respond briefly only with the number of years of experience.
                            # E.g The number of years of experience for this resume content: .... is what?
                            # You respond with: 5 years of experience
                        """
        assistant_message = "Respond briefly only with the number of years of experience:\n\n"
        user_message = f"\n\nThe content of the resume is:\n\n{resume_text}\n\n"
        user_message += "respond briefly with the number of years of experience the person of the resume has"
        messages = [{"role": "system", "content": system_message},
                     {"role":"user", "content": user_message}, {"role": "assistant", "content": assistant_message}]
        with st.spinner("🔍 Getting years of experience, please be patient..."):
            try:
                response = get_experience_no(ollama.chat(model="deepseek-r1:7b",
                                                        messages=messages)["message"]["content"].split("</think>")[-1])
            except Exception as e:
                print(f"Error getting response, Error detail:\n {e}")
                st.error(f"Error getting response, Error detail:\n {e}")
                self.experience = 0
            else:
                self.experience = abs(response)


# -------FUNCTIONS -----------------------------------

def hugging_face_login() -> None:
    load_dotenv(override=True)
    # Check if HF_TOKEN is already set in environment
    if 'HF_TOKEN' not in os.environ or not os.environ['HF_TOKEN']:
        st.warning("Hugging Face token not found. Embedding generation might fail.")
        # Optionally, prompt user for token here if needed
    else:
        login(os.environ['HF_TOKEN'], add_to_git_credential=True)


def read_job_description(file: str | UploadedFile) -> str:
    if isinstance(file, UploadedFile):
        try:
            if file.name.endswith('.pdf'):
                file_bytes = file.read()
                if not file_bytes:
                    file.seek(0)  # Reset the stream and try reading again
                    file_bytes = file.read()
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    file_content = "\n".join([page.get_text() for page in doc])
                    return file_content.strip()
            else:
                file.seek(0)  # Make sure the pointer is at the beginning
                return file.read().decode('utf-8').strip()
        except Exception as e:
            st.error(f"Error reading job description file → {file.name}: {e}")
            return ""  # Return empty string on error
    return file.strip()


def cosine_similarity(a, b) -> int|float:
    # Avoid division by zero if one of the vectors is zero
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    # Calculate and return cosine similarity as a percentage
    return round(np.dot(a,b)/(norm_a*norm_b)*100, 2)


@st.cache_resource
def generate_embedding(text, model='sentence-transformers/all-MiniLM-L6-v2')\
        -> list[Tensor] | np.ndarray | Tensor | dict[str, Tensor] | list[dict[str, Tensor]]:
    hugging_face_login()
    try:
        # Initialize and return the sentence transformer model
        model = SentenceTransformer(model)
        return model.encode(text)
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        # Return a zero vector or handle appropriately, dimension is 384 for this model
        return np.zeros(384)


# Function to handle button click and set session state
def view_resume_clicked(profile) -> None:
    st.session_state.selected_profile_for_modal = profile
    st.session_state.show_pdf_modal = True


# pdf modal
@st.dialog("Resume Content")
def display_pdf_content(selected_profile) -> None:
    st.write(f"Viewing: {selected_profile.name}")
    # Use pdf_viewer with the bytes content of the uploaded file
    pdf_viewer(selected_profile.resume.getvalue(), width=1000, height=800)
    # Add a close button to the modal
    if st.button("Close", key="close_modal_button"):
        st.session_state.show_pdf_modal = False
        st.session_state.selected_profile_for_modal = None # Clear the selected profile
        st.rerun() # Rerun to close the modal


# Function to display candidate cards and dashboard charts
def display_analysis_results(sorted_profiles: list[Profile]) -> None:
    if not sorted_profiles:
        st.warning("No candidates to display.")
        return

    # Get job description tags from the first profile if available
    desc_tags = set(sorted_profiles[0].job_description_tags) if sorted_profiles and sorted_profiles[0].job_description_tags else set()

    num_cols = 3
    rows = math.ceil(len(sorted_profiles) / num_cols)
    idx = 0

    st.markdown(f'#### Top {len(sorted_profiles)} Candidates Overview')
    with st.container(border=True):
        for _ in range(rows):
            cols = st.columns(num_cols, gap='large')
            for col in cols:
                if idx >= len(sorted_profiles):
                    break

                prof = sorted_profiles[idx]
                unique_tags = list(set(prof.tags)) if prof.tags is not None else []
                unique_tags = unique_tags[:9] # Limit to 9 tags

                if not unique_tags:
                    tags_content_html = "<span class='no-tags'>No relevant tags found</span>"
                else:
                    tags_content_html = "".join([
                        f"<span class='{'badge match' if tag in desc_tags else 'badge'}'>{tag}</span>"
                        for tag in unique_tags
                    ])

                card_html = f"""
                <div class='card'>
                    <div>
                        <h3>{prof.name}</h3>
                        <p><strong>Score:</strong> {prof.score}%</p>
                        <p>Years of experience: {prof.experience}</p>
                        <p>Tags:</p>
                    </div>
                    <div class='tags-container'>
                        {tags_content_html}
                    </div>
                    <div class='button-container'>
                        <span>
                        </div>
                </div>
                """

                with col:
                    st.markdown(card_html, unsafe_allow_html=True)
                    st.button("🔍View Resume",
                            key=f"view_resume_{prof.name}_{idx}",
                                on_click=view_resume_clicked, args=(prof,))

                idx += 1

    st.divider()
    st.header("Analysis Dashboard")

    chart_col1, chart_col2 = st.columns(2, gap='large')
    chart_col3, chart_col4 = st.columns(2, gap='large')

    # --- Chart 1 (Top Left): Scores Across Selected Candidates ---
    with chart_col1:
        st.subheader('Scores Across Selected Candidates')
        df_scores = pd.DataFrame({
            'Name': [p.name for p in sorted_profiles],
            'Score': [p.score for p in sorted_profiles]
        }).sort_values('Score', ascending=False)
        if not df_scores.empty:
            st.bar_chart(df_scores.set_index('Name')['Score'], color="#C77DFF")
        else:
            st.info("Not enough data for Scores Across Selected Candidates chart.")

    # --- Chart 2 (Top Right): Tag Frequency Bar Chart ---
    with chart_col2:
        st.subheader('Most Frequent Tags')
        all_tags = [tag for prof in sorted_profiles for tag in (prof.tags if prof.tags is not None else [])]
        tag_counts = collections.Counter(all_tags)
        most_common_tags = tag_counts.most_common(15)
        if most_common_tags:
            df_tags = pd.DataFrame(most_common_tags, columns=['Tag', 'Frequency'])
            st.bar_chart(df_tags.set_index('Tag')['Frequency'], color="#F15BB5")
        else:
            st.info("No significant tags found in selected resumes for frequency analysis.")

    # --- Chart 3 (Bottom Left): Score vs. Number of Tags Scatter Plot ---
    with chart_col3:
        st.subheader('Score vs. Number of Tags')
        scatter_data = {
            'Score': [p.score for p in sorted_profiles],
            'Number of Tags': [len(p.tags) if p.tags is not None else 0 for p in sorted_profiles],
            'Name': [p.name for p in sorted_profiles]
        }
        df_scatter = pd.DataFrame(scatter_data)
        if not df_scatter.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df_scatter, x='Number of Tags', y='Score', ax=ax, c="#F15BB5")
            ax.set_title('Score vs. Number of Tags')
            ax.set_xlabel('Number of Tags Found')
            ax.set_ylabel('Score (%)')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Not enough data to generate Score vs. Number of Tags scatter plot.")

    # --- Chart 4 (Bottom Right): Tag Word Cloud ---
    with chart_col4:
        st.subheader('Tag Word Cloud')
        if all_tags:
            text = " ".join(all_tags)
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                colormap='Blues',
                collocations=False,
                min_font_size=10
            ).generate(text)
            fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            st.pyplot(fig_wc)
            plt.close(fig_wc)
        else:
            st.info("No tags available to generate a word cloud.")


def get_experience_no(s) -> int:
    s = s.replace('$','').replace(',','')
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return int(match.group()) if match else 0


# ----- APP -----------------

text_upload = "Enter Text"
file_upload = "Upload File"

st.divider()
st.title("📄 Resume Analyzer")
st.header("Analyze Resumes for Businesses and Recruitment")
st.divider()
st.subheader("Provide the Job Description and Upload Resumes (PDF)")

st.info("💡 **Tip:** For best results, ask candidates to name their resume files with their full names.")

job_description_text = None
job_description_uploaded_file = None

job_description_input_type = st.radio(
        "**Select job description input type:**",
        (text_upload, file_upload),
        key="jd_input_type_radio",
        horizontal=True
    )

if job_description_input_type == text_upload:
        job_description_text = st.text_area("**Enter job description:**", height=200,
                                             key="jd_text_area",
                                               help="Paste the text of the job description here.")
elif job_description_input_type == file_upload:
        job_description_uploaded_file = st.file_uploader("**Upload job description:**", type=["txt", "pdf"],
                                                          key="jd_file_uploader",
                                                         help="Upload a .txt or .pdf file containing the job description.")

uploaded_resumes = st.file_uploader(label="**Drag/Upload Resumes (PDF):**", type=["pdf"],
                                     accept_multiple_files=True, key="resume_upload",
                                       help="Upload one or more resume files in PDF format.")

# Determine if the form should be disabled
form_disabled = True
if job_description_input_type == text_upload and job_description_text and uploaded_resumes:
    form_disabled = False
elif job_description_input_type == file_upload and job_description_uploaded_file and uploaded_resumes:
    form_disabled = False

with st.sidebar:
    st.header("🪛Filter candidates")
    amount = st.slider("**Analyze for Top:**", 0, 20, value=min(5, len(uploaded_resumes) if uploaded_resumes else 0),
                        key="amount_slider", help="Select the number of top candidates to analyze.")
    years_of_experience = st.slider("**At least what Years of Experience?**", 0, 5,
                                    key="experience_slider", help="Select your preferred  candidate years of experience.")
    print(years_of_experience)

analyze = st.button("🚀 Analyze Resumes", disabled=form_disabled, key="analyze_button")
st.divider()

# --- Analysis Logic ---
# Only perform analysis if the analyze button is clicked
if analyze:
    if amount == 0:
        st.warning("Please select a number of candidates greater than 0 to analyze.")
    elif not uploaded_resumes:
        st.warning("Please upload resumes to analyze.")
    else:
        # Create profiles based on current inputs
        current_profiles = []
        if job_description_input_type == text_upload:
            current_profiles = [Profile(resume=resume, job_description=job_description_text) for resume in uploaded_resumes]
        elif job_description_input_type == file_upload:
            current_profiles = [Profile(resume=resume, job_description=job_description_uploaded_file) for resume in uploaded_resumes]

        valid_profiles = [p for p in current_profiles if p.resume_text is not None]
        valid_profiles = [p for p in valid_profiles if p.experience is not None and p.experience >= years_of_experience]
        invalid_profiles_with_less_experience = [p for p in current_profiles if p not in valid_profiles if p.score > 50]

        if len(invalid_profiles_with_less_experience) > 0:
            st.info("💡 There are candidates with high similarity compatibility scores that are not selected"
                    " due to low years of experience."
                    " Lower experience requirements to potentially view them")

        if not valid_profiles:
            st.warning("No valid resumes were processed. Please check the uploaded files and job description.")
        else:
            # Sort and limit profiles
            sorted_profiles = sorted(valid_profiles, key=lambda p: p.score, reverse=True)[:amount]

            # Store analysis results in session state
            st.session_state.profiles = valid_profiles # Store all valid profiles
            st.session_state.sorted_profiles = sorted_profiles # Store the sorted/limited profiles for display
            st.session_state.analysis_performed = True # Set flag that analysis was done

# --- Display Results ---
# Display the dashboard and cards if analysis has been performed and there are sorted profiles
if st.session_state.analysis_performed and st.session_state.sorted_profiles:
    display_analysis_results(st.session_state.sorted_profiles)
elif st.session_state.analysis_performed and not st.session_state.sorted_profiles:
     st.info("No candidates match the criteria after analysis.")


# --- Modal Display ---
# Display the modal if show_pdf_modal is True and a profile is selected
if st.session_state.show_pdf_modal and st.session_state.selected_profile_for_modal:
    display_pdf_content(st.session_state.selected_profile_for_modal)

