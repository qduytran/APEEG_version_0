import streamlit as st
import os
# import time # Not typically needed for interactive Streamlit apps

# Import your utility functions
from utils_store.M01_DataLoader import ui_eeg_subjects_uploader, delete_folder, ui_select_subject, ui_eeg_groups_uploader
from utils_store.M02_PSDTransform import UI_plot_psd
from utils_store.M03_FeatureExtraction import UI_feature_extraction, ui_plot_topo, UI_feature_extraction_groups, ui_adjust_param_fooof, plot_fooof
from utils_store.M04_Classification import UI_predict_ml, UI_train_ml

# --- Page Configuration ---
st.set_page_config(
    page_title="EEG Analysis Pro",
    page_icon="🧠",  # You can use an emoji or a path to an image
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
# You can expand this CSS significantly for more detailed styling
st.markdown("""
    <style>
        /* Main app background */
        .stApp {
            /* background-color: #f0f2f6; */ /* Light gray background */
        }

        /* Sidebar styling */
        .css-1d391kg { /* This class might change with Streamlit versions, inspect if needed */
            /* background-color: #2c3e50; /* Dark blue-gray sidebar */
            /* padding: 1rem; */
        }
        .css-1d391kg .stButton>button { /* Sidebar buttons */
            color: #ffffff; /* White text */
            background-color: #3498db; /* Blue */
            border-radius: 8px;
            padding: 10px 15px;
            width: 100%;
            text-align: left;
            margin-bottom: 8px;
            border: none;
            font-weight: 500;
            transition: background-color 0.3s ease;
        }
        .css-1d391kg .stButton>button:hover {
            background-color: #2980b9; /* Darker blue on hover */
        }
        .css-1d391kg .stButton>button:focus {
            outline: none;
            box-shadow: 0 0 0 2px #ffffff, 0 0 0 4px #3498db; /* Focus ring */
        }
        .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3 { /* Sidebar headers */
            color: #ecf0f1; /* Light text for dark background */
        }

        /* Main content headers */
        h1, h2, h3 {
            /* color: #2c3e50; /* Dark blue-gray for main content headers */
        }

        /* Style for selected sidebar button (more advanced, requires JS or more complex CSS) */
        /* For simplicity, we'll rely on Streamlit's default behavior or a simpler visual cue */

    </style>
    """, unsafe_allow_html=True)


# --- App Title ---
st.title("🧠 EEG Analysis Pro")
st.caption("Your all-in-one platform for EEG data processing and machine learning.")
st.markdown("---")


# --- Initialize Session State for Current Page ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"  # Default page

# --- Function to Change Page ---
def set_page(page_name):
    st.session_state.current_page = page_name

# --- Navigation Sidebar ---
with st.sidebar:
    st.header("Navigation Panel")
    # Using a dictionary for cleaner button creation with icons
    pages = {
        "Home": "🏠 Home",
        "Load Data": "📤 Load & View EEG Data",
        "Feature Extraction": "🛠️ Feature Extraction",
        "Train Model": "🎓 Train ML Model",
        "Predict": "🔮 Make Predictions"
    }

    for page_key, page_display_name in pages.items():
        if st.button(page_display_name, on_click=set_page, args=(page_key,), key=f"btn_{page_key}", use_container_width=True):
            pass # on_click handles the action

    st.markdown("---")
    st.info("💡 Tip: Upload your EEG data in the 'Load Data' section to begin.")
    # You could add a logo here:
    # st.image("path/to/your/logo.png", width=150)


# --- Display Content Based on Current Page ---
if st.session_state.current_page == "Home":
    st.header("Welcome to the EEG Analysis Dashboard!")
    st.markdown("""
        <div style="background-color:#eaf5ff; padding: 20px; border-radius: 10px; border: 1px solid #cce0ff;">
        <p style="font-size: 1.1em;">
        This dashboard provides a comprehensive suite of tools for analyzing EEG data.
        Navigate through the different sections using the sidebar to:
        </p>
        <ul>
            <li>📤 <strong>Load and Visualize</strong> your EEG recordings.</li>
            <li>🛠️ <strong>Extract Meaningful Features</strong> from the signals.</li>
            <li>🎓 <strong>Train Machine Learning Models</strong> for classification or regression tasks.</li>
            <li>🔮 <strong>Make Predictions</strong> on new, unseen data.</li>
        </ul>
        <p style="font-size: 1.1em;">
        Select an option from the navigation panel on the left to get started.
        </p>
        </div>
    """, unsafe_allow_html=True)
    st.subheader("Getting Started")
    col1, col2 = st.columns(2)
    with col1:
        st.info("New to EEG analysis? Start by uploading your data in the 'Load & View EEG Data' section.")
    with col2:
        st.success("Have features and want to train a model? Head to 'Train ML Model'.")


elif st.session_state.current_page == "Load Data":
    st.header("📤 Load and Explore EEG Data")
    input_path = "input/temp_rawData"  # Define when needed

    st.subheader("1. Load Single Subject Data")
    # Initialize session state for single subject data
    if 'raw_dataset_single' not in st.session_state:
        st.session_state.raw_dataset_single = None
    if 'raw_data_selected' not in st.session_state:
        st.session_state.raw_data_selected = None

    st.session_state.raw_dataset_single = ui_eeg_subjects_uploader(input_path=input_path)

    if st.session_state.raw_dataset_single:
        st.session_state.raw_data_selected = ui_select_subject(raw_dataset=st.session_state.raw_dataset_single)

        if st.session_state.raw_data_selected:
            st.markdown("### Power Spectral Density (PSD) & FOOOF Analysis")
            with st.expander("Show PSD and FOOOF Plots", expanded=True):
                freqs, psd, selected_channels = UI_plot_psd(st.session_state.raw_data_selected)

                if freqs is not None and psd is not None:
                    channel_names = st.session_state.raw_data_selected.ch_names
                    # Note: Ensure ui_adjust_param_fooof() is also translated and styled if it has UI elements
                    pe_settings, ape_settings = ui_adjust_param_fooof()

                    selected_channel_fooof = st.selectbox(
                        "Select Channel for FOOOF Fitting:",
                        channel_names,
                        key="fooof_channel_selector_single" # Unique key
                    )
                    _, ffitting_fig = plot_fooof(
                        freqs=freqs,
                        psds=psd,
                        raw_data=st.session_state.raw_data_selected,
                        pe_settings=pe_settings,
                        ape_settings=ape_settings,
                        channel_names=[selected_channel_fooof],
                        show_fplot=True
                    )
                    st.pyplot(ffitting_fig)
                else:
                    st.info("PSD data not available or not plotted yet.")
    st.markdown("---")
    # (Optional) Group data loading section
    # st.subheader("2. Load Group Data")
    # with st.expander("Upload EEG Data for Multiple Groups (Optional)"):
    #     ui_eeg_groups_uploader(input_path=input_path)

elif st.session_state.current_page == "Feature Extraction":
    st.header("🛠️ EEG Feature Extraction")

    if 'raw_dataset_single' in st.session_state and st.session_state.raw_dataset_single:
        st.subheader("Extract Features for Single Subjects")
        with st.expander("Feature Extraction and Topomap", expanded=True):
            if 'features_subjects_single' not in st.session_state:
                st.session_state.features_subjects_single = None

            st.session_state.features_subjects_single = UI_feature_extraction(raw_dataset=st.session_state.raw_dataset_single)
            # if st.session_state.features_subjects_single is not None:
            ui_plot_topo(raw_dataset=st.session_state.raw_dataset_single, features_subjects=st.session_state.features_subjects_single)
            # else:
            #     st.info("Features not extracted yet or no data available.")
    else:
        st.warning("Please load EEG data in the 'Load & View EEG Data' section before extracting features.")

    st.markdown("---")
    # (Optional) Group feature extraction section
    # st.subheader("Extract Features for Groups")
    # with st.expander("Group-Level Feature Extraction (Optional)"):
    #     UI_feature_extraction_groups() # Ensure this function has appropriate UI for data input

elif st.session_state.current_page == "Train Model":
    st.header("🎓 Train Machine Learning Model")
    st.info("Upload your feature set and labels to train a classification or regression model.")
    with st.expander("Model Training Interface", expanded=True):
        UI_train_ml() # Ensure this function provides UI for file uploads and parameter settings

elif st.session_state.current_page == "Predict":
    st.header("🔮 Make Predictions")
    st.info("Load a pre-trained model and new EEG data (or features) to generate predictions.")
    with st.expander("Prediction Interface", expanded=True):
        UI_predict_ml() # Ensure this function provides UI for model and data uploads

# --- Footer or Cleanup Notes ---
st.sidebar.markdown("---")
st.sidebar.caption("© 2024 EEG Analysis Pro. All rights reserved.")
st.sidebar.caption("Developed by: Nguyen Duc Kien")

# Regarding delete_folder:
# Automatic deletion in Streamlit is tricky. Consider these:
# 1. Clean up within `ui_eeg_subjects_uploader` before new uploads.
# 2. Add a manual "Clear Temporary Data" button.
# For now, automatic deletion is omitted for stability.