import streamlit as st
import pydeck as pdk
import numpy
import pandas as pd
import geopandas as gpd
import os
import json
import math
import re
import requests
import time

st.set_page_config(layout="wide")

key = os.getenv("API_KEY")

geojson_dir = "data/geojson_states"

geojson_file = "data/service_area_df_with_population_and_trial_counts.geojson"  # Replace with your file path

# Read the GeoJSON file as a GeoPandas DataFrame
@st.cache_data
def load_geojson_data(filepath):
    return gpd.read_file(filepath)

# Use the cached function to load the GeoJSON file
cancer_center_service_areas = load_geojson_data(geojson_file)

control_ui_prompt = """
"You are an assistant embedded in a web application. Based on user input, you will output a JSON object that controls the behaviour of the app."
Below is the JSON format with all available options:
{
  "select_demographic": "All", "American Indian/Alaska Native NH", "Asian/Pacific Islander NH", "Black NH", "Hispanic", "White NH",
  "select_sex": "All", "Male", "Female"
  "select_site": ["Urban", "Rural/Suburban"],
  "state_selection:" ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "New York", "Rhode Island", "Vermont", "New Jersey", "Pennsylvania"]
}
Choose the options that best fit the user input. 'select_demographic' and 'select_sex' can only have one value each, while 'select_site' and 'state_selection' can have multiple values. 
Do not respond with any other text even if it makes sense to do so.

"""

def call_google_gemini(prompt,
                       api_key,
                       gemini_endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                       num_retries=3,
                       delay_after_api_failure=5):
    """
    Calls the Google Gemini API to generate content based on a given prompt.

    Args:
        prompt (str): The input prompt for the Gemini model.
        api_key (str): Your Google API key.
        gemini_endpoint (str): The endpoint for the Gemini API.
        num_retries (int): Number of retries in case of failure.
        delay_after_api_failure (int): Delay in seconds between retries.

    Returns:
        dict: The response from the Gemini API.
    """
    
    # construct the endpoint URL with the API key and define payload
    gemini_endpoint += f"?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [{
            "parts":[{"text": prompt}]
    }]
    }

    # retry logic for handling API failures
    for i in range(num_retries + 1):
        try:
            response = requests.post(gemini_endpoint, headers=headers, json=payload)
            if response.status_code == 200:
                output = response.json()
                text_output = output['candidates'][0]['content']['parts'][0]['text'].rstrip("\n")
                input_tokens = output['usageMetadata']['promptTokenCount']
                output_tokens = output['usageMetadata']['candidatesTokenCount']
                output = {
                    "text_output": text_output,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
                return output
            else:
                print(f"Error: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"Exception occurred: {e}")

        if i < num_retries:
            print(f"Retrying... ({i + 1}/{num_retries})")
            time.sleep(delay_after_api_failure)
        else:
            raise Exception(f"Failed to call Gemini API after {num_retries} retries")

# Function to load GeoJSON files
def load_geojson_files(directory, states):
    geojson_data = []
    states_with_extension = [state + ".geojson" for state in states]
    for filename in states_with_extension:
        if filename.endswith(".geojson"):  # Only process GeoJSON files
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as f:
                geojson_data.append(json.load(f))
    return geojson_data

# Function to calculate bounding box from GeoJSON data
def calculate_bounds(geojson_files):
    min_lat, min_lon = float("inf"), float("inf")
    max_lat, max_lon = float("-inf"), float("-inf")

    for geojson in geojson_files:
        for feature in geojson["features"]:
            coords = feature["geometry"]["coordinates"]
            if feature["geometry"]["type"] == "Polygon":
                for polygon in coords:
                    for lon, lat in polygon:
                        min_lat = min(min_lat, lat)
                        max_lat = max(max_lat, lat)
                        min_lon = min(min_lon, lon)
                        max_lon = max(max_lon, lon)
            elif feature["geometry"]["type"] == "MultiPolygon":
                for multipolygon in coords:
                    for polygon in multipolygon:
                        for lon, lat in polygon:
                            min_lat = min(min_lat, lat)
                            max_lat = max(max_lat, lat)
                            min_lon = min(min_lon, lon)
                            max_lon = max(max_lon, lon)

    return (min_lat, max_lat, min_lon, max_lon)



# Function to calculate zoom level based on bounds
def calculate_zoom(min_lat, max_lat, min_lon, max_lon, map_width=350, map_height=350, padding=0.3):
    # Constants for zoom calculation
    WORLD_DIM = {"height": 256, "width": 256}
    ZOOM_MAX = 20

    def lat_rad(lat):
        sin_val = math.sin(lat * math.pi / 180)
        rad_x2 = math.log((1 + sin_val) / (1 - sin_val)) / 2
        return max(min(rad_x2, math.pi), -math.pi) / 2

    def zoom(map_px, world_px, fraction):
        # Use a smoother scaling factor for zoom calculation
        return math.log2(map_px / world_px / fraction)

    # Apply padding to the bounds
    lat_padding = (max_lat - min_lat) * padding
    lon_padding = (max_lon - min_lon) * padding
    min_lat -= lat_padding
    max_lat += lat_padding
    min_lon -= lon_padding
    max_lon += lon_padding

    # Latitude and longitude differences
    lat_fraction = (lat_rad(max_lat) - lat_rad(min_lat)) / math.pi
    lon_fraction = (max_lon - min_lon) / 360

    # Calculate zoom levels for width and height
    lat_zoom = zoom(map_height, WORLD_DIM["height"], lat_fraction)
    lon_zoom = zoom(map_width, WORLD_DIM["width"], lon_fraction)

    # Use the minimum zoom level and limit it to ZOOM_MAX
    return min(lat_zoom, lon_zoom, ZOOM_MAX)

def get_highest_incidence_cancer_site(df, cancer_site_columns):
    """
    Calculate the cancer site with the highest total incidence.

    Args:
        df (pd.DataFrame): The DataFrame containing cancer site incidence data.
        cancer_site_columns (list): List of column names representing cancer sites.

    Returns:
        tuple: A tuple containing the cancer site with the highest incidence and its total value.
    """
    # Ensure the columns exist in the DataFrame
    existing_columns = [col for col in cancer_site_columns if col in df.columns]

    if not existing_columns:
        raise ValueError("None of the specified cancer site columns exist in the DataFrame.")

    # Calculate the total incidence for each cancer site
    total_incidence = df[existing_columns].sum()

    # Find the cancer site with the highest total incidence
    highest_incidence_site = total_incidence.idxmax()
    highest_incidence_value = total_incidence.max()

    return highest_incidence_site, highest_incidence_value

def calculate_urban_rural_percentage(df):
    """
    Calculate the percentage of urban centers compared to the total centers.

    Args:
        df (pd.DataFrame): The DataFrame containing the "Rurality" column.

    Returns:
        str: A formatted string reporting the percentage of urban centers.
    """
    if "Rurality" not in df.columns:
        return "The 'Rurality' column is missing in the DataFrame."

    # Count the number of urban and total centers
    urban_count = df[df["Rurality"] == "Urban"].shape[0]
    total_count = df.shape[0]

    if total_count == 0:
        return "No centers found. Cannot calculate the percentage."

    # Calculate the percentage
    urban_percentage = (urban_count / total_count) * 100

    # Round to the nearest whole number and return as a formatted string
    return round(urban_percentage)

st.markdown(
    """
    <style>
    /* Reduce top margin of the main container */
    .block-container {
        padding-top: 2rem;  /* Remove padding at the top */
        padding-bottom: 0rem;  /* Optional: Adjust bottom padding */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <hr style="border:1px solid #555; margin:10px 0;">
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 5px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        top: 125%; /* Position above the icon */
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <style>
        .st-e4 {{
            font-size: 15px;
            padding: 2px;
        }}
    </style>""",
    unsafe_allow_html=True,
)


demographic_options = ["All", "American Indian/Alaska Native NH", 
                       "Asian/Pacific Islander NH",
                       "Black NH",
                       "Hispanic",
                       "White NH"]

sex_options = ("All", "Male", "Female")

select_site_options = ["Urban", "Rural/Suburban"]

state_options = ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "New York", "Rhode Island", "Vermont", "New Jersey", "Pennsylvania"]

cancer_sites = ["Brain & ONS",
                   "Esophagus",
                   "Leukemia",
                   "Lung & Bronchus",
                   "Melanoma of the Skin",
                   "Oral Cavity & Pharynx",
                   "Pancreas",
                   "Stomach",
                   "Head and Neck",
                   "Bladder",
                   "Cervix",
                   "Colon & Rectum",
                   "Corpus Uteri & Uterus, NOS",
                   "Female Breast",
                   "Kidney & Renal Pelvis",
                   "Liver & IBD",
                   "Non-Hodgkin Lymphoma",
                   "Ovary",
                   "Prostate",
                   "Thyroid"]

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Map" 
    
# Initialize session state for the selectbox
if "select_demographic" not in st.session_state:
    st.session_state.select_demographic = "All"  # Default value

if "select_sex" not in st.session_state:
    st.session_state.select_sex = "All"  # Default value

if "select_site" not in st.session_state:
    st.session_state.select_site = ["Urban", "Rural/Suburban"] 
    
if "select_state" not in st.session_state:
    st.session_state.select_state = state_options
    
# Create tabs
tab1, tab2 = st.tabs(["Cancer Center Finder", "Tutorial"])

with tab1:
    #st.write(cancer_center_service_areas)
    # Create a two-column layout with the left column narrower
    left_column_top, right_column_top = st.columns([5, 2], gap="small")  # 1:3 width ratio
    
    
    with right_column_top:
        st.markdown(
            """
            <label for="">
            Control App Behavior with LLM Prompt
                <span class="tooltip">ℹ
                    <span class="tooltiptext">Control the app with free-text instructions instead of clicking buttons.</span>
                </span>
            </label>
            """,
            unsafe_allow_html=True,
        )   
        txt = st.text_area(
            "placeholder",
            "",
            height=80,
            max_chars=100,
            label_visibility="collapsed"
            )

        if st.button("Control App with Free-Text Instructions", type="primary") and len(txt) > 0:
            
            response = call_google_gemini(control_ui_prompt + txt,
                    api_key = key,
                    gemini_endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                    num_retries=3,
                    delay_after_api_failure=5)
            if type(response) == dict:
                json_string = response["text_output"]
        
                #cleaned_json_string = response["text_output"].strip('"')
                # Use a regular expression to extract the content inside the curly braces
                match = re.search(r"\{.*\}", json_string, re.DOTALL)
                if match:
                    json_string = match.group(0) 
                
                    # with open("data/test.txt", "w") as file:
                    #     file.write(json_string)
                    gemini_output = json.loads(json_string)
                    if all(item in ["select_demographic", "select_sex", "select_site", "state_selection"] for item in gemini_output.keys()):
                        try:
                            gemini_select_demographic = gemini_output["select_demographic"]
                            if gemini_select_demographic in demographic_options and len(gemini_select_demographic) > 0:
                                st.session_state.select_demographic = gemini_select_demographic
                            gemini_select_sex = gemini_output["select_sex"]
                            if gemini_select_sex in sex_options and len(gemini_select_sex) > 0:
                                st.session_state.select_sex = gemini_select_sex
                            gemini_select_site = gemini_output["select_site"]
                            if all(item in select_site_options for item in gemini_select_site) and len(gemini_select_site) > 0:
                                st.session_state.select_site = gemini_select_site
                            gemini_select_state = gemini_output["state_selection"] 
                            if all(item in state_options for item in gemini_select_state) and len(gemini_select_state) > 0:
                                st.session_state.select_state = gemini_select_state
                            # gemini_select_site = gemini_output["select_site"]
                            # gemini_state_selection = gemini_output["state_selection"]
                            
                        except Exception as e:
                            st.warning("LLM instruction failed. Please try again.")
                                

                    
                

                
            
    with left_column_top:
        
        left_column_inner, middle_column_inner, right_column_inner = st.columns([1, 1, 1], gap="small")
        with left_column_inner:
            st.markdown(
                """
                <label for="demographic-selection">
                    Select patient demographic to include in table
                    <span class="tooltip">ℹ
                        <span class="tooltiptext">This widget selects what demographic the cancer incidence data in the bottom right table corresponds to. For example, selecting 'Hispanic' will cause the table to show cancer incidence from the Hispanic population.</span>
                    </span>
                </label>
                """,
                unsafe_allow_html=True,
                )   
            
  

            # Button to update the selectbox value
            

            select_demographic = st.selectbox(
                "placeholder",
                demographic_options,
                index=demographic_options.index(st.session_state.select_demographic),
                label_visibility="collapsed", 
                key="select_demographic")

        with middle_column_inner:
            
    
            
            st.markdown(
                """
                <label for="sex-selection">
                    Select patient sex to include in table
                    <span class="tooltip">ℹ
                        <span class="tooltiptext">Control what sex the cancer incidence data in the bottom right table corresponds to. For example, selecting 'Male' will cause the table to show cancer incidence from the Male population.</span>
                    </span>
                </label>
                """,
                unsafe_allow_html=True,
                )   
            select_sex = st.selectbox(
                "placeholder",
                sex_options,
                index=sex_options.index(st.session_state.select_sex),
                label_visibility="collapsed",
                key="select_sex"
            )
        with right_column_inner:

            st.markdown(
                """
                <label for="center-type-selection">
                    Select cancer center rurality
                    <span class="tooltip">ℹ
                        <span class="tooltiptext">Filter cancer centers based on whether they are located in urban or rural/suburban areas.</span>
                    </span>
                </label>
                """,
                unsafe_allow_html=True,
                )   
            select_site_types = st.multiselect(
                "placeholder",
                select_site_options,
                #default=st.session_state.select_site,
                label_visibility="collapsed",
                key="select_site"
            )
        st.markdown(
            """
            <label for="">
               Filter cancer centers by state
                <span class="tooltip">ℹ
                    <span class="tooltiptext">Select one or more states to filter the data.</span>
                </span>
            </label>
            """,
            unsafe_allow_html=True,
        )   
        state_selection = st.multiselect(
                "placeholder",
                state_options,
                #default=st.session_state.select_state,
                label_visibility="collapsed",
                key="select_state")
            
    st.markdown(
        """
        <hr style="border:1px solid #ccc; margin:10px 0;">
        """,
        unsafe_allow_html=True,
    )
    left_column_bottom, right_column_bottom = st.columns([4, 8], gap="small")  # 1:3 width ratio
    
        # Add content to the right column
    with right_column_bottom:
        columns_to_display = ["Place Name", "Address", "Rurality"]
  
        incidence_column_start = select_demographic + "_" +  select_sex 
        cancer_center_service_areas_columns = cancer_center_service_areas.columns
        incidence_columns = [item for item in cancer_center_service_areas_columns if item.startswith(incidence_column_start)]
        
        columns_to_display.extend(incidence_columns)

        columns_to_display.extend([
            "Total Trial Count",
            "Leukemia Trial Count",                                              
            "Corpus Uteri & Uterus, NOS Trial Count",                            
            "Ovary Trial Count",                                                 
            "Head and Neck Trial Count",                                         
            "Colon & Rectum Trial Count",                                        
            "Female Breast Trial Count",                                         
            "Lung & Bronchus Trial Count",                                       
            "Brain & ONS Trial Count",                                           
            "Prostate Trial Count",                                              
            "Kidney & Renal Pelvis Trial Count",                                 
            "Kidney and Renal Pelvis Trial Count",                               
            "Melanoma of the Skin Trial Count",                                  
            "Pancreas Trial Count",                                              
            "Liver & IBD Trial Count",                                          
            "Oral Cavity & Pharynx Trial Count",                                 
            "Stomach Trial Count",                                               
            "Bladder Trial Count",                                               
            "Latitude", 
            "Longitude",
            "geometry"])
        
        cancer_center_service_areas = cancer_center_service_areas[cancer_center_service_areas["Rurality"].isin(select_site_types)]
        cancer_center_service_areas = cancer_center_service_areas[cancer_center_service_areas["State"].isin(state_selection)]
        if not cancer_center_service_areas.empty:
            df_to_display = cancer_center_service_areas[columns_to_display].copy()
            incidence_column_start = incidence_column_start + "_"
            df_to_display.columns = [col[len(incidence_column_start):] if col.startswith(incidence_column_start) else col for col in df_to_display.columns]

            #cancer_center_service_areas
            
            if select_sex == "All":
                select_sex = ""
            else:
                select_sex += " "
                
            if select_demographic == "All":
                select_demographic = ""
            else:
                select_demographic += " "
            
            table_title = "Cancer Centers and " + select_sex + select_demographic + " cancer Incidence (*Select Sites by Clicking the Checkboxes in the Left Border of the Table*)"
            
            st.markdown(
                """<label for="">""" +
                table_title +
                """<span class="tooltip">ℹ<span class="tooltiptext">Select one or more states to filter the data</span>
                    </span>
                </label>""",
                unsafe_allow_html=True,
                )   
        left_table_ui_column, right_table_ui_column = st.columns([1, 1], gap="small")  # 1:3 width ratio
    
        # Add content to the right column
        with left_table_ui_column:
            if not cancer_center_service_areas.empty:
                numeric_columns = df_to_display.select_dtypes(include=[numpy.number]).columns.tolist()
                numeric_columns = [col for col in numeric_columns if col not in ["Latitude", "Longitude"]]
                select_column_to_order = st.selectbox(
                        "Select column to order by (descending)",
                        numeric_columns,
                        index=0)
                df_to_display = df_to_display.sort_values(by=select_column_to_order, ascending=False)
        with right_table_ui_column:
            if not cancer_center_service_areas.empty:
                # Define the number of rows per page
                rows_per_page = 60

                # Calculate the total number of pages
                total_rows = len(df_to_display)
                total_pages = (total_rows + rows_per_page - 1) // rows_per_page  # Ceiling division

                # Add a widget to select the page number
                if total_pages > 1:
                    current_page = st.number_input(
                        f"Page (n = {str(total_pages)})",
                        min_value=1,
                        max_value=total_pages,
                        value=1,
                        step=1,
                        key="pagination"
                    )

                    # Calculate the start and end indices for the current page
                    start_idx = (current_page - 1) * rows_per_page
                    end_idx = start_idx + rows_per_page

                    # Slice the DataFrame for the current page
                    df_to_display = df_to_display.iloc[start_idx:end_idx]
        # Sort the DataFrame by the selected column in descending order
    # Sort the DataFrame by the selected column in descending order
        
        if not cancer_center_service_areas.empty:
        # Render the dataframe
            selected_sites = st.dataframe(
                df_to_display.drop(columns=["geometry"]),
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="multi-row",
                key="selected_rows"
            )

            # Update the selection state in session state
            if "selected_rows_state" not in st.session_state:
                st.session_state.selected_rows_state = []

            # Access the selected rows
            selected_rows = st.session_state.selected_rows_state
            if selected_sites.selection:
                st.session_state.selected_rows_state = selected_sites.selection.rows
            
          
        st.markdown(
            """
            Cancer indidence data is from [CancerInFocus](https://doi.org/10.1158/1055-9965.EPI-22-1319) and the list of clinical trials from the [HemOnc knowledgebase](https://doi.org/10.7910/DVN/FPO4HB).
            """,
            unsafe_allow_html=True,
        )
    # Add content to the left column
    with left_column_bottom:
        if not cancer_center_service_areas.empty:
            map_tab, llm_narrative = st.tabs(["Map", "LLM Narrative"])

            # Content for the first tab
            with map_tab:
                geojson_files = load_geojson_files(geojson_dir, states = state_selection)
                #st.warning(geojson_files)
                
                filtered_df = df_to_display.iloc[selected_sites.selection.rows]
                # Calculate bounds and center the map
                if geojson_files:
                    min_lat, max_lat, min_lon, max_lon = calculate_bounds(geojson_files)
                    center_lat = (min_lat + max_lat) / 2
                    center_lon = (min_lon + max_lon) / 2
                    zoom = calculate_zoom(min_lat, max_lat, min_lon, max_lon)
                else:
                    # Default view if no GeoJSON files are loaded
                    center_lat, center_lon, zoom = 37.7749, -122.4194, 4
                
                geojson_layers = [
                    pdk.Layer(
                        "GeoJsonLayer",
                        data=geojson,
                        get_fill_color="[200, 30, 0, 160]",  # Red fill color with transparency
                        get_line_color="[255, 255, 255]",  # White border color
                        line_width_min_pixels=1,
                        pickable=True,
                    )
                    for geojson in geojson_files
                ]
                
                filtered_geojson = json.loads(filtered_df.to_json())

                # Create a GeoJsonLayer for filtered_df
                filtered_layer = pdk.Layer(
                    "GeoJsonLayer",
                    data=filtered_geojson,
                    get_fill_color="[0, 100, 200, 160]",  # Blue fill color with transparency
                    get_line_color="[255, 255, 255, 100]",  # White border color
                    line_width_min_pixels=1,
                    pickable=True,
                )

                # Add the filtered_df layer to geojson_layers
                geojson_layers.append(filtered_layer)
                

                # marker_layer = pdk.Layer(
                #     "ScatterplotLayer",
                #     data=filtered_df,
                #     get_position=["longitude", "latitude"],  # Specify longitude and latitude columns
                #     get_fill_color="[173, 216, 230, 240]",  # Green fill color with transparency
                #     get_radius=10000,  # Radius of the markers
                #     pickable=True,  # Enable interactivity
                # )
                filtered_df["site_shape"] = "+"
                site_location_layer = pdk.Layer(
                    "TextLayer",
                    data=filtered_df,
                    get_position=["Longitude", "Latitude"],
                    get_text="site_shape",
                    get_size=20,  # Font size
                    get_color=[255, 255, 255],
                )
                # Add the marker layer to geojson_layers
                geojson_layers.append(site_location_layer)
                
                # Define the map view
                view_state = pdk.ViewState(
                    latitude=center_lat,  # Center latitude (San Francisco, for example)
                    longitude=center_lon,  # Center longitude
                    zoom=zoom,  # Zoom level
                    pitch=0,
                    height = 350
                )

                # Render the map with custom dimensions
                
                st.pydeck_chart(
                    pdk.Deck(
                        layers=geojson_layers,
                        map_style="mapbox://styles/mapbox/streets-v11",  # OpenStreetMap style
                        initial_view_state=view_state
                    )
                )
                with llm_narrative:
                    if st.button("Generate LLM Narrative Summary", type="primary") and df_to_display.shape[0] > 0:
                        make_narrative = True
                        cancer_results = get_highest_incidence_cancer_site(df_to_display, cancer_sites)
                        perc_urban = calculate_urban_rural_percentage(df_to_display)
                        centers = ", ".join(df_to_display["Place Name"].head(5).astype(str).tolist())
                        narrative_prompt = f"Give a concise summary of the following results. Do not mention demographic information other than sex. The site with the highest incidence is {cancer_results[0]} with a total incidence of {str(cancer_results[1])}. {perc_urban}% of the centers are urban. The selected states are {state_selection}. The selected demographic is {select_demographic}. The selected sex is {select_sex}. The 5 most relevant cancer centers to your query are: {centers}."

                        narrative = call_google_gemini(narrative_prompt,
                            api_key = key,
                            gemini_endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                            num_retries=3,
                            delay_after_api_failure=5)
                        st.write(narrative["text_output"])
with tab2:
    st.write("Welcome to the Cancer Center Finder App.")
    st.markdown("Summary:<br>This app helps clinical trial organizers find cancer centers to recruit patients from.<br>You find cancer centers by filtering based on center rurality and state.<br>Additionally, you can rank centers based on the cancer incidence within their 2-hour driving-time service areas or the number of clinical trials they have hosted.<br>The app can be controlled with free text instructions by leveraging LLMs.<br>This web app won first place during the 2025 Ci4CC hackathon.",
                unsafe_allow_html=True)   
    st.markdown(
        """
        <hr style="border:1px solid #555; margin:10px 0;">
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown("Filter the data you want displayed by patient demographic, sex, center rurality, and state.",
            unsafe_allow_html=True)  
    image_column_1, image_column_2 = st.columns([1.5, 1], gap="small")
    with image_column_1:
        st.image("data/screenshots_for_tutorial/filtering_widgets.png", 
                 caption="App Screenshot")
        st.markdown(
            """
            <hr style="border:1px solid #555; margin:10px 0;">
            """,
            unsafe_allow_html=True,
        )
    st.write("")
    st.markdown("You can also control the app with free-text instructions.<br>For example, you can type 'Show me cancer centers in New York and display cancer incidence for white women.'",
                unsafe_allow_html=True)
    image_column_3, image_column_4 = st.columns([0.33, 1], gap="small")
    with image_column_3:
        st.image("data/screenshots_for_tutorial/llm_app_control_widget.png", 
                 caption="App Screenshot")
    image_column_5, image_column_6 = st.columns([1.5, 1], gap="small")
    with image_column_5:
        st.markdown(
            """
            <hr style="border:1px solid #555; margin:10px 0;">
            """,
            unsafe_allow_html=True,
        )
    st.write("")
    image_column_7, image_column_8 = st.columns([1.5, 1], gap="small")
    with image_column_7:
        st.markdown("You can visualize the location and 2-hour driving-time service areas of cancer centers by checking the boxes in the table.",
            unsafe_allow_html=True)
        st.image("data/screenshots_for_tutorial/table_and_map.png", 
                 caption="App Screenshot")
        st.markdown(
            """
            <hr style="border:1px solid #555; margin:10px 0;">
            """,
            unsafe_allow_html=True,
        )
    st.write("")
    st.markdown("Use the Google Gemini LLM to generate a narrative summary of the results.",
            unsafe_allow_html=True)
    image_column_9, image_column_10 = st.columns([0.5, 1], gap="small")
    with image_column_9:
        st.image("data/screenshots_for_tutorial/llm_narrative.png", 
                caption="App Screenshot")
    image_column_5, image_column_6 = st.columns([1.5, 1], gap="small")
     
            