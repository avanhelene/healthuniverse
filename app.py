import streamlit as st
import pydeck as pdk
import numpy
import pandas as pd
import geopandas as gpd
import os
import json
import math
import plotly.express as px

geojson_dir = "data/geojson_states"
geojson_file = "data/service_area_df_with_population_counts.geojson"  # Replace with your file path

# Read the GeoJSON file as a GeoPandas DataFrame
cancer_center_service_areas = gpd.read_file(geojson_file)

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

data = [
    {"ID": 1, "Name": "Alice", "Age": 25, "City": "New York"},
    {"ID": 2, "Name": "Bob", "Age": 30, "City": "Los Angeles"},
    {"ID": 3, "Name": "Charlie", "Age": 35, "City": "Chicago"},
    {"ID": 4, "Name": "Diana", "Age": 40, "City": "Houston"},
]

st.set_page_config(layout="wide")

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
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 5px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%; /* Position above the icon */
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

cancer_type_options = ("All Site",
                        "Brain & ONS",
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
                        "Thyroid")

demographic_options = ("All", "American Indian/Alaska Native NH", 
                       "Asian/Pacific Islander NH",
                       "Black NH",
                       "Hispanic",
                       "White NH")

sex_options = ("All", "Male", "Female")

#selected_sites = []

# Create tabs
tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])

# Content for the first tab
with tab1:
    st.write("This is the first tab. Add your content here.")
    
# Content for the second tab (current content)
with tab2:
    #st.write(cancer_center_service_areas)
    # Create a two-column layout with the left column narrower
    left_column_top, right_column_top = st.columns([5, 2], gap="small")  # 1:3 width ratio
    with left_column_top:
        
        left_column_top_inner, left_middle_top_inner, right_column_top_inner = st.columns([1,1, 1], gap="small")
        
        
        with left_column_top_inner:
            st.markdown(
                """
                <label for="state-selection">
                    Select cancer incidence type
                    <span class="tooltip">ℹ
                        <span class="tooltiptext">Select one or more states to filter the data</span>
                    </span>
                </label>
                """,
                unsafe_allow_html=True,
                )   
            select_cancer_type = st.selectbox(
                "",
                cancer_type_options,
                index=0,
                label_visibility="collapsed",
            )

        with left_middle_top_inner:
            st.markdown(
                """
                <label for="demographic-selection">
                    Select cancer incidence patient demographic
                    <span class="tooltip">ℹ
                        <span class="tooltiptext">Select one or more states to filter the data</span>
                    </span>
                </label>
                """,
                unsafe_allow_html=True,
                )   
            select_demographic = st.selectbox(
                "",
                demographic_options,
                index=0,
                label_visibility="collapsed",)

        with right_column_top_inner:
            st.markdown(
                """
                <label for="sex-selection">
                    Select cancer incidence patient sex
                    <span class="tooltip">ℹ
                        <span class="tooltiptext">Select one or more states to filter the data</span>
                    </span>
                </label>
                """,
                unsafe_allow_html=True,
                )   
            select_sex = st.selectbox(
                "",
                sex_options,
                index=0,
                label_visibility="collapsed",
            )

        left_column_middle_inner, right_column_middle_inner = st.columns([1, 1], gap="small")

        with left_column_middle_inner:
            st.markdown(
                """
                <label for="center-type-selection">
                    Select cancer center types to incude
                    <span class="tooltip">ℹ
                        <span class="tooltiptext">Select cancer center types to incude</span>
                    </span>
                </label>
                """,
                unsafe_allow_html=True,
                )   
            select_site_types = st.multiselect(
                "",
                ["NCI-designated", "Academic", "Community", "Urban", "Rural"],
                default=["NCI-designated", "Urban", "Rural"],
                label_visibility="collapsed",
            )
        
        with right_column_middle_inner:
            st.markdown(
                """
                <label for="demographic-selection" style="margin-bottom: 5px; display: block;">
                    Select order of priority when sorting cancer centers by variable value
                    <span class="tooltip">ℹ
                        <span class="tooltiptext">Select one or more states to filter the data</span>
                    </span>
                </label>
                """,
                unsafe_allow_html=True,
                )   
            
            selection = st.segmented_control("", 
                ["Cancer incidence then Count of trials", "Count of trials then Cancer incidence"], 
                selection_mode="single",
                default=["Cancer incidence then Count of trials"],
                label_visibility="collapsed"
                )

        st.markdown(
            """
            <label for="">
               Filter cancer centers by state
                <span class="tooltip">ℹ
                    <span class="tooltiptext">Select one or more states to filter the data</span>
                </span>
            </label>
            """,
            unsafe_allow_html=True,
        )   
        state_selection = st.multiselect(
                "",
                ["Connecticut", "Maine", "Massachusetts", "New Hampshire", "New York", "Rhode Island", "Vermont", "New Jersey", "Pennsylvania"],
                default=["Connecticut", "Maine", "Massachusetts", "New Hampshire", "New York", "Rhode Island", "Vermont", "New Jersey", "Pennsylvania"],
                label_visibility="collapsed")
        

    
    with right_column_top:
            st.markdown(
                """
                <label for="">
                Control app behavior with LLM prompt
                    <span class="tooltip">ℹ
                        <span class="tooltiptext">Select one or more states to filter the data</span>
                    </span>
                </label>
                """,
                unsafe_allow_html=True,
            )   
            txt = st.text_area(
                "",
                "It was the best of times, it was the worst of times, it was the age of "
                "wisdom, it was the age of foolishness, it was the epoch of belief, it "
                "was the epoch of incredulity, it was the season of Light, it was the "
                "season of Darkness, it was the spring of hope, it was the winter of "
                "despair, (...)",
                height=123,
                max_chars=1000,
                label_visibility="collapsed"
                )
            left_buttom_top_inner, right_button_top_inner = st.columns([1,1.2], gap="small")
        
            with left_buttom_top_inner:
                st.button("Control app with LLM prompt", type="primary")
                
            with right_button_top_inner:
                st.button("Generate LLM summary of results", type="primary")
            
    st.markdown(
        """
        <hr style="border:1px solid #ccc; margin:10px 0;">
        """,
        unsafe_allow_html=True,
    )
    left_column_bottom, right_column_bottom = st.columns([4, 8], gap="small")  # 1:3 width ratio
    
        # Add content to the right column
    with right_column_bottom:
  
        incidence_column_start = select_demographic + "_" +  select_sex 
        cancer_center_service_areas_columns = cancer_center_service_areas.columns
        cancer_center_service_areas_columns = [item for item in cancer_center_service_areas_columns if item.startswith(incidence_column_start)]
        columns_to_display = ["place_name"]
        columns_to_display.extend(cancer_center_service_areas_columns)
        columns_to_display.extend(["address", "latitude", "longitude"])
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
        
        table_title = "Cancer Centers and " + select_sex + select_demographic + select_cancer_type + " cancer Incidence (*Select Sites by Clicking the Checkboxes in the Left Border of the Table*)"
        
        st.markdown(
            """<label for="">""" +
            table_title +
            """<span class="tooltip">ℹ<span class="tooltiptext">Select one or more states to filter the data</span>
                </span>
            </label>""",
            unsafe_allow_html=True,
            )   
        selected_sites = st.dataframe(
            df_to_display,
            #column_config=column_configuration,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
        )
        st.write(selected_sites.selection.rows)
    
    # Add content to the left column
    with left_column_bottom:
        map_tab, chart_tab = st.tabs(["Map", "Graphs"])

        # Content for the first tab
        with map_tab:
            geojson_files = load_geojson_files(geojson_dir, states = state_selection)
            #st.warning(geojson_files)
            
            filtered_df = cancer_center_service_areas.iloc[selected_sites.selection.rows]
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
                get_position=["longitude", "latitude"],
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
        with chart_tab:
            
            if filtered_df.shape[0] > 0:
                incidence_column = incidence_column_start + select_cancer_type
                   # Calculate the number of bins dynamically using Freedman-Diaconis Rule
                       # Calculate the number of bins dynamically using Freedman-Diaconis Rule
                data = filtered_df[incidence_column].dropna()  # Drop NaN values
                q25, q75 = data.quantile(0.25), data.quantile(0.75)  # Calculate IQR
                iqr = q75 - q25
                bin_width = 2 * iqr / (len(data) ** (1 / 3))  # Freedman-Diaconis formula
                num_bins = max(1, int((data.max() - data.min()) / bin_width))  # Ensure at least 1 bin

                # Create bins and calculate counts
                filtered_df["bins"] = pd.cut(filtered_df[incidence_column], bins=num_bins)
                bin_counts = filtered_df["bins"].value_counts().sort_index()

                # Convert bin counts to a DataFrame
                bin_counts_df = bin_counts.reset_index()
                bin_counts_df.columns = ["Bins", "Counts"]

                # Convert Bins to ranges with scientific notation for large numbers
                def format_bin_label(interval):
                    left = f"{interval.left:.2e}" if abs(interval.left) > 1e3 else f"{interval.left:.2f}"
                    right = f"{interval.right:.2e}" if abs(interval.right) > 1e3 else f"{interval.right:.2f}"
                    return f"{left}-{right}"

                bin_counts_df["Bins"] = bin_counts_df["Bins"].apply(format_bin_label)

                # Create a bar chart with Plotly
                fig = px.bar(
                    bin_counts_df,
                    x="Bins",
                    y="Counts",
                    title="Cancer Incidence within Centers's 2-hour Service Area Histogram",
                    labels={"Bins": "Value Ranges", "Counts": "Frequency"},
                )

                # Rotate x-axis labels
                fig.update_layout(xaxis_tickangle=45)

                # Display the chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)
   