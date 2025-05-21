import streamlit as st
import numpy as np
import plotly.graph_objects as go
from functions import *

st.set_page_config(layout="wide")

# Streamlit inputs
st.title('Likelihood Surface for Two Classes')

# User inputs for labels y1, y2
y1 = st.selectbox('Select label for y₁', [0, 1])
y2 = st.selectbox('Select label for y₂', [0, 1])

# User input for lambda
lambda_val = st.slider('Select lambda', 0., 1.0, 0.01)

# Create a meshgrid for p₁, p₂ values
p1_vals = np.linspace(0.01, 0.99, 200)  # Increased from 100 to 200 points
p2_vals = np.linspace(0.01, 0.99, 200)  # Increased from 100 to 200 points
P1, P2 = np.meshgrid(p1_vals, p2_vals)

# Compute likelihoods for all functions over the meshgrid
BCE_LIKELIHOOD = np.array([[bce_likelihood(p1, p2, y1, y2) for p1, p2 in zip(p1_row, p2_row)] for p1_row, p2_row in zip(P1, P2)])
ANY_CLASS_LIKELIHOOD = np.array([[any_class_likelihood(p1, p2, y1, y2, lambda_val) for p1, p2 in zip(p1_row, p2_row)] for p1_row, p2_row in zip(P1, P2)])
PROD_LIKELIHOOD = np.array([[any_class_likelihood_prod(p1, p2, y1, y2, lambda_val) for p1, p2 in zip(p1_row, p2_row)] for p1_row, p2_row in zip(P1, P2)])

# Calculate combined likelihoods
MODIFIED_LIKELIHOOD = BCE_LIKELIHOOD * ANY_CLASS_LIKELIHOOD
MODIFIED_PROD_LIKELIHOOD = BCE_LIKELIHOOD * PROD_LIKELIHOOD

# Compute product and geometric mean of probabilities
PRODUCT_PROBS = np.array([[prod_p(p1, p2) for p1, p2 in zip(p1_row, p2_row)] for p1_row, p2_row in zip(P1, P2)])
GEO_MEAN_PROBS = np.array([[geo_mean(p1, p2) for p1, p2 in zip(p1_row, p2_row)] for p1_row, p2_row in zip(P1, P2)])
    
# Calculate losses
BCE_LOSS = -np.log(BCE_LIKELIHOOD)
ANY_CLASS_LOSS = -np.log(ANY_CLASS_LIKELIHOOD)
PROD_LOSS = -np.log(PROD_LIKELIHOOD)
MODIFIED_LOSS = -np.log(MODIFIED_LIKELIHOOD)
MODIFIED_PROD_LOSS = -np.log(MODIFIED_PROD_LIKELIHOOD)

# Color palette options that are publication-friendly
color_options = st.radio(
    "Select color palette for contour plots:",
    ["Blues", "Greys", "Plasma", "RdBu", "YlGnBu"],
    horizontal=True
)

# Function to create a contour plot using Plotly
def plot_contour(x, y, z):
    fig = go.Figure(data=[go.Contour(
        z=z, x=x[0], y=y[:,0], 
        colorscale=color_options,
        ncontours=40,  # Increased from 25 to 40 for more detailed visualization
        contours=dict(
            showlabels=True,
            labelfont=dict(size=16, color='black', family="Times New Roman")
        )
    )])
    fig.update_layout(
        xaxis_title=dict(
            text='P₁',
            font=dict(family="Times New Roman", size=20, color="black")
        ),
        yaxis_title=dict(
            text='P₂',
            font=dict(family="Times New Roman", size=20, color="black")
        ),
        margin=dict(l=0, r=0, b=0, t=20),
        height=800,  # Updated to exactly 800
        width=1046,  # Updated to exactly 1046
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(
            family="Times New Roman",
            size=16,
            color="black"
        )
    )
    
    # Explicitly set axis and tick colors to black
    fig.update_xaxes(
        showline=False, 
        linewidth=1, 
        linecolor='black',
        ticks="outside", 
        tickwidth=1,
        tickcolor='black',
        tickfont=dict(family="Times New Roman", color='black', size=16),
        mirror=True,
        title=dict(
            font=dict(family="Times New Roman", color="black", size=20)
        )
    )
    
    fig.update_yaxes(
        showline=False, 
        linewidth=1, 
        linecolor='black',
        ticks="outside", 
        tickwidth=1,
        tickcolor='black',
        tickfont=dict(family="Times New Roman", color='black', size=16),
        mirror=True,
        title=dict(
            font=dict(family="Times New Roman", color="black", size=20)
        )
    )
    
    return fig

# Product and Geometric Mean of Probabilities
st.subheader("Product and Geometric Mean of Probabilities")
col_prod, col_geo = st.columns(2)

with col_prod:
    st.markdown("**Product of Probabilities**")
    fig_prod = plot_contour(P1, P2, PRODUCT_PROBS)
    st.plotly_chart(fig_prod, use_container_width=True, config={'toImageButtonOptions': {'width': 1046, 'height': 800, 'scale': 4}})

with col_geo:
    st.markdown("**Geometric Mean of Probabilities**")
    fig_geo = plot_contour(P1, P2, GEO_MEAN_PROBS)
    st.plotly_chart(fig_geo, use_container_width=True, config={'toImageButtonOptions': {'width': 1046, 'height': 800, 'scale': 4}})

# BCE Likelihood
st.subheader("BCE Likelihood")
col_bce_likelihood, col_empty1 = st.columns(2)
with col_bce_likelihood:
    st.markdown("**BCE Likelihood**")
    fig_bce_likelihood = plot_contour(P1, P2, BCE_LIKELIHOOD)
    st.plotly_chart(fig_bce_likelihood, use_container_width=True, config={'toImageButtonOptions': {'width': 1046, 'height': 800, 'scale': 4}})

# Any Class Likelihood - Power Mean vs Product
st.subheader("Any Class Likelihood - Power Mean vs Product")
col1, col2 = st.columns(2)

with col1:
    # Power Mean version
    st.markdown("**Any Class Likelihood (Power Mean)**")
    fig_any_likelihood = plot_contour(P1, P2, ANY_CLASS_LIKELIHOOD)
    st.plotly_chart(fig_any_likelihood, use_container_width=True, config={'toImageButtonOptions': {'width': 1046, 'height': 800, 'scale': 4}})

with col2:
    # Product version
    st.markdown("**Any Class Likelihood (Product)**")
    fig_prod_likelihood = plot_contour(P1, P2, PROD_LIKELIHOOD)
    st.plotly_chart(fig_prod_likelihood, use_container_width=True, config={'toImageButtonOptions': {'width': 1046, 'height': 800, 'scale': 4}})

# Modified Likelihood - Power Mean vs Product
st.subheader("Modified Likelihood - Power Mean vs Product")
col3, col4 = st.columns(2)

with col3:
    # Power Mean version
    st.markdown("**Modified Likelihood (Power Mean)**")
    fig_modified_likelihood = plot_contour(P1, P2, MODIFIED_LIKELIHOOD)
    st.plotly_chart(fig_modified_likelihood, use_container_width=True, config={'toImageButtonOptions': {'width': 1046, 'height': 800, 'scale': 4}})

with col4:
    # Product version
    st.markdown("**Modified Likelihood (Product)**")
    fig_modified_prod_likelihood = plot_contour(P1, P2, MODIFIED_PROD_LIKELIHOOD)
    st.plotly_chart(fig_modified_prod_likelihood, use_container_width=True, config={'toImageButtonOptions': {'width': 1046, 'height': 800, 'scale': 4}})

# Loss Functions
st.subheader("Loss Functions")

# BCE Loss
st.subheader("BCE Loss")
col_bce_loss, col_empty2 = st.columns(2)
with col_bce_loss:
    st.markdown("**BCE Loss**")
    fig_bce_loss = plot_contour(P1, P2, BCE_LOSS)
    st.plotly_chart(fig_bce_loss, use_container_width=True, config={'toImageButtonOptions': {'width': 1046, 'height': 800, 'scale': 4}})

# Any Class Loss - Power Mean vs Product
st.subheader("Any Class Loss - Power Mean vs Product")
col5, col6 = st.columns(2)

with col5:
    # Power Mean version
    st.markdown("**Any Class Loss (Power Mean)**")
    fig_any_loss = plot_contour(P1, P2, ANY_CLASS_LOSS)
    st.plotly_chart(fig_any_loss, use_container_width=True, config={'toImageButtonOptions': {'width': 1046, 'height': 800, 'scale': 4}})

with col6:
    # Product version
    st.markdown("**Any Class Loss (Product)**")
    fig_prod_loss = plot_contour(P1, P2, PROD_LOSS)
    st.plotly_chart(fig_prod_loss, use_container_width=True, config={'toImageButtonOptions': {'width': 1046, 'height': 800, 'scale': 4}})

# Modified Loss - Power Mean vs Product
st.subheader("Modified Loss - Power Mean vs Product")
col7, col8 = st.columns(2)

with col7:
    # Power Mean version
    st.markdown("**Modified Loss (Power Mean)**")
    fig_modified_loss = plot_contour(P1, P2, MODIFIED_LOSS)
    st.plotly_chart(fig_modified_loss, use_container_width=True, config={'toImageButtonOptions': {'width': 1046, 'height': 800, 'scale': 4}})

with col8:
    # Product version
    st.markdown("**Modified Loss (Product)**")
    fig_modified_prod_loss = plot_contour(P1, P2, MODIFIED_PROD_LOSS)
    st.plotly_chart(fig_modified_prod_loss, use_container_width=True, config={'toImageButtonOptions': {'width': 1046, 'height': 800, 'scale': 4}})

