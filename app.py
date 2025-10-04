import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
from utils import SentimentAnalyzer, Visualizer, filter_reviews, get_sample_reviews, calculate_metrics, load_sample_data

# Page configuration
st.set_page_config(
    page_title="Customer Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sentiment-positive {
        color: #2E8B57;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #DC143C;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #4682B4;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = SentimentAnalyzer()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'df' not in st.session_state:
    st.session_state.df = None

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📋 Dashboard Controls")
    
    # Data loading section
    st.sidebar.header("📁 Data Loading")
    
    # Option to use sample data or upload
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Use Sample Data", "Upload Your Own CSV"]
    )
    
    if data_option == "Use Sample Data":
        if st.sidebar.button("Load Sample Data"):
            with st.spinner("Loading sample data..."):
                st.session_state.df = load_sample_data()
                if st.session_state.df is not None:
                    st.session_state.data_processed = False
                    st.sidebar.success("Sample data loaded successfully!")
                else:
                    st.sidebar.error("Failed to load sample data.")
    
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV file should contain a 'review' column"
        )
        
        if uploaded_file is not None:
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.data_processed = False
                st.sidebar.success("File uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error uploading file: {e}")
    
    # Check if data is loaded
    if st.session_state.df is None:
        st.info("👆 Please load sample data or upload a CSV file to get started!")
        return
    
    # Display data info
    st.sidebar.header("📊 Data Information")
    st.sidebar.write(f"**Total Reviews:** {len(st.session_state.df)}")
    st.sidebar.write(f"**Columns:** {list(st.session_state.df.columns)}")
    
    # Check if review column exists
    if 'review' not in st.session_state.df.columns:
        st.error("❌ The uploaded CSV must contain a 'review' column!")
        return
    
    # Sentiment Analysis Section
    st.sidebar.header("🔍 Sentiment Analysis")
    
    if st.sidebar.button("🚀 Run Sentiment Analysis"):
        with st.spinner("Loading sentiment analysis model..."):
            if st.session_state.analyzer.load_model():
                st.sidebar.success("Model loaded successfully!")
                
                with st.spinner("Analyzing sentiments..."):
                    st.session_state.df = st.session_state.analyzer.process_reviews(st.session_state.df)
                    st.session_state.data_processed = True
                    st.sidebar.success("Sentiment analysis completed!")
            else:
                st.sidebar.error("Failed to load sentiment analysis model!")
    
    # Check if data is processed
    if not st.session_state.data_processed:
        st.info("👆 Please run sentiment analysis to see the dashboard!")
        return
    
    # Main dashboard content
    st.header("📈 Sentiment Analysis Results")
    
    # Calculate metrics
    metrics = calculate_metrics(st.session_state.df)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", metrics['total_reviews'])
    
    with col2:
        st.metric(
            "Positive Reviews", 
            f"{metrics['positive_count']} ({metrics['positive_percentage']:.1f}%)",
            delta=f"{metrics['positive_percentage']:.1f}%"
        )
    
    with col3:
        st.metric(
            "Negative Reviews", 
            f"{metrics['negative_count']} ({metrics['negative_percentage']:.1f}%)",
            delta=f"{metrics['negative_percentage']:.1f}%"
        )
    
    with col4:
        st.metric(
            "Neutral Reviews", 
            f"{metrics['neutral_count']} ({metrics['neutral_percentage']:.1f}%)",
            delta=f"{metrics['neutral_percentage']:.1f}%"
        )
    
    # Charts section
    st.header("📊 Visualizations")
    
    # Chart type selection
    chart_type = st.selectbox("Select chart type:", ["Pie Chart", "Bar Chart"])
    
    # Sentiment distribution chart
    chart_type_lower = chart_type.lower().replace(" ", "_")
    fig_dist = st.session_state.visualizer.plot_sentiment_distribution(
        st.session_state.df, 
        chart_type_lower
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Timeline chart (if timestamp column exists)
    if 'timestamp' in st.session_state.df.columns:
        st.subheader("📅 Sentiment Timeline")
        fig_timeline = st.session_state.visualizer.plot_timeline(st.session_state.df)
        if fig_timeline:
            st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("Timeline chart could not be generated. Please check your timestamp column format.")
    
    # Sentiment score distribution
    st.subheader("📊 Sentiment Score Distribution")
    fig_scores = st.session_state.visualizer.plot_sentiment_scores(st.session_state.df)
    st.plotly_chart(fig_scores, use_container_width=True)
    
    # Word clouds section
    st.header("☁️ Word Clouds")
    
    col1, col2, col3 = st.columns(3)
    
    sentiments = ['Positive', 'Negative', 'Neutral']
    
    for i, sentiment in enumerate(sentiments):
        with [col1, col2, col3][i]:
            st.subheader(f"{sentiment} Reviews")
            
            # Create word cloud
            wordcloud = st.session_state.visualizer.create_wordcloud(
                st.session_state.df, 
                sentiment
            )
            
            if wordcloud:
                # Convert wordcloud to image
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                
                # Convert to base64 for display
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
                img_buffer.seek(0)
                img_str = base64.b64encode(img_buffer.getvalue()).decode()
                
                st.image(f"data:image/png;base64,{img_str}", use_column_width=True)
                plt.close()
            else:
                st.info(f"No {sentiment.lower()} reviews found.")
    
    # Sample reviews section
    st.header("📝 Sample Reviews by Sentiment")
    
    for sentiment in sentiments:
        with st.expander(f"🔍 {sentiment} Reviews"):
            sample_reviews = get_sample_reviews(st.session_state.df, sentiment, 5)
            if sample_reviews:
                for i, review in enumerate(sample_reviews, 1):
                    st.write(f"**{i}.** {review}")
            else:
                st.info(f"No {sentiment.lower()} reviews found.")
    
    # Search and filter section
    st.header("🔍 Search & Filter Reviews")
    
    col1, col2 = st.columns(2)
    
    with col1:
        search_keyword = st.text_input("Search reviews by keyword:")
    
    with col2:
        sentiment_filter = st.selectbox(
            "Filter by sentiment:",
            ["All", "Positive", "Negative", "Neutral"]
        )
    
    # Apply filters
    if search_keyword or sentiment_filter != "All":
        filtered_df = filter_reviews(
            st.session_state.df, 
            search_keyword, 
            sentiment_filter if sentiment_filter != "All" else None
        )
        
        st.subheader(f"🔍 Filtered Results ({len(filtered_df)} reviews)")
        
        if len(filtered_df) > 0:
            # Display filtered reviews
            for idx, row in filtered_df.head(10).iterrows():
                sentiment_class = f"sentiment-{row['sentiment'].lower()}"
                st.markdown(f"""
                <div class="metric-card">
                    <span class="{sentiment_class}">{row['sentiment']}</span> 
                    (Score: {row['sentiment_score']:.3f})<br>
                    {row['review']}
                </div>
                """, unsafe_allow_html=True)
                st.write("---")
            
            if len(filtered_df) > 10:
                st.info(f"Showing first 10 results out of {len(filtered_df)} filtered reviews.")
        else:
            st.info("No reviews match your search criteria.")
    
    # Data export section
    st.header("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Processed Data"):
            csv = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Download Summary Report"):
            summary_data = {
                'Metric': ['Total Reviews', 'Positive Reviews', 'Negative Reviews', 'Neutral Reviews', 
                          'Positive Percentage', 'Negative Percentage', 'Neutral Percentage', 'Average Score'],
                'Value': [metrics['total_reviews'], metrics['positive_count'], metrics['negative_count'], 
                         metrics['neutral_count'], f"{metrics['positive_percentage']:.2f}%", 
                         f"{metrics['negative_percentage']:.2f}%", f"{metrics['neutral_percentage']:.2f}%", 
                         f"{metrics['average_score']:.3f}"]
            }
            summary_df = pd.DataFrame(summary_data)
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download Summary",
                data=csv,
                file_name="sentiment_summary.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📊 Customer Sentiment Analysis Dashboard | Built with Streamlit & Hugging Face Transformers</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()