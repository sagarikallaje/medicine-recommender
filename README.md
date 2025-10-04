# Customer Sentiment Analysis Dashboard

A comprehensive Streamlit-based dashboard for analyzing customer sentiment from product reviews using Hugging Face's DistilBERT model.

## 🚀 Features

- **Data Loading**: Upload your own CSV files or use the provided sample dataset
- **Text Preprocessing**: Automatic cleaning, tokenization, and lemmatization
- **Sentiment Analysis**: Uses Hugging Face's `distilbert-base-uncased-finetuned-sst-2-english` model
- **Interactive Visualizations**: 
  - Pie and bar charts for sentiment distribution
  - Timeline charts (if timestamps are available)
  - Sentiment score histograms
  - Word clouds for each sentiment category
- **Search & Filter**: Find reviews by keywords and sentiment
- **Export Results**: Download processed data and summary reports

## 📋 Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`

## 🛠️ Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

## 📊 Data Format

Your CSV file should contain at least a `review` column. Optional columns:
- `rating`: Numeric rating (1-5)
- `timestamp`: Date/time information for timeline analysis

### Sample Data Format:
```csv
review,rating,timestamp
"This product is amazing!",5,2024-01-15
"Terrible quality, waste of money.",1,2024-01-16
```

## 🌐 Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your GitHub repository to Streamlit Cloud
3. Deploy with the following settings:
   - **Main file path**: `app.py`
   - **Python version**: 3.8+

### Hugging Face Spaces
1. Create a new Space on Hugging Face
2. Upload your files
3. Create a `Dockerfile` or use the Streamlit template
4. Deploy your space

## 📁 Project Structure

```
customer-sentiment-dashboard/
├── app.py                 # Main Streamlit application
├── utils.py              # Utility functions for preprocessing and analysis
├── sample_reviews.csv     # Sample dataset
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔧 Customization

### Adding New Sentiment Models
To use a different Hugging Face model, modify the `model_name` in `utils.py`:

```python
self.model_name = "your-preferred-model-name"
```

### Customizing Visualizations
Modify the `Visualizer` class in `utils.py` to add new chart types or customize existing ones.

### Adding New Preprocessing Steps
Extend the `clean_text` method in the `SentimentAnalyzer` class to add custom text cleaning steps.

## 🐛 Troubleshooting

### Common Issues:

1. **Model Loading Error**: Ensure you have a stable internet connection for downloading the Hugging Face model
2. **Memory Issues**: For large datasets, consider processing reviews in batches
3. **NLTK Data Missing**: The app automatically downloads required NLTK data on first run

### Performance Tips:

- For large datasets (>1000 reviews), consider running sentiment analysis in batches
- Use the search/filter functionality to focus on specific subsets of data
- Export results to avoid re-processing large datasets

## 📈 Model Information

This dashboard uses the `distilbert-base-uncased-finetuned-sst-2-english` model from Hugging Face, which is:
- Based on DistilBERT (a smaller, faster version of BERT)
- Fine-tuned on the Stanford Sentiment Treebank (SST-2) dataset
- Optimized for English text sentiment analysis
- Provides binary classification (Positive/Negative) with confidence scores

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this dashboard.

## 📄 License

This project is open source and available under the MIT License.