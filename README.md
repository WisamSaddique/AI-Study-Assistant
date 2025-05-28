# AI Study Assistant

A comprehensive AI-powered study companion that helps students upload, analyze, and interact with their study materials through intelligent document processing, summarization, and question-answering capabilities.

## 🚀 Features

- **Multi-format Document Upload**: Support for PDF, DOCX, and TXT files
- **Intelligent Text Processing**: Advanced text extraction and preprocessing
- **AI-Powered Summarization**: Generate concise summaries with customizable length
- **Question Answering System**: Ask questions about your study materials and get accurate answers
- **Semantic Search**: Find relevant information using advanced embedding techniques
- **Interactive Web Interface**: User-friendly Gradio-based interface
- **Real-time Processing**: Instant document analysis and response generation
- **Document Management**: Add, list, and manage multiple study documents
- **Context-Aware Responses**: Answers include confidence scores and source document references

## 📸 System Screenshots

### 1. Document Upload Interface
<img width="736" alt="1" src="https://github.com/user-attachments/assets/eb6044d5-0035-41de-b78e-0356f1b01320" />

### 2. Document Summarization
<img width="911" alt="2" src="https://github.com/user-attachments/assets/a7f9335c-31bf-43dc-8b17-1f037fe2e740" />
*Generate customizable summaries with adjustable minimum and maximum length parameters*

### 3. Question Answering System
<img width="941" alt="3" src="https://github.com/user-attachments/assets/bacc667a-d4b8-4974-9de0-a4baca6611e7" />
*Ask questions about your uploaded materials and receive detailed, context-aware answers*

## 🛠️ Technologies Used

### AI/ML Libraries
- **Transformers**: Hugging Face transformers for NLP models
- **PyTorch**: Deep learning framework
- **Sentence-Transformers**: Semantic text embeddings
- **FAISS**: Fast similarity search and clustering
- **NLTK**: Natural language processing toolkit

### Document Processing
- **PyPDF2**: PDF text extraction
- **python-docx**: DOCX file processing
- **NumPy**: Numerical computing

### Web Interface
- **Gradio**: Interactive web interface creation
- **HTML/CSS**: Frontend styling

### Pre-trained Models
- **BART-Large-CNN**: Text summarization
- **RoBERTa-Base-Squad2**: Question answering
- **MiniLM-L6-v2**: Sentence embeddings

## 📋 Prerequisites

Before running this project, ensure you have the following installed:

- **Python 3.8 or higher**
- **pip** (Python package installer)
- **CUDA** (optional, for GPU acceleration)

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended for better performance)
- **Storage**: At least 2GB free space for model downloads
- **Internet Connection**: Required for initial model downloads

## 💻 Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/ai-study-assistant.git
cd ai-study-assistant
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv study_assistant_env

# Activate virtual environment
# On Windows:
study_assistant_env\Scripts\activate

# On macOS/Linux:
source study_assistant_env/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install required packages
pip install transformers torch sentence-transformers faiss-cpu numpy PyPDF2 python-docx gradio datasets nltk

# Or install from requirements file (if available)
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
```python
# Run this once to download required NLTK data
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### Step 5: Run the Application
```bash
python main.py
```

The application will start and provide a local URL (typically `http://127.0.0.1:7860`) to access the web interface.

## 📁 Project Structure

```
ai-study-assistant/
│
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── models/                 # Directory for downloaded models
├── uploads/               # Temporary file uploads
├── results/               # Training results (if applicable)
├── README.md              # This file
├── LICENSE                # License file
└── examples/              # Example documents and usage
    ├── sample_document.pdf
    ├── sample_notes.txt
    └── usage_examples.md
```

## 🎯 Usage Guide

### 1. Adding Documents

#### Upload Files
1. Navigate to the **"Add Documents"** tab
2. Click **"Upload Files"** or drag and drop your files
3. Supported formats: PDF, DOCX, TXT
4. Click **"Upload Files"** button to process

#### Add Text Documents
1. Enter a **Document Name**
2. Paste your text content in the **Document Content** field
3. Click **"Add Document"**

### 2. Document Management
- Click **"List Available Documents"** to see all uploaded materials
- View document names and word counts
- Documents are automatically processed for searching and questioning

### 3. Summarization
1. Go to the **"Summarize"** tab
2. Enter the **Document Name** you want to summarize
3. Adjust **Maximum Summary Length** (50-500 words)
4. Set **Minimum Summary Length** (30-200 words)
5. Click **"Summarize Document"**

### 4. Question Answering
1. Navigate to the **"Ask Questions"** tab
2. Type your question in the **"Your Question"** field
3. Click **"Ask"** to get an AI-generated answer
4. Answers include:
   - Direct response to your question
   - Confidence score
   - Source document references

## 🤖 AI Models Used

### Summarization Model
- **Model**: `facebook/bart-large-cnn`
- **Purpose**: Generate concise summaries of documents
- **Features**: Adjustable length, maintains key information

### Question Answering Model
- **Model**: `deepset/roberta-base-squad2`
- **Purpose**: Answer questions based on document content
- **Features**: Confidence scoring, context-aware responses

### Embedding Model
- **Model**: `all-MiniLM-L6-v2`
- **Purpose**: Create semantic embeddings for document search
- **Features**: Fast similarity search, multilingual support

## ⚙️ Configuration

### Model Settings
You can customize model behavior by modifying these parameters in the code:

```python
# Summarization settings
max_length = 150  # Maximum summary length
min_length = 50   # Minimum summary length

# Search settings
top_k = 3         # Number of relevant chunks to retrieve

# Device settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # GPU/CPU usage
```

### Performance Optimization
- **GPU Usage**: Automatically detects and uses CUDA if available
- **Batch Processing**: Handles large documents by chunking
- **Memory Management**: Efficient embedding storage with FAISS

## 🔧 Advanced Features

### Custom Model Training
The system includes functionality for training custom QA models:

```python
# Train a custom model on SQuAD dataset
train_qa_model()
```

### Document Chunking Strategy
- Documents are split into overlapping chunks of 3-5 sentences
- Ensures context preservation across chunk boundaries
- Optimizes retrieval accuracy for question answering

### Semantic Search
- Uses vector embeddings for intelligent document search
- FAISS indexing for fast similarity computations
- Context-aware chunk retrieval

## 🚨 Troubleshooting

### Common Issues

#### Model Download Errors
```bash
# If models fail to download, try:
pip install --upgrade transformers
# Or manually download models
```

#### Memory Issues
```python
# Reduce batch size for lower memory usage
per_device_train_batch_size = 4  # Reduce from 8
```

#### CUDA Errors
```bash
# Install CPU-only versions if GPU issues occur
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu  # Instead of faiss-gpu
```

#### File Upload Issues
- Ensure files are not corrupted
- Check file size limits
- Verify file format compatibility

### Performance Issues
- **Slow Processing**: Consider using GPU acceleration
- **Memory Errors**: Reduce document size or use smaller models
- **Long Load Times**: Models download on first use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update README for new functionality


## 👥 Team & Contributors

- **Project Lead**: Wisam Saddique
- **AI/ML Developer**: Abdul Haseeb Khan


## 🔄 Version History

- **v1.0.0** - Initial release with document upload and Q&A
- **v1.1.0** - Added summarization capabilities
- **v1.2.0** - Improved semantic search and embedding system
- **v1.3.0** - Enhanced UI and batch processing

## 🎉 Acknowledgments

- **Hugging Face**: For providing excellent pre-trained models
- **Gradio Team**: For the intuitive web interface framework
- **OpenAI**: For inspiration in AI-powered educational tools
- **FAISS Contributors**: For efficient similarity search capabilities
- **Community Contributors**: For feedback and feature suggestions

## 🔮 Future Enhancements

### Planned Features
- **Multi-language Support**: Process documents in various languages
- **Advanced Analytics**: Study progress tracking and insights
- **Collaborative Features**: Share documents and notes with peers
- **Mobile App**: Native mobile application
- **API Integration**: RESTful API for third-party integrations
- **Export Features**: Export summaries and Q&A sessions
- **Advanced Search**: Boolean and proximity search operators

### Technical Improvements
- **Model Fine-tuning**: Domain-specific model training
- **Caching System**: Improved response times
- **Database Integration**: Persistent document storage
- **Authentication**: User accounts and access control

---

**Made with ❤️ for students and lifelong learners**

## 📚 Additional Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Gradio Documentation](https://gradio.app/docs/)
- [FAISS Documentation](https://faiss.ai/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

*For the latest updates and releases, please check our [GitHub repository](https://github.com/yourusername/ai-study-assistant).*
