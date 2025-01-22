# ![image](https://github.com/yhnx/tymato/blob/main/server/static/logo.png)

Tymato is a web application that leverages deep learning to identify tomato plant diseases from leaf images. It helps farmers and gardeners maintain healthy crops through early disease detection and diagnosis.

## Features

- **Simple Image Upload**: Drag-and-drop or click to upload leaf images
- **Real-time Validation**: Ensures only valid image formats are processed
- **AI-Powered Analysis**: Utilizes deep learning for accurate disease prediction
- **Responsive Design**: Clean, intuitive interface that works across devices
- **Image Preview**: Review uploaded images before submission
- **Instant Results**: Get disease predictions with confidence scores

## Tech Stack

### Frontend
- HTML5
- CSS3
- Bootstrap 5
- JavaScript

### Backend
- Python
- FastAPI
- PyTorch (For Model Training)

### Deep Learning
- RESNET-50 (fine-tuned for tomato disease classification)
- Dataset: [Tomato Disease Multiple Sources Dataset](https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources) on Kaggle

## Installation

1. Clone the repository
```bash
git clone https://github.com/yhnx/tymato.git
cd tymato
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Start the FastAPI server
```bash
python3 server/backend/app.py
```

4. Open `localhost:8000` in your browser



## Usage

1. Visit the Tymato web interface
2. Upload a tomato leaf image using drag-and-drop or file selection
3. Verify the image preview
4. Click "Analyze Image"
5. View the predicted disease and confidence score

## Roadmap

- [ ] In-app camera capture functionality
- [ ] Enhanced mobile responsiveness
- [ ] Historical data tracking
- [ ] Offline detection capability
- [ ] Integration with agricultural management systems


## Dataset

This project uses the `Tomato Disease Multiple Sources` Dataset for tomato diseases, available on Kaggle. The dataset includes:
- 10 different tomato plant disease classes
- Over 25,000 images
- Verified disease labels

[Access the dataset here](https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources)

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Project Link: [https://github.com/yhnx/tymato](https://github.com/your-username/tymato)


---
Made with ❤️ for healthier crops
