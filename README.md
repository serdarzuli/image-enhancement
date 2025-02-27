# **Image Enhancement API**  

## 📌 **Project Overview**  
Image Enhancement is an API designed to enhance image quality using **CodeFormer** for face restoration and **Azure Blob Storage** for secure image storage. This project processes images, improves their quality, and provides users with a downloadable URL for the enhanced image.  

## 🚀 **Features**  
- ✅ **Image Quality Enhancement** – Uses **CodeFormer** to improve face details.  
- ✅ **Background & Face Upsampling** – Enhances overall image clarity.  
- ✅ **Azure Blob Storage Integration** – Securely uploads processed images.  
- ✅ **REST API Support** – Provides an easy-to-use API for image processing.  
- ✅ **Asynchronous Processing** – Efficient handling of image requests.  

## 🛠 **Installation**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/serdarzuli/image-enhancement.git
cd image-enhancement
```

### **2. Create a Virtual Environment & Install Dependencies**  
```bash
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows

pip install -r requirements.txt
```

### **3. Set Up Environment Variables**  
Create a `.env` file in the project root and add the following:  
```
CONNECTING_STRING=your_azure_storage_connection_string
CONTAINER_NAME=your_container_name
```

## 🔧 **Usage**  
Run the API using Flask:  
```bash
python api.py
```

To process an image, send a **POST request** with an image URL:  
```python
import requests

url = "http://localhost:5000/process"
data = {"image_url": "https://example.com/sample.jpg"}

response = requests.post(url, json=data)
print(response.json())  # Returns the processed image URL
```

## 🤦️ **Running Tests**  
Ensure all tests pass before deployment:  
```bash
pytest -v
```

## 📝 **License**  
This project is licensed under the **MIT License**.

### 🛡️ **Acknowledgment**  
This project uses **[CodeFormer](https://github.com/sczhou/CodeFormer)**, which is licensed under the **S-Lab License 1.0**.  
For more details, please refer to the **official CodeFormer repository**.

