# Windows PC Migration and Deployment Guide

This document outlines the exact steps required to move this YOLO Computer Vision project to a new Windows PC, set it up, run it, and update the Streamlit web interface.

## 1. Transferring the Project
You can migrate this project to your Windows laptop in two ways:
- **Using a USB Drive or Cloud Storage (Google Drive, Zip file):** Simply copy the entire `project 1` folder to your Windows PC.
- **Using Git / GitHub:** If the project is on GitHub, open **Command Prompt** or **PowerShell** and run:
  ```cmd
  git clone <your-repository-url>
  cd <repository-folder>
  ```

## 2. System Requirements (Prerequisites)
Before running the Python code, make sure you have the following installed on your Windows PC:
- **Python (3.9 to 3.11 recommended):** Download and install from [python.org](https://www.python.org/downloads/windows/). 
  - *IMPORTANT:* During installation, make sure to check the box that says **"Add Python to PATH"**.
- **Camera/Webcam:** Ensure your laptop's camera drivers are working correctly in Windows.

*(Note: The system packages listed in `packages.txt` like `ffmpeg` are generally pre-handled by Windows media foundations or OpenCV, so you don't need to manually install them like on Linux.)*

## 3. Installing Python Dependencies
Once the folder is on your new laptop, follow these steps in **Command Prompt** or **PowerShell**:

1. **Navigate to the project directory** (if you haven't already):
   ```cmd
   cd path\to\your\project 1
   ```
2. **Create a Virtual Environment** (to keep everything organized):
   ```cmd
   python -m venv .venv
   ```
3. **Activate the Virtual Environment:**
   ```cmd
   .venv\Scripts\activate
   ```
   *(You should see `(.venv)` appear at the start of your command line prompt)*

4. **Install the Required Packages:**
   Install everything listed in the `requirements.txt` file by running:
   ```cmd
   pip install -r requirements.txt
   ```
   *Note: If your Windows laptop has an NVIDIA graphics card and you want the YOLO model to run faster, you can install the CUDA-accelerated version of PyTorch by following the instructions commented inside `requirements.txt`.*

## 4. Running the Project
Once the installation is complete, start the web interface by running this command in your terminal (while the `.venv` is still activated):

```cmd
streamlit run ui/streamlit_app.py
```
This will start a local server and automatically open a new tab in your default web browser (usually at `http://localhost:8501`) where you can interact with the app, upload videos, or stream your webcam.

## 5. How to Update the Stream (Streamlit App)
If you need to make changes to the code and see them updated on the Streamlit interface:

### Updating Locally:
1. Make your code changes in `ui/streamlit_app.py` or any other file using a code editor (like VS Code, Notepad++).
2. Save the file.
3. Go back to the browser tab where Streamlit is running. At the top right, Streamlit will show a prompt saying "Source file changed".
4. Click **Rerun** or **Always Rerun**. The stream will automatically update with your new code without you needing to restart the server in the Command Prompt.

### Updating on a Cloud Server (e.g., Streamlit Community Cloud):
If the project is deployed online via Streamlit Community Cloud, you need to push your changes:
1. Make your changes and run these Git commands:
   ```cmd
   git add .
   git commit -m "Updated the stream"
   git push origin main
   ```
2. Once the new code is uploaded to GitHub, Streamlit Cloud will automatically detect the changes, rebuild the app, and update the live website for you.
