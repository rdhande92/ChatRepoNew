**ChatRepoNew**

Repochat is a chatbot that talks about GitHub repositories using a powerful AI model. It helps users ask questions, get information, and have useful conversations about repos. This guide explains how to set up and use Repochat on your computer.

**To get started with Repochat, you'll need to follow these installation steps:**

**Step 1:**
    python -m venv repochat_env
    chatrepo_env/bin/activate

**Step 2:**
Clone the Repochat repository and navigate to the project directory.
git clone https://github.com/pnkvalavala/repochat.git
cd ChatRepoNew

**Step 3:**
Install the required Python packages using pip.
pip install -r requirements.txt

**Step 4:**
Open your terminal and run the following command to start the Repochat application:
python app.py

**Step 5:**
Open Postman and use the following URL to test the application.
**URL :-** http://127.0.0.1:5000/get_response
**Body :- **
{
    "question": ""
}

**Response:**
{
"question": "", 
"response": ""
}
















