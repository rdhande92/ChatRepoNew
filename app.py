from flask import Flask, request, jsonify
from get_response import get_response_llm

app = Flask(__name__)

@app.route("/get_response", methods=["POST"])
def get_response():
    """
    Flask route to handle POST requests with a question as input.
    """
    try:
        data = request.get_json()  
        question = data.get("question")  

        if not question:
            return jsonify({"error": "Question field is required"}), 400

        # Get LLM response for the question
        response = get_response_llm(question)

        return jsonify({"question": question, "response": response})  # Return the response as JSON

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)  # Run the Flask app