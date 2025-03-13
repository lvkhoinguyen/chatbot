// This file contains JavaScript code that adds interactivity to the chatbot application.

document.addEventListener("DOMContentLoaded", function () {
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const chatWindow = document.getElementById("chat-window");

    chatForm.addEventListener("submit", function (event) {
        event.preventDefault();
        const userMessage = chatInput.value.trim();
        if (userMessage) {
            displayMessage("You: " + userMessage);
            chatInput.value = "";
            sendMessageToChatbot(userMessage);
        }
    });

    document.getElementById('send-button').addEventListener('click', sendMessage);
    document.getElementById('user-input').addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    function displayMessage(message) {
        const messageElement = document.createElement("div");
        messageElement.textContent = message;
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight; // Scroll to the bottom
    }

    function sendMessageToChatbot(message) {
        fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question: message })
        })
            .then(response => response.json())
            .then(data => {
                displayMessage("Gemini: " + data.answer);
            })
            .catch(error => {
                console.error("Error:", error);
                displayMessage("Error: Unable to get a response from the chatbot.");
            });
    }

    function sendMessage() {
        const userInput = document.getElementById('user-input');
        const messageText = userInput.value.trim();
        if (messageText === '') return;

        addMessage('user', messageText);
        userInput.value = '';

        // Simulate bot response
        setTimeout(() => {
            addMessage('bot', 'Đang suy nghĩ...');
            setTimeout(() => {
                addMessage('bot', 'Đây là câu trả lời của tôi.');
            }, 2000);
        }, 500);
    }

    function addMessage(sender, text) {
        const messages = document.getElementById('messages');
        const message = document.createElement('div');
        message.classList.add('message', sender);
        message.innerText = text;
        messages.appendChild(message);
        messages.scrollTop = messages.scrollHeight;
    }
});