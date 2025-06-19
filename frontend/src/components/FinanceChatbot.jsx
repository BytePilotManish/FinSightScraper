import React, { useState, useRef, useEffect } from 'react';
import './FinanceChatbot.css';

const FinanceChatbot = () => {
    const [messages, setMessages] = useState([
        { text: "Hello! I'm your finance assistant. Ask me anything about finance, interest rates, bonds, or market trends!", isUser: false }
    ]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!input.trim()) return;

        // Add user message
        const userMessage = { text: input, isUser: true };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        try {
            const response = await fetch('http://localhost:5000/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: input }),
            });

            const data = await response.json();
            
            if (response.ok) {
                // Add bot response
                setMessages(prev => [...prev, { text: data.response, isUser: false }]);
            } else {
                setMessages(prev => [...prev, { 
                    text: 'Sorry, I encountered an error. Please try again.', 
                    isUser: false 
                }]);
            }
        } catch (error) {
            setMessages(prev => [...prev, { 
                text: 'Sorry, I encountered an error. Please try again.', 
                isUser: false 
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="chatbot-container">
            <div className="chatbot-header">
                <h2>Finance Assistant</h2>
            </div>
            
            <div className="chatbot-messages">
                {messages.map((message, index) => (
                    <div 
                        key={index} 
                        className={`message ${message.isUser ? 'user-message' : 'bot-message'}`}
                    >
                        {message.text}
                    </div>
                ))}
                {isLoading && (
                    <div className="loading">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>
            
            <form onSubmit={handleSubmit} className="chatbot-input">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Type your question here..."
                    disabled={isLoading}
                />
                <button type="submit" disabled={isLoading}>
                    Send
                </button>
            </form>
        </div>
    );
};

export default FinanceChatbot; 