# restaurant-chatbot
AI NLP system for a restaurant chatbot

This application uses natural language processing and AI to act as the operator of a pizza restaurant's phone line. The main functionality includes:
- Taking a customer's order
- Taking a sit-down booking and saving it to a CSV
- Making small talk with customers
- Answering FAQs about the restaurant
- Providing appropriate responses when an answer cannot be found, based on what kind of question the user asked

The AI used in this application allows it to categorise queries made by the user, followed by using similarity metrics to match it to appropriate responses. This allows it to respond even if the answer to the user's question is not stored in the database.

Given more time, I would have liked to make these changes:
 - Update the booking system to automatically amend clashes based on the restaurant's size
 - Provide a system for order tracking, where the status of the order can be changed by workers and checked by the user
 - Implement machine learning to grow the database of queries and responses. i.e. if a query did not match closely to the response given, but the response was still correct, add the query and response to the database to improve future accuracy
