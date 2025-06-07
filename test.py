import requests
import json

def test_chat():
    response = requests.post(
        "http://localhost:8000/chat",
        json={
            "messages": [
                {"role": "user", "content": "What is artificial intelligence?"}
            ]
        }
    )
    print("\nChat Response:", json.dumps(response.json(), indent=2))

def test_image_chat():
    with open('test.jpg', 'rb') as f:
        files = {'file': ('test.jpg', f, 'image/jpeg')}
        data = {
            'message': 'Describe this image',
            'conversation': json.dumps([
                {"role": "user", "content": "Hi!"},
                {"role": "assistant", "content": "Hello! How can I help you?"}
            ])
        }
        response = requests.post(
            "http://localhost:8000/chat-with-image",
            files=files,
            data=data
        )
        print("\nImage Chat Response:", json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_chat()
    # test_image_chat()