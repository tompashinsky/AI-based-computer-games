import requests

def test_api():
    url = "https://jsonplaceholder.typicode.com/posts/1"
    response = requests.get(url)
    if response.status_code == 200:
        print("Website is up")
    else:
        print("An error occurred")
