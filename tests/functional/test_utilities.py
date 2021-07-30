from src import create_app

def test_home_page():
    flask_app = create_app()
    with flask_app.test_client() as test_client:
        response = test_client.get('/')
        assert response.status_code == 200
        assert b"ok" in response.data
        assert b"Hello, stranger!" in response.data


def test_404():
    flask_app = create_app()
    with flask_app.test_client() as test_client:
        response = test_client.get('/unknown')
        assert response.status_code == 404
        assert b"The requested URL was not found on the server" in response.data
