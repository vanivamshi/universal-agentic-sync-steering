from src.auth.login import authenticate, login


def test_authenticate_ok():
    assert authenticate("a@b.com")


def test_login_missing_email_raises():
    try:
        login({})
        raised = False
    except KeyError:
        raised = True
    assert raised
