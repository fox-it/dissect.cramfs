import pytest

from dissect.cramfs import exception


@pytest.mark.parametrize(
    ("exc", "std"),
    [
        (exception.FileNotFoundError, FileNotFoundError),
        (exception.IsADirectoryError, IsADirectoryError),
        (exception.NotADirectoryError, NotADirectoryError),
    ],
)
def test_filesystem_error_subclass(exc: exception.Error, std: Exception) -> None:
    assert issubclass(exc, std)
    assert isinstance(exc(), std)

    with pytest.raises(std):
        raise exc()
