from fastapi import status


class PaddingtonError(Exception):
    """Base exception for all domain errors in production_rag."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class NotFoundError(PaddingtonError):
    status_code = status.HTTP_404_NOT_FOUND


class AlreadyExistsError(PaddingtonError):
    status_code = status.HTTP_409_CONFLICT


class ValidationError(PaddingtonError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


class ForbiddenError(PaddingtonError):
    status_code = status.HTTP_403_FORBIDDEN


class UnsupportedFileTypeError(PaddingtonError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


class FileTooLargeError(PaddingtonError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
