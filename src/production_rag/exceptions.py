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


class InvalidSourceURIError(ValidationError):
    """A source URI is malformed or names a scheme this system does not ingest.

    A subclass of ValidationError (422): the caller sent something we cannot
    parse, which is a request defect rather than an upstream failure.
    """


class SourceFetchError(PaddingtonError):
    """A connector could not retrieve the bytes behind a source URI.

    502 rather than 500: the failure is upstream (DNS, network, a 404 from the
    origin, a blocked address), not a defect in this service.
    """

    status_code = status.HTTP_502_BAD_GATEWAY


class ConnectorNotConfiguredError(PaddingtonError):
    """A connector's scheme is recognised but its credentials are not configured.

    503 rather than 501: the capability exists in the code and is expected to
    work once the deployment supplies credentials — it is unavailable, not
    unimplemented.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
