class ServiceError(Exception):
    def __init__(self, operation, code, status_code, detail):
        self.operation = operation
        self.code = code
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)
