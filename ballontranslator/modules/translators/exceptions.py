class BaseError(Exception):
    """
    base error structure class
    """

    def __init__(self, val, message):
        """
        @param val: actual value
        @param message: message shown to the user
        """
        self.val = val
        self.message = message
        super().__init__()

    def __str__(self):
        return "{} --> {}".format(self.val, self.message)


class InvalidSourceOrTargetLanguage(BaseError):
    """
    exception thrown if the user enters an invalid payload
    """

    def __init__(self,
                 val,
                 message="source and target language can't be the same"):
        super(InvalidSourceOrTargetLanguage, self).__init__(val, message)


class TranslatorSetupFailure(Exception):
    pass

class MissingTranslatorParams(Exception):
    pass

class TranslatorNotValid(Exception):
    pass
