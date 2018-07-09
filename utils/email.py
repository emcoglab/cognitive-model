"""
===========================
Send emails!
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2018
---------------------------
"""
import json
import logging
import smtplib
from typing import Dict

logger = logging.getLogger(__name__)


class Emailer(object):

    MAX_SUBJECT_LENGTH = 80

    def __init__(self, connection_details_path: str):
        try:
            with open(connection_details_path, mode="r", encoding="utf-8") as connection_details_file:
                self.__connection_details: Dict = json.load(connection_details_file)
        except Exception as ex:
            self.__connection_details: Dict = None
            logger.warning(f"The following exception was thrown and suppressed:")
            logger.warning(str(ex))

    def send_email(self, message: str, recipient_address: str, max_subject_length: int = MAX_SUBJECT_LENGTH):
        """
        Send and email to the target address.
        Catches all exceptions.
        :param message:
        :param recipient_address:
        :param max_subject_length:
        :return:
        """
        try:
            # need to prepend message with appropriate headers
            subject = f"Notification: {message}"
            subject.replace("\n", " ").replace("\r", " ")
            if len(subject) > max_subject_length:
                subject = subject[:Emailer.MAX_SUBJECT_LENGTH - 3] + "..."
            server = smtplib.SMTP(self.__connection_details['smtp-server'], int(self.__connection_details['smtp-port']))
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(self.__connection_details['user-name'], self.__connection_details['password'])
            server.sendmail(self.__connection_details['email-address'],
                            recipient_address,
                            f"From: {self.__connection_details['email-address']}\r\n"
                            f"To: {recipient_address}\r\n"
                            f"Subject: {subject}\r\n\r\n"
                            f"{message}")
        except Exception as ex:
            logger.warning(f"The following exception was thrown and suppressed:")
            logger.warning(str(ex))
