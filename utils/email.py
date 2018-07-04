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

from ...preferences import Preferences

logger = logging.getLogger(__name__)

MAX_SUBJECT_LENGTH = 80


def send_email(message: str, recipient_address: str, max_subject_length: int = MAX_SUBJECT_LENGTH):
    """
    Send and email to the target address.
    Catches all exceptions.
    :param message:
    :param recipient_address:
    :param max_subject_length: 
    :return:
    """
    try:
        with open(Preferences.email_connection_details_path, mode="r", encoding="utf-8") as connection_details_file:
            connection_details: Dict = json.load(connection_details_file)
        # need to prepend message with appropriate headers
        subject = f"Notification: {message}"
        subject.replace("\n", " ").replace("\r", " ")
        if len(subject) > max_subject_length:
            subject = subject[:MAX_SUBJECT_LENGTH - 3] + "..."
        server = smtplib.SMTP(connection_details['smtp-server'], int(connection_details['smtp-port']))
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(connection_details['user-name'], connection_details['password'])
        server.sendmail(connection_details['email-address'],
                        recipient_address,
                        f"From: {connection_details['email-address']}\r\n"
                        f"To: {recipient_address}\r\n"
                        f"Subject: {subject}\r\n\r\n"
                        f"{message}")
    except Exception as ex:
        logger.warning(f"The following exception was thrown and suppressed:")
        logger.warning(str(ex))


if __name__ == '__main__':
    # send a test email
    send_email("Test email!", Preferences.target_email_address)
