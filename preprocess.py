import regex as re
from fastai.text import *

"""
Handles Preprocessing Tags
"""


def create_user_tags(instring):
    """
    Masks a string with `xxUSER` tags for Twitter-style mentions, IP Addresses, <USER> tags.
    :param instring: The input string.
    :return: Returns those masked version of the string.
    """
    pattern = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|<user>|<USER>|@[A-Za-z0-9_-]*"
    return re.sub(pattern, "xxuser", instring)


def preprocess(strings, **kwargs):
    """
    Inputs a list of strings, returns a list of tokens instead.
    :param strings: List of strings.
    :return: Returns list of lists of tokens.
    """
    text_pre_rules = [
        fix_html,
        replace_rep,
        replace_wrep,
        spec_add_spaces,
        rm_useless_spaces,
        create_user_tags,
    ]
    text_post_rules = [replace_all_caps, deal_caps]
    tokenizer = Tokenizer(
        pre_rules=text_pre_rules, post_rules=text_post_rules, **kwargs
    )
    return tokenizer.process_all(strings)

