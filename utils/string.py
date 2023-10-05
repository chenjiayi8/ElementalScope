""" String utilities."""


def get_common_prefix(str1, str2):
    i = 0
    while i < min(len(str1), len(str2)) and str1[i] == str2[i]:
        i += 1
    return str1[:i]
