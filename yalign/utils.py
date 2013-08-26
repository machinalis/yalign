"""
Module for miscellaneous functions.
"""


def host_and_page(url):
    url = url.split('//')[1]
    parts = url.split('/')
    host = parts[0]
    page = "/".join(parts[1:])
    return host, '/' + page


def read_from_url(url):
    import httplib
    host, page = host_and_page(url)
    conn = httplib.HTTPConnection(host)
    conn.request("GET", page)
    response = conn.getresponse()
    return response.read()
